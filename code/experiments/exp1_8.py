"""Exp 1.8 — Module curvature disparity on REAL data, at initialization.

Theorem 1 and Definition (module curvature disparity) are both stated at
RANDOM INITIALIZATION, so this experiment needs no trained checkpoint: we
instantiate the production BlockViT-v2 architecture, feed it real FTIR/QCL
tissue cores, and power-iterate the true Gauss-Newton blocks.

Two testable halves, not one:
    (a) lambda_max(G_phiphi) grows linearly in the spatial width M
    (b) lambda_max(G_thetatheta) stays CAPPED, independent of M
        (Lemma opcap: the cap contains no C_g)
so we sweep M = hidden_dim and report both, plus their ratio D_curv.
Half (b) is the sharper falsification target: the cap is proved, so a
lambda_theta that climbs with M would contradict the theory outright.

Measures GGN blocks (code/hessian/ggn.py), NOT loss-Hessian blocks --
the theory is stated on G = J^T H_tau J and the two differ by the
functional-Hessian term (audit A24).

Usage:
    python code/experiments/exp1_8.py --widths 48 96 192 384 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

THEORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THEORY_ROOT / "code"))
SIDE = Path("/home/u37314kd/Projects/spectral_tokenization/side_project")
sys.path.insert(0, str(SIDE))

from hessian.ggn import top_eigenvalue_ggn_block  # noqa: E402
from models.blockvit_v2 import BlockViTv2         # noqa: E402
from data.core_dataset import CoreDataset         # noqa: E402


def freeze_bn_running_stats(model: nn.Module) -> None:
    """Keep batch-norm batch statistics but stop running-stat drift.

    We measure in train() mode because that is what the optimizer sees at
    step 0. Setting momentum to 0 makes repeated forward passes across the
    sweep deterministic rather than slowly shifting the running estimates.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0.0


def build_model(width: int, seed: int, args) -> nn.Module:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = BlockViTv2(
        in_channels=3, num_classes=args.num_classes, num_spectral=args.num_spectral,
        reduce_dim=args.reduce_dim, patch_tok_size=args.patch_tok_size,
        hidden_dim=width, num_layers=args.num_layers, num_heads=args.num_heads,
        mlp_ratio=4.0, dropout=0.0, spatial_size=args.spatial_size,
        spectral_norm=not getattr(args, "no_spectral_norm", False),
    ).to(args.device)
    model.train()
    freeze_bn_running_stats(model)
    return model


def split_params(model: nn.Module):
    """Theory's two modules: f_theta = spectral_reduce, g_phi = everything else."""
    theta, phi = [], []
    for name, p in model.named_parameters():
        (theta if name.startswith("spectral_reduce") else phi).append(p)
    return theta, phi


def load_real_batches(args):
    """Load the densest real cores of a split as fixed measurement batches.

    Returns ``(batches, sigma_stats)``: ``batches = [(X, Y, n_valid), ...]``
    on ``args.device``; ``sigma_stats[i]`` holds the raw input-Gram
    concentration of batch i over valid pixels (lam_max, trace, effective
    rank = trace/lam_max) — the quantity that sets Lemma opcap's cap.
    """
    split_file = Path(args.data_dir) / f"splits_fold{args.fold}.json"
    if not split_file.exists():
        raise SystemExit(f"missing split file: {split_file}")

    # CoreDataset eagerly preloads every core in the split (~0.4 GB each), so
    # asking for the full train split to measure a few batches would cost
    # ~50 GB of I/O. Hand it a trimmed split file instead: same class, same
    # preprocessing, only the cores we might touch — then keep the densest.
    n_cores_needed = args.n_batches * args.batch_size
    full_split = json.loads(split_file.read_text())
    core_names = full_split[args.split][: n_cores_needed * args.core_oversample]
    trimmed = {k: (core_names if k == args.split else []) for k in full_split}
    tmp_split = Path(args.tmp_dir) / f"_split_{args.dataset_name}_f{args.fold}_{args.split}.json"
    tmp_split.parent.mkdir(parents=True, exist_ok=True)
    tmp_split.write_text(json.dumps(trimmed))
    print(f"[data] using {len(core_names)} of {len(full_split[args.split])} "
          f"{args.split} cores: {core_names}", flush=True)

    ds = CoreDataset(
        data_dir=args.data_dir, split_file=str(tmp_split), split=args.split,
        spatial_size=args.spatial_size, augment=False,
    )
    print(f"[data] {args.dataset_name} fold{args.fold} {args.split}: {len(ds)} cores "
          f"@ {args.spatial_size}x{args.spatial_size}", flush=True)

    # Rank the loaded cores by labelled-pixel count, keep the densest.
    scored = []
    for i in range(len(ds)):
        _, y = ds[i]
        scored.append((int((torch.as_tensor(y) != 255).sum().item()), i))
    scored.sort(reverse=True)
    chosen = [i for n, i in scored if n > 0][: args.n_batches * args.batch_size]
    if not chosen:
        raise SystemExit("no usable cores (all pixels ignored)")

    batches = []
    for b in range(0, len(chosen), args.batch_size):
        idxs = chosen[b: b + args.batch_size]
        xs, ys = zip(*(ds[i] for i in idxs))
        X = torch.stack([torch.as_tensor(v) for v in xs]).float().to(args.device)
        Y = torch.stack([torch.as_tensor(v) for v in ys]).long().to(args.device)
        n_valid = int((Y != 255).sum().item())
        if n_valid == 0:
            continue
        if getattr(args, "center_inputs", False):
            keep = (Y != 255)
            mu = (X * keep[:, None, :, :, None]).sum(dim=(0, 2, 3)) / n_valid
            X = X - mu[None, :, None, None, :]
        batches.append((X, Y, n_valid))
    print(f"[data] {len(batches)} batches, valid px/batch: "
          f"{[b[2] for b in batches]} "
          f"(valid frac {[f'{b[2]/b[1][0].numel():.1%}' for b in batches]})",
          flush=True)

    # Raw input-Gram concentration on valid pixels: how much of the input's
    # energy sits in its single top direction (breast: eff. rank ~1.07/942).
    sigma_stats = []
    for X, Y, n_valid in batches:
        B, C, H, W, S = X.shape
        keep = (Y != 255).reshape(-1)
        with torch.no_grad():
            xv = X.permute(0, 2, 3, 1, 4).reshape(B * H * W, C * S)[keep]
            Sig = (xv.T @ xv) / xv.shape[0]
            lam1 = float(torch.linalg.eigvalsh(Sig.double())[-1].item())
            tr = float(Sig.diagonal().sum().item())
        sigma_stats.append(dict(
            lam_max_sigma=lam1, trace_sigma=tr, eff_rank_sigma=tr / lam1,
        ))
        print(f"[sigma] batch {len(sigma_stats)-1}: lam_max={lam1:.4e} "
              f"trace={tr:.4e} eff_rank={tr/lam1:.2f} of {C*S}", flush=True)
    return batches, sigma_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/hdd2/u37314kd/data_breast_v2_pca23")
    ap.add_argument("--dataset_name", default="breast")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--widths", type=int, nargs="+", default=[48, 96, 192, 384])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_batches", type=int, default=4, help="real cores per config")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--n_iter", type=int, default=25, help="power iterations")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--spatial_size", type=int, default=336)
    ap.add_argument("--reduce_dim", type=int, default=64)
    ap.add_argument("--num_spectral", type=int, default=314)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--patch_tok_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--core_oversample", type=int, default=6,
                    help="load this many times n_batches cores, keep densest")
    ap.add_argument("--no_spectral_norm", action="store_true",
                    help="build the model WITHOUT the internal per-wavenumber "
                         "BatchNorm (wn_norm), which otherwise re-standardizes "
                         "the input and absorbs any external centering")
    ap.add_argument("--center_inputs", action="store_true",
                    help="subtract the valid-pixel mean spectrum (per channel x "
                         "wavenumber) from the whole input tensor before "
                         "measurement -- tests the centering-restores-disparity "
                         "prediction (E3e)")
    ap.add_argument("--tmp_dir", default="/tmp/claude-1008/spectral_shortcut_scratch")
    ap.add_argument("--out", default=str(THEORY_ROOT / "results" / "exp1_8_real_dcurv.csv"))
    args = ap.parse_args()

    for w in args.widths:
        if w % args.num_heads != 0:
            raise SystemExit(f"width {w} not divisible by num_heads {args.num_heads}")

    # Fixed batches, identical across every (width, seed) config, so the
    # sweep varies only the architecture.
    batches, sigma_stats = load_real_batches(args)

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for width in args.widths:
        for seed in args.seeds:
            model = build_model(width, seed, args)
            theta, phi = split_params(model)
            Cf = sum(p.numel() for p in theta)
            Cg = sum(p.numel() for p in phi)
            for bi, (X, Y, n_valid) in enumerate(batches):
                t0 = time.time()
                torch.cuda.reset_peak_memory_stats()
                lam_phi = top_eigenvalue_ggn_block(
                    model, X, Y, phi, n_iter=args.n_iter, tol=args.tol, seed=seed
                )
                lam_th = top_eigenvalue_ggn_block(
                    model, X, Y, theta, n_iter=args.n_iter, tol=args.tol, seed=seed
                )
                peak = torch.cuda.max_memory_allocated() / 1e9
                dcurv = lam_phi / lam_th if lam_th > 0 else float("inf")
                rows.append(dict(
                    dataset=args.dataset_name, fold=args.fold, split=args.split,
                    width=width, seed=seed, batch=bi, n_valid_px=n_valid,
                    lam_phi=lam_phi, lam_theta=lam_th, D_curv=dcurv,
                    C_f=Cf, C_g=Cg, ratio_Cg_Cf=Cg / Cf,
                    reduce_dim=args.reduce_dim, spatial_size=args.spatial_size,
                    peak_gb=peak, secs=time.time() - t0,
                    **sigma_stats[bi],
                ))
                print(f"  M={width:4d} seed={seed} b={bi} | "
                      f"lam_phi={lam_phi:.4e} lam_theta={lam_th:.4e} "
                      f"D_curv={dcurv:8.1f} | {peak:.1f}GB {time.time()-t0:.0f}s",
                      flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
            del model, theta, phi
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[write] {out_path}  ({len(df)} rows)")

    # --- summary: does the width-linear law hold on real data? -------------
    g = df.groupby("width").agg(
        lam_phi=("lam_phi", "mean"), lam_phi_sd=("lam_phi", "std"),
        lam_theta=("lam_theta", "mean"), lam_theta_sd=("lam_theta", "std"),
        D_curv=("D_curv", "mean"), D_curv_sd=("D_curv", "std"),
    ).reset_index()
    print("\n" + g.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    if len(g) >= 2:
        lw = np.log(g["width"].to_numpy(float))
        for col, pred in (("lam_phi", "1.0 (width-linear)"),
                          ("lam_theta", "0.0 (capped)"),
                          ("D_curv", "1.0 (Theorem 1)")):
            slope, intercept = np.polyfit(lw, np.log(g[col].to_numpy(float)), 1)
            print(f"  log-log slope  {col:9s} vs M = {slope:+.3f}   predicted {pred}")

    summary = {
        "dataset": args.dataset_name, "fold": args.fold,
        "widths": args.widths, "seeds": args.seeds,
        "n_batches": len(batches), "spatial_size": args.spatial_size,
        "per_width": g.to_dict(orient="records"),
    }
    sp = out_path.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2))
    print(f"[write] {sp}")


if __name__ == "__main__":
    main()
