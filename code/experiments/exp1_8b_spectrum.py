"""Exp 1.8b — Spectrum shape of the spectral GGN block on real data.

Exp 1.8 found D_curv < 1 on real tissue: the spectral block's TOP
curvature exceeds the spatial block's. The diagnosis was that Sigma_X is
effectively rank-1 (all tissue spectra share one shape), so the spectral
block should have ONE huge curvature direction — the mean-spectrum
reader — and a near-flat remainder, with the discriminative chemistry in
the flat part. This script measures that claim directly:

  1. Top-k eigenvalues of G_thetatheta (Lanczos) — is the spectrum a
     single spike over a collapsed tail?
  2. Concentration lambda_1 / tr(G_thetatheta) (Hutchinson trace) — the
     curvature analogue of Sigma_X's lam_max/trace = 0.937.
  3. Alignment: does the top eigenvector's proj-weight component factor
     as (something) x v_data^T, with v_data the top eigenvector of the
     post-BatchNorm input Gram (the mean-spectrum direction)?
  4. THE key number: max curvature available along class-contrast
     directions d = mu_classA - mu_classB (component orthogonal to
     v_data), via the exact 64x64 restricted block max_u R(u (x) d).
     Compare against the same quantity for v_data itself.
  5. Top-k of G_phiphi for the side-by-side figure.

Usage:
    python code/experiments/exp1_8b_spectrum.py --seeds 0 1 2
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

THEORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THEORY_ROOT / "code"))
sys.path.insert(0, str(THEORY_ROOT / "code" / "experiments"))
sys.path.insert(0, "/home/u37314kd/Projects/spectral_tokenization/side_project")

from hessian.lanczos import GGNBlockOperator, lanczos_topk   # noqa: E402
from exp1_8 import build_model, split_params, load_real_batches  # noqa: E402

CLASS_NAMES = {0: "CancerEpi", 1: "CAS", 2: "NormalStroma", 3: "NormalEpi"}


def post_bn_inputs(model, X, Y):
    """The matrix the proj conv actually reads: wn_norm output, valid px.

    Returns (xv, keep): xv (N_valid, C*S) in the exact train-mode BN state
    used during the GGN forward (batch statistics over the full padded
    core, matching what the measurement sees).
    """
    sr = model.spectral_reduce
    B, C, H, W, S = X.shape
    with torch.no_grad():
        xf = X.permute(0, 2, 3, 1, 4).reshape(B * H * W * C, S)
        xf = sr.wn_norm(xf) if sr.spectral_norm else xf
        xall = xf.reshape(B, H, W, C, S).reshape(B * H * W, C * S)
    keep = (Y != 255).reshape(-1)
    return xall[keep], keep


def data_directions(xv, topn=3):
    """Top eigenvectors + eigenvalues of the (valid-pixel) input Gram."""
    Sig = (xv.T @ xv) / xv.shape[0]
    ev, U = torch.linalg.eigh(Sig.double())
    vals = ev.flip(0)[:topn]
    vecs = U.flip(1)[:, :topn].T.to(xv.dtype)     # (topn, C*S)
    tr = float(Sig.diagonal().sum().item())
    return vals, vecs, tr


def class_contrasts(xv, Y, keep, pairs):
    """Unit contrast directions mu_a - mu_b in post-BN input space."""
    yv = Y.reshape(-1)[keep]
    means, counts = {}, {}
    for c in torch.unique(yv).tolist():
        sel = yv == c
        counts[c] = int(sel.sum().item())
        means[c] = xv[sel].mean(0)
    out = {}
    for a, b in pairs:
        if counts.get(a, 0) >= 100 and counts.get(b, 0) >= 100:
            d = means[a] - means[b]
            out[(a, b)] = d / d.norm()
    return out, counts


def direction_in_params(op, proj_weight, u_vec, d_vec):
    """Flatten the rank-one direction u (x) d into proj.weight coordinates."""
    flat = torch.zeros(op.dim, device=d_vec.device, dtype=d_vec.dtype)
    ofs = 0
    for p in op.params:
        if p is proj_weight:
            W = torch.einsum("k,s->ks", u_vec, d_vec).reshape(p.shape)
            flat[ofs: ofs + p.numel()] = W.reshape(-1)
        ofs += p.numel()
    return flat


def restricted_block_lambda_max(op, proj_weight, d_vec, K):
    """max_u R(u (x) d): exact lambda_max of the K x K restricted GGN block.

    M[i, j] = (e_i (x) d)^T G (e_j (x) d), assembled from K matvecs.
    This is the largest curvature obtainable along ANY head-combination
    of the single input direction d — the exact analogue of the witness
    construction, aimed at d instead of the top feature direction.
    """
    basis = []
    cols = []
    for j in range(K):
        u = torch.zeros(K, device=d_vec.device, dtype=d_vec.dtype)
        u[j] = 1.0
        basis.append(direction_in_params(op, proj_weight, u, d_vec))
    for j in range(K):
        cols.append(op.matvec_flat(basis[j]))
    M = torch.zeros(K, K, dtype=torch.float64)
    for i in range(K):
        for j in range(K):
            M[i, j] = float((basis[i] @ cols[j]).item())
    M = 0.5 * (M + M.T)
    return float(torch.linalg.eigvalsh(M)[-1].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/hdd2/u37314kd/data_breast_v2_pca23")
    ap.add_argument("--dataset_name", default="breast")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--core_oversample", type=int, default=6)
    ap.add_argument("--m_theta", type=int, default=80)
    ap.add_argument("--k_theta", type=int, default=40)
    ap.add_argument("--m_phi", type=int, default=30)
    ap.add_argument("--k_phi", type=int, default=15)
    ap.add_argument("--n_probes", type=int, default=20)
    ap.add_argument("--restricted_seeds", type=int, nargs="+", default=[0, 1],
                    help="seeds on which to run the 64-matvec restricted blocks")
    ap.add_argument("--spatial_size", type=int, default=336)
    ap.add_argument("--reduce_dim", type=int, default=64)
    ap.add_argument("--num_spectral", type=int, default=314)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--patch_tok_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tmp_dir", default="/tmp/claude-1008/spectral_shortcut_scratch")
    ap.add_argument("--out_prefix", default=str(THEORY_ROOT / "results" / "exp1_8b"))
    args = ap.parse_args()

    batches, sigma_stats = load_real_batches(args)

    spec_rows, sum_rows = [], []
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    PAIRS = [(0, 2), (0, 1), (1, 2), (3, 2)]

    for seed in args.seeds:
        model = build_model(args.width, seed, args)
        theta, phi = split_params(model)
        proj_weight = model.spectral_reduce.proj.weight
        K = args.reduce_dim

        for bi, (X, Y, n_valid) in enumerate(batches):
            t0 = time.time()
            # ---- data directions in the space proj actually reads ----
            xv, keep = post_bn_inputs(model, X, Y)
            dvals, dvecs, dtr = data_directions(xv, topn=3)
            contrasts, counts = class_contrasts(xv, Y, keep, PAIRS)

            # ---- theta block: top-k spectrum + vectors + trace ----
            op_t = GGNBlockOperator(model, X, Y, theta)
            vals_t, vecs_t, info_t = lanczos_topk(
                op_t, m=args.m_theta, k=args.k_theta, seed=seed, n_vectors=3)
            tr_t, tr_t_se = op_t.hutchinson_trace(args.n_probes, seed=seed)

            # alignment of top eigenvectors with data directions
            aligns = {}
            ofs = 0
            sl = None
            for p in op_t.params:
                if p is proj_weight:
                    sl = (ofs, ofs + p.numel())
                ofs += p.numel()
            for i in range(vecs_t.shape[0]):
                Vfull = vecs_t[i]
                Vp = Vfull[sl[0]: sl[1]].reshape(K, -1)      # (K, C*S)
                frac_proj = float((Vp.norm() / Vfull.norm()).item()) ** 2
                for j in range(dvecs.shape[0]):
                    a = float((Vp @ dvecs[j]).norm().item() / max(Vp.norm().item(), 1e-30))
                    aligns[f"align_v{i+1}_data{j+1}"] = a
                aligns[f"projfrac_v{i+1}"] = frac_proj

            # ---- phi block: top-k spectrum ----
            op_p = GGNBlockOperator(model, X, Y, phi)
            vals_p, _, info_p = lanczos_topk(
                op_p, m=args.m_phi, k=args.k_phi, seed=seed, n_vectors=0)
            tr_p, tr_p_se = op_p.hutchinson_trace(args.n_probes, seed=seed)

            # ---- restricted blocks: curvature along chosen input directions
            restricted = {}
            if seed in args.restricted_seeds:
                restricted["lam_along_vdata"] = restricted_block_lambda_max(
                    op_t, proj_weight, dvecs[0], K)
                for (a, b), d in contrasts.items():
                    d_perp = d - (d @ dvecs[0]) * dvecs[0]
                    overlap = float((d @ dvecs[0]).abs().item())
                    d_perp = d_perp / d_perp.norm()
                    lam = restricted_block_lambda_max(op_t, proj_weight, d_perp, K)
                    key = f"{CLASS_NAMES[a]}-{CLASS_NAMES[b]}"
                    restricted[f"lam_contrast_{key}"] = lam
                    restricted[f"overlap_{key}_vdata"] = overlap

            for r, v in enumerate(vals_t.tolist()):
                spec_rows.append(dict(dataset=args.dataset_name, block="theta",
                                      seed=seed, batch=bi, rank=r + 1, eigenvalue=v))
            for r, v in enumerate(vals_p.tolist()):
                spec_rows.append(dict(dataset=args.dataset_name, block="phi",
                                      seed=seed, batch=bi, rank=r + 1, eigenvalue=v))

            row = dict(
                dataset=args.dataset_name, fold=args.fold, width=args.width,
                seed=seed, batch=bi, n_valid_px=n_valid,
                lam1_theta=float(vals_t[0]), lam2_theta=float(vals_t[1]),
                lam5_theta=float(vals_t[4]), lam20_theta=float(vals_t[min(19, len(vals_t) - 1)]),
                trace_theta=tr_t, trace_theta_se=tr_t_se,
                conc_theta=float(vals_t[0]) / tr_t,
                lam1_phi=float(vals_p[0]), lam2_phi=float(vals_p[1]),
                trace_phi=tr_p, conc_phi=float(vals_p[0]) / tr_p,
                lam1_data=float(dvals[0]), lam2_data=float(dvals[1]),
                trace_data=dtr, conc_data=float(dvals[0]) / dtr,
                theta_resid_max=max(info_t["residuals"]) if info_t["residuals"] else None,
                class_counts=json.dumps({CLASS_NAMES[c]: n for c, n in counts.items()}),
                secs=time.time() - t0,
                **aligns, **restricted,
            )
            sum_rows.append(row)
            pd.DataFrame(spec_rows).to_csv(f"{prefix}_spectra.csv", index=False)
            pd.DataFrame(sum_rows).to_csv(f"{prefix}_summary.csv", index=False)

            msg = (f"seed={seed} b={bi} | l1_th={row['lam1_theta']:.3e} "
                   f"l2_th={row['lam2_theta']:.3e} l20_th={row['lam20_theta']:.3e} "
                   f"conc_th={row['conc_theta']:.2f} | l1_phi={row['lam1_phi']:.3e} "
                   f"conc_phi={row['conc_phi']:.3f} | align11={row.get('align_v1_data1', float('nan')):.3f}")
            if "lam_along_vdata" in restricted:
                msg += f" | lam(vdata)={restricted['lam_along_vdata']:.3e}"
                for k2, v2 in restricted.items():
                    if k2.startswith("lam_contrast"):
                        msg += f" {k2.replace('lam_contrast_', '')}={v2:.3e}"
            print(msg, flush=True)

        del model
        torch.cuda.empty_cache()

    # ---------------- aggregate summary ----------------
    df = pd.DataFrame(sum_rows)
    agg = df[["lam1_theta", "lam2_theta", "conc_theta", "lam1_phi",
              "conc_phi", "conc_data", "align_v1_data1"]].agg(["mean", "std"])
    print("\n" + agg.to_string(float_format=lambda v: f"{v:.4g}"))
    if "lam_along_vdata" in df.columns:
        sub = df.dropna(subset=["lam_along_vdata"])
        print(f"\nlam along v_data (mean-spectrum reader): "
              f"{sub['lam_along_vdata'].mean():.4e}")
        for c in [c for c in df.columns if c.startswith("lam_contrast")]:
            s = df[c].dropna()
            if len(s):
                ratio = sub["lam_along_vdata"].mean() / s.mean()
                print(f"{c}: {s.mean():.4e}   (v_data / contrast = {ratio:,.0f}x)")
    print(f"\n[write] {prefix}_spectra.csv, {prefix}_summary.csv")


if __name__ == "__main__":
    main()
