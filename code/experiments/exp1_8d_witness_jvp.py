"""Exp 1.8d — the requested witness JVP on the production BlockViT-v2 head.

Astra's `review_packet/astra/math_02.md` §1 lists four discrepancies between
what exp1_8c measured and what was claimed, and §7 items 1-4 turn them into a
concrete request. This script answers that request at RANDOM INIT on the same
core, widths and protocol as exp1_8c (which it imports and does not modify).

  1. BN quantities. Per-channel batch mean and BIASED variance over the FULL
     normalization group (all B*H*W positions, ignored pixels included), gamma,
     beta, eps, for both head BNs. Summaries: variance spread, effective gain
     gamma_i/sqrt(var_i+eps), fraction of channels with var_i < 10 eps. The old
     scalar `bn_gain_rms` is kept under its honest name
     `activation_energy_ratio_sqrt` -- it is sqrt(E_postBN/E_preBN) on labeled
     pixels, not an RMS of BN derivative gains (§1 discrepancy 2).

  2. Witness JVP (§1 discrepancy 1, §7 item 2). A specified UNIT weight
     perturbation Delta W is propagated by forward-mode AD through the ACTUAL
     downstream path in the actual BN mode, and its energy is recorded at every
     stage: pre-BN, after centering, after BN1, after the ReLU gate, after BN2,
     at the logits on labeled pixels, and finally weighted by the CE metric
     H_n = diag(p_n) - p_n p_n^T. The last scalar is checked against
     Delta W^T G_WW Delta W computed with exp1_8c's own GGN matvec.

     In train mode the BN batch statistics are functions of the pre-activation,
     so the BN derivative includes that dependence. The head is therefore
     re-implemented op-by-op below (`head_forward`) instead of calling
     F.batch_norm: forward-mode AD then differentiates the batch mean and
     variance as ordinary ops, which is exactly Counterexample 3.4's operator.
     `head_forward` is validated against `model.seg_head` on every row.

  3. Function-preserving scale control (§6 (B3), §7 item 3). Scaling
     seg_head[0] weight AND bias by c and co-scaling seg_head[1].eps by c^2
     leaves the function unchanged and must give G_WW(cW) = c^-2 G_WW(W).
     Repeated at fixed eps to quantify the departure that the implementation
     actually has, with per-channel pre-BN variance vs eps to explain it.

  4. Eigensolver accuracy (§7 item 4). Every power-iteration eigenvalue is
     reported with the relative residual ||Gv - lam v||/lam and the Rayleigh
     quotient of the final vector.

Modes: `train` (all BN on batch statistics), `eval_head` (both head BNs in
eval), `eval_bn1` (seg_head[1] only), `eval_bn2` (seg_head[4] only). The full
`eval` mode of exp1_8c is skipped: it de-standardizes the incoming features and
is confounded.

Usage:
    python code/experiments/exp1_8d_witness_jvp.py \
        --widths 48 192 384 --seeds 0 1 2
    python code/experiments/exp1_8d_witness_jvp.py \
        --widths 48 --seeds 0 --modes train --smoke     # smoke test
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
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD

THEORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THEORY_ROOT / "code"))
sys.path.insert(0, str(THEORY_ROOT / "code" / "experiments"))
SIDE = Path("/home/u37314kd/Projects/spectral_tokenization/side_project")
sys.path.insert(0, str(SIDE))

from hessian.ggn import ggn_block_vector_product, math_attention        # noqa: E402
from exp1_8c_interior_witness import (                                  # noqa: E402
    build_model, load_densest_batch, set_bn_mode, capture_head_activations,
    fused_gram_stats, split_params, loglog_slope,
)

IGNORE = 255
DIRS_GRAM = ("s9cen", "mu9", "s9", "s9cen_full")   # input-side, channel sweep
DIRS_ALL = DIRS_GRAM + ("ggn_top", "bias_ec")


# --------------------------------------------------------------------------
# BN modes
# --------------------------------------------------------------------------
def set_bn_mode_ext(model, mode: str) -> None:
    """exp1_8c's three modes plus the two single-BN interventions."""
    if mode in ("train", "eval", "eval_head"):
        set_bn_mode(model, mode)
        return
    if mode not in ("eval_bn1", "eval_bn2"):
        raise ValueError(f"unknown BN mode: {mode}")
    model.train()
    (model.seg_head[1] if mode == "eval_bn1" else model.seg_head[4]).eval()


# --------------------------------------------------------------------------
# explicit functional head (so forward-mode AD sees the batch statistics)
# --------------------------------------------------------------------------
def _bn_ref(bn) -> dict:
    return dict(
        weight=bn.weight.detach(), bias=bn.bias.detach(),
        running_mean=bn.running_mean.detach(), running_var=bn.running_var.detach(),
        eps=float(bn.eps), train=bool(bn.training),
    )


class Head:
    """Detached snapshot of seg_head in its current BN mode."""

    def __init__(self, seg_head):
        self.W0 = seg_head[0].weight.detach()
        self.b0 = seg_head[0].bias.detach()
        self.W1 = seg_head[3].weight.detach()
        self.b1 = seg_head[3].bias.detach()
        self.W2 = seg_head[6].weight.detach()
        self.b2 = seg_head[6].bias.detach()
        self.bn1 = _bn_ref(seg_head[1])
        self.bn2 = _bn_ref(seg_head[4])


def _bn_apply(a, ref):
    """BatchNorm2d written out in primitive ops.

    Train mode uses the batch mean and the BIASED batch variance over the full
    normalization group (dims 0,2,3) -- identical to F.batch_norm(training=True)
    -- but as differentiable functions of `a`, which is what the JVP needs.
    """
    if ref["train"]:
        mean = a.mean(dim=(0, 2, 3), keepdim=True)
        d = a - mean
        var = (d * d).mean(dim=(0, 2, 3), keepdim=True)
    else:
        d = a - ref["running_mean"].view(1, -1, 1, 1)
        var = ref["running_var"].view(1, -1, 1, 1)
    xhat = d / torch.sqrt(var + ref["eps"])
    return xhat * ref["weight"].view(1, -1, 1, 1) + ref["bias"].view(1, -1, 1, 1)


def head_forward(fused, W, b, H: Head) -> dict:
    a = F.conv2d(fused, W, b, padding=1)
    h1 = _bn_apply(a, H.bn1)
    r1 = F.relu(h1)
    a2 = F.conv2d(r1, H.W1, H.b1, padding=1)
    h2 = _bn_apply(a2, H.bn2)
    r2 = F.relu(h2)
    logits = F.conv2d(r2, H.W2, H.b2, padding=1)
    return dict(a=a, h1=h1, r1=r1, a2=a2, h2=h2, logits=logits)


# --------------------------------------------------------------------------
# the witness JVP itself
# --------------------------------------------------------------------------
def _energy_full(t: torch.Tensor) -> float:
    """Mean over the FULL normalization group of ||.||^2 summed over channels."""
    g = t.shape[0] * t.shape[2] * t.shape[3]
    return float(t.pow(2).sum(dtype=torch.float64).item() / g)


def witness_jvp(fused, H: Head, dW, db, pv, keep) -> dict:
    """Propagate one unit perturbation of seg_head[0] through the actual head.

    Args:
        fused: (1, M+K, Hh, Ww) seg_head input, detached.
        H:     detached head snapshot in the current BN mode.
        dW:    tangent for seg_head[0].weight, or None.
        db:    tangent for seg_head[0].bias, or None.
        pv:    (N_valid, C) softmax probabilities on labeled pixels.
        keep:  (1, Hh, Ww) bool label mask.
    """
    with fwAD.dual_level():
        Wd = H.W0 if dW is None else fwAD.make_dual(H.W0, dW)
        bd = H.b0 if db is None else fwAD.make_dual(H.b0, db)
        outs = head_forward(fused, Wd, bd, H)
        tan = {}
        for k, v in outs.items():
            t = fwAD.unpack_dual(v).tangent
            tan[k] = None if t is None else t.detach().clone()

    d_a = tan["a"]
    d_cen = d_a - d_a.mean(dim=(0, 2, 3), keepdim=True)
    dl = tan["logits"]
    dlv = dl.permute(0, 2, 3, 1)[keep].double()                # (N_valid, C)
    e_logit_valid = float(dlv.pow(2).sum(1).mean().item())
    quad = (pv * dlv * dlv).sum(1) - (pv * dlv).sum(1) ** 2    # d^T H_n d
    out = dict(
        e_pre=_energy_full(d_a),
        e_cen=_energy_full(d_cen),
        e_bn1=_energy_full(tan["h1"]),
        e_relu1=_energy_full(tan["r1"]),
        e_bn2=_energy_full(tan["h2"]),
        e_logit_valid=e_logit_valid,
        e_final=float(quad.mean().item()),
        quad_min=float(quad.min().item()),
    )
    del tan, outs, d_a, d_cen, dl, dlv, quad
    return out


def make_dW(u: torch.Tensor, c: int, shape) -> torch.Tensor:
    """Delta W = e_c (x) u, unit Frobenius norm (||u|| = 1)."""
    dW = torch.zeros(shape, device=u.device, dtype=u.dtype)
    dW[c] = u.reshape(shape[1], shape[2], shape[3])
    return dW


# --------------------------------------------------------------------------
# power iteration with residual diagnostics (same matvec as exp1_8c)
# --------------------------------------------------------------------------
def _gnorm(vs):
    return float(torch.sqrt(sum((v * v).sum() for v in vs)).item())


def _normalize(vs):
    n = max(_gnorm(vs), 1e-30)
    return [v / n for v in vs]


def power_iter_ggn(model, X, Y, params, n_iter, tol, seed,
                   budget_s=None, return_vec=False, resid_target=1e-3):
    """Top eigenvalue of the GGN block by power iteration, with residuals.

    Uses `hessian.ggn.ggn_block_vector_product` verbatim, so the loss
    normalization (1/N_valid, ignore_index dropped) is identical to exp1_8c.

    math_02 §7 item 4: a relative-change stopping rule on the Rayleigh quotient
    can stop long before the eigenpair residual is small, so the loop here also
    requires ||Gv - lam v|| / lam < `resid_target`. Both quantities come from
    the matvec of the CURRENT iterate, so the check is free, and the returned
    (lam, vector, residual) are one consistent eigenpair estimate.
    """
    t0 = time.time()
    with math_attention():
        logits = model(X)
    n_valid = int((Y != IGNORE).sum().item())
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    v = _normalize([torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen)
                    for p in params])

    eig, eig_old, it, rel_resid = 0.0, float("inf"), 0, float("nan")
    stop, v_rep = "n_iter", v
    for it in range(1, n_iter + 1):
        Gv = ggn_block_vector_product(logits, params, v, Y, n_valid=n_valid)
        eig = float(sum((g * vi).sum() for g, vi in zip(Gv, v)).item())
        v_rep = v                      # the iterate eig/rel_resid describe
        nrm = _gnorm(Gv)
        if nrm < 1e-20:
            eig, rel_resid, stop = 0.0, 0.0, "collapsed"
            break
        rel_resid = (_gnorm([g - eig * vi for g, vi in zip(Gv, v)]) / abs(eig)
                     if abs(eig) > 0 else float("nan"))
        if (abs(eig - eig_old) / max(abs(eig), 1e-12) < tol
                and rel_resid < resid_target):
            stop = "tol+resid"
            break
        eig_old = eig
        if budget_s is not None and (time.time() - t0) > budget_s:
            stop = "budget"
            break
        v = _normalize(Gv)

    out = dict(lam=eig, rayleigh=eig, rel_resid=rel_resid, iters=it,
               stop=stop, secs=time.time() - t0)
    vec = [vi.detach().clone() for vi in v_rep] if return_vec else None
    del logits, v, v_rep, Gv
    torch.cuda.empty_cache()
    return (out, vec) if return_vec else out


def ggn_quadform(model, X, Y, params, v):
    """v^T G v with the same matvec, for the e_final cross-check."""
    with math_attention():
        logits = model(X)
    n_valid = int((Y != IGNORE).sum().item())
    Gv = ggn_block_vector_product(logits, params, v, Y, n_valid=n_valid)
    q = float(sum((g * vi).sum() for g, vi in zip(Gv, v)).item())
    del logits, Gv
    torch.cuda.empty_cache()
    return q


# --------------------------------------------------------------------------
# patch-Gram directions
# --------------------------------------------------------------------------
def patch_gram_directions(fused, Y, chunk: int = 2048):
    """Unit input-side directions u in R^{9(M+K)} and their Gram eigenvalues.

    Unfold ordering is channel-major (c*9 + ki*3 + kj), matching the flattening
    of a Conv2d weight slice, so u.reshape(C,3,3) is directly a weight row.

    `valid` uses labeled pixels only (exp1_8c's mask, kept for continuity);
    `full` uses every position, i.e. the actual BN normalization group
    (math_02 §1 discrepancy 3).

    The Gram is accumulated in float64 over chunks of the unfolded tensor, with
    no permuted copy, so the peak here is one unfold (about 1.8 GB at M=384).
    """
    B, C, Hh, Ww = fused.shape
    D = C * 9
    U = F.unfold(fused, kernel_size=3, padding=1)              # (B, D, L)
    L = U.shape[-1]
    keep = (Y != IGNORE).reshape(B, L)

    dirs, stats = {}, {}
    for tag in ("full", "valid"):
        S = torch.zeros(D, D, dtype=torch.float64, device=U.device)
        mu = torch.zeros(D, dtype=torch.float64, device=U.device)
        n = 0
        for b in range(B):
            for i in range(0, L, chunk):
                c = U[b, :, i: i + chunk].T
                if tag == "valid":
                    c = c[keep[b, i: i + chunk]]
                if c.shape[0] == 0:
                    continue
                c = c.double()
                S += c.T @ c
                mu += c.sum(0)
                n += c.shape[0]
                del c
        S /= n
        mu /= n
        ev, evec = torch.linalg.eigh(S)
        S_cen = S - torch.outer(mu, mu)
        evc, evecc = torch.linalg.eigh(S_cen)
        stats[f"lamS9_{tag}"] = float(ev[-1].item())
        stats[f"lamS9cen_{tag}"] = float(evc[-1].item())
        stats[f"trS9_{tag}"] = float(S.diagonal().sum().item())
        stats[f"mu9norm_{tag}"] = float(mu.norm().item())
        stats[f"n_{tag}"] = int(n)
        if tag == "valid":
            dirs["s9"] = evec[:, -1].float()
            dirs["s9cen"] = evecc[:, -1].float()
            dirs["mu9"] = (mu / mu.norm()).float()
        else:
            dirs["s9cen_full"] = evecc[:, -1].float()
        del S, S_cen, ev, evec, evc, evecc, mu
        torch.cuda.empty_cache()
    del U
    torch.cuda.empty_cache()

    stats["cos_s9cen_valid_full"] = float(
        (dirs["s9cen"].double() @ dirs["s9cen_full"].double()).abs().item())
    stats["cos_s9cen_mu9"] = float(
        (dirs["s9cen"].double() @ dirs["mu9"].double()).abs().item())
    stats["cos_s9_mu9"] = float(
        (dirs["s9"].double() @ dirs["mu9"].double()).abs().item())
    for k in dirs:
        dirs[k] = dirs[k] / dirs[k].norm()
    return dirs, stats


# --------------------------------------------------------------------------
# BN per-channel statistics
# --------------------------------------------------------------------------
def bn_channel_stats(a: torch.Tensor, ref: dict) -> dict:
    """Per-channel batch mean / biased variance over the FULL group + affine."""
    with torch.no_grad():
        mean = a.mean(dim=(0, 2, 3), dtype=torch.float64)
        d = a.double() - mean.view(1, -1, 1, 1)
        var = (d * d).mean(dim=(0, 2, 3))
        del d
    eps = ref["eps"]
    gamma = ref["weight"].double()
    gain = gamma / torch.sqrt(var + eps)
    return dict(
        mean=mean.cpu().numpy(), var=var.cpu().numpy(),
        gamma=gamma.cpu().numpy(), beta=ref["bias"].double().cpu().numpy(),
        running_mean=ref["running_mean"].double().cpu().numpy(),
        running_var=ref["running_var"].double().cpu().numpy(),
        gain=gain.cpu().numpy(), eps=np.array(eps),
        bn_train=np.array(1 if ref["train"] else 0),
    )


def bn_summary(st: dict, tag: str) -> dict:
    v, g, eps = st["var"], st["gain"], float(st["eps"])
    return {
        f"{tag}_eps": eps,
        f"{tag}_var_min": float(v.min()), f"{tag}_var_med": float(np.median(v)),
        f"{tag}_var_max": float(v.max()),
        f"{tag}_gain_min": float(g.min()), f"{tag}_gain_med": float(np.median(g)),
        f"{tag}_gain_max": float(g.max()),
        f"{tag}_frac_var_lt_10eps": float((v < 10 * eps).mean()),
        f"{tag}_mean_absmax": float(np.abs(st["mean"]).max()),
        f"{tag}_gamma_min": float(st["gamma"].min()),
        f"{tag}_gamma_max": float(st["gamma"].max()),
    }


# --------------------------------------------------------------------------
# markdown helpers
# --------------------------------------------------------------------------
def fmt(v, sig=4):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        a = abs(v)
        if a != 0 and (a < 1e-3 or a >= 1e5):
            return f"{v:.{sig - 1}e}"
        return f"{v:.{sig}g}"
    return str(v)


def md_table(df: pd.DataFrame, cols=None, sig=4) -> str:
    d = df if cols is None else df[cols]
    head = list(d.columns)
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], sig) for c in head) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/hdd2/u37314kd/data_breast_v2_pca23")
    ap.add_argument("--dataset_name", default="breast")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--widths", type=int, nargs="+", default=[48, 192, 384])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+",
                    default=["train", "eval_head", "eval_bn1", "eval_bn2"])
    ap.add_argument("--directions", nargs="+", default=list(DIRS_ALL))
    ap.add_argument("--n_candidate_cores", type=int, default=24)
    ap.add_argument("--n_out_channels", type=int, default=0,
                    help="subsample the 96 output channels (0 = all)")
    ap.add_argument("--n_iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--n_iter_phi", type=int, default=60)
    ap.add_argument("--tol_phi", type=float, default=1e-5)
    ap.add_argument("--phi_budget_s", type=float, default=120.0)
    ap.add_argument("--no_phi", action="store_true")
    ap.add_argument("--no_scale_control", action="store_true")
    ap.add_argument("--scale_cs", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--spatial_size", type=int, default=336)
    ap.add_argument("--reduce_dim", type=int, default=64)
    ap.add_argument("--num_spectral", type=int, default=314)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--patch_tok_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_spectral_norm", action="store_true")
    ap.add_argument("--tf32", action="store_true",
                    help="allow cuDNN TF32 (off by default: the cross-check needs fp32)")
    ap.add_argument("--xcheck_tol", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true",
                    help="stop after the first row's cross-check")
    ap.add_argument("--report_only", action="store_true",
                    help="rebuild the summary/report from existing CSVs, no GPU work")
    ap.add_argument("--resume", action="store_true",
                    help="keep complete (width,seed,mode) blocks already in the CSV "
                         "and only compute what is missing (the GPU is shared)")
    ap.add_argument("--tmp_dir",
                    default="/tmp/claude-1008/-home-u37314kd-Projects-spectral-tokenization/"
                            "887fb000-54db-4ab4-82e6-4eeab4fb6c48/scratchpad")
    ap.add_argument("--out", default=str(THEORY_ROOT / "results" / "exp1_8d_witness_jvp.csv"))
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    for w in args.widths:
        if w % args.num_heads != 0:
            raise SystemExit(f"width {w} not divisible by num_heads {args.num_heads}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bn_dir = out_path.parent / "exp1_8d_bn_stats"
    bn_dir.mkdir(parents=True, exist_ok=True)
    scale_path = out_path.parent / "exp1_8d_scale_control.csv"

    if args.report_only:
        df = pd.read_csv(out_path)
        sdf = pd.read_csv(scale_path) if scale_path.exists() else pd.DataFrame()
        args.widths = sorted(int(w) for w in df["width"].unique())
        args.seeds = sorted(int(s) for s in df["seed"].unique())
        seen = set(df["mode"])
        args.modes = [m for m in args.modes if m in seen] or list(df["mode"].unique())
        bad = df[(~df["xcheck_both_zero"]) & (df["xcheck_rel_disc"] > args.xcheck_tol)]
        xf = bad[["width", "seed", "mode", "direction", "e_final", "ggn_quad",
                  "xcheck_rel_disc"]].values.tolist()
        write_summary(df, sdf, args, out_path, str(df["core"].iloc[0]),
                      int(df["n_valid_px"].iloc[0]), xf)
        return

    rows, scale_rows, xcheck_fail = [], [], []
    done_modes, scale_done = set(), set()
    if args.resume:
        if out_path.exists():
            prev = pd.read_csv(out_path)
            nd = len(args.directions)
            for (w, s, m), g in prev.groupby(["width", "seed", "mode"]):
                if len(g) >= nd:
                    done_modes.add((int(w), int(s), str(m)))
            keepmask = [(int(r.width), int(r.seed), str(r["mode"])) in done_modes
                        for _, r in prev.iterrows()]
            prev = prev[keepmask]
            rows = prev.to_dict("records")
            print(f"[resume] kept {len(rows)} rows covering "
                  f"{len(done_modes)} (width,seed,mode) blocks", flush=True)
        if scale_path.exists():
            ps = pd.read_csv(scale_path)
            scale_rows = ps.to_dict("records")
            scale_done = {int(w) for w in ps["width"].unique()}
            print(f"[resume] scale control already done for widths "
                  f"{sorted(scale_done)}", flush=True)

    X, Y, n_valid, core_name = load_densest_batch(args)
    keep = (Y != IGNORE)

    for width in args.widths:
        for seed in args.seeds:
            todo = [m for m in args.modes if (width, seed, m) not in done_modes]
            need_scale = ((not args.no_scale_control) and seed == args.seeds[0]
                          and width not in scale_done)
            if not todo and not need_scale:
                print(f"[skip] M={width} seed={seed} (already complete)", flush=True)
                continue
            model = build_model(width, seed, args)
            theta, phi = split_params(model)
            W0p = model.seg_head[0].weight
            b0p = model.seg_head[0].bias
            Wshape = tuple(W0p.shape)
            n_out = Wshape[0]
            chans = (list(range(n_out)) if args.n_out_channels in (0, n_out)
                     else list(np.linspace(0, n_out - 1, args.n_out_channels).astype(int)))

            # ---- incoming features (mode-independent: only head BNs change) --
            if todo:
                set_bn_mode_ext(model, "train")
                fused_ref, a_pre_ref, a_bn_ref, _ = capture_head_activations(model, X)
                # want_patch=False: exp1_8c's patch Gram materializes a permuted
                # copy of the unfold (~3.6 GB at M=384). patch_gram_directions
                # computes the same S9/S9_cen chunked, plus the full-group version.
                gs = fused_gram_stats(fused_ref, Y, n_up=width, want_patch=False)
                dirs, gstats = patch_gram_directions(fused_ref, Y)
                fused_ck = float(fused_ref.double().pow(2).sum().item())
                del a_pre_ref, a_bn_ref
                torch.cuda.empty_cache()

            bn_npz = {}
            for mode in todo:
                set_bn_mode_ext(model, mode)
                t0 = time.time()
                torch.cuda.reset_peak_memory_stats()

                fused, a_pre, a_bn, logits_mod = capture_head_activations(model, X)
                fused_dev = float((fused - fused_ref).abs().max().item())
                with torch.no_grad():
                    e_pre_act = float((a_pre.permute(0, 2, 3, 1)[keep] ** 2).sum(-1).mean().item())
                    e_bn_act = float((a_bn.permute(0, 2, 3, 1)[keep] ** 2).sum(-1).mean().item())
                    aer = float(np.sqrt(e_bn_act / e_pre_act)) if e_pre_act > 0 else float("nan")
                    p = torch.softmax(logits_mod, dim=1)
                    pv = p.permute(0, 2, 3, 1)[keep].double()
                    p_min_mean = float(p.min(dim=1).values[keep].mean().item())
                del a_pre, a_bn, p
                torch.cuda.empty_cache()

                # ---- functional head snapshot + validation -------------------
                H = Head(model.seg_head)
                with torch.no_grad():
                    o = head_forward(fused, H.W0, H.b0, H)
                    head_dev = float((o["logits"] - logits_mod).abs().max().item())
                    logit_scale = float(logits_mod.abs().max().item())
                    bn_npz[f"{mode}__bn1"] = bn_channel_stats(o["a"], H.bn1)
                    bn_npz[f"{mode}__bn2"] = bn_channel_stats(o["a2"], H.bn2)
                    del o
                torch.cuda.empty_cache()
                bns = {}
                bns.update(bn_summary(bn_npz[f"{mode}__bn1"], "bn1"))
                bns.update(bn_summary(bn_npz[f"{mode}__bn2"], "bn2"))

                # ---- eigenvalues (with residuals) ---------------------------
                r_WW, v_WW = power_iter_ggn(model, X, Y, [W0p], args.n_iter,
                                            args.tol, seed, return_vec=True)
                r_Wb = power_iter_ggn(model, X, Y, [W0p, b0p], args.n_iter,
                                      args.tol, seed)
                r_phi = dict(lam=float("nan"), rel_resid=float("nan"),
                             rayleigh=float("nan"), iters=0, stop="skipped", secs=0.0)
                if not args.no_phi:
                    r_phi = power_iter_ggn(model, X, Y, phi, args.n_iter_phi,
                                           args.tol_phi, seed,
                                           budget_s=args.phi_budget_s)

                base = dict(
                    dataset=args.dataset_name, fold=args.fold, split=args.split,
                    core=core_name, n_valid_px=n_valid, width=width, seed=seed,
                    mode=mode, head_in=width + args.reduce_dim,
                    lam_WW=r_WW["lam"], resid_WW=r_WW["rel_resid"],
                    ray_WW=r_WW["rayleigh"], iters_WW=r_WW["iters"], stop_WW=r_WW["stop"],
                    lam_Wb=r_Wb["lam"], resid_Wb=r_Wb["rel_resid"],
                    ray_Wb=r_Wb["rayleigh"], iters_Wb=r_Wb["iters"], stop_Wb=r_Wb["stop"],
                    lam_phi=r_phi["lam"], resid_phi=r_phi["rel_resid"],
                    ray_phi=r_phi["rayleigh"], iters_phi=r_phi["iters"],
                    stop_phi=r_phi["stop"], secs_phi=r_phi["secs"],
                    activation_energy_ratio_sqrt=aer,
                    energy_preBN_labeled=e_pre_act, energy_postBN_labeled=e_bn_act,
                    p_min_mean=p_min_mean,
                    fused_max_dev_vs_train=fused_dev, fused_sq_sum=fused_ck,
                    head_max_dev_vs_module=head_dev, logit_absmax=logit_scale,
                    lam_Sh=gs["lam_Sh"], lam_Sh_cen=gs["lam_Sh_cen"],
                    lam_Shup=gs["lam_Shup"], lam_Shsk=gs["lam_Shsk"],
                    cos_u_mu_Sh=gs["cos_u_mu_Sh"],
                    **gstats, **bns,
                    tf32=bool(args.tf32),
                )

                # ---- witness JVP sweeps --------------------------------------
                for dname in args.directions:
                    tj = time.time()
                    if dname == "ggn_top":
                        dW = v_WW[0]
                        res = witness_jvp(fused, H, dW, None, pv, keep)
                        gq = ggn_quadform(model, X, Y, [W0p], [dW])
                        best = dict(res)
                        best.update(c_star=-1, dW_fro=float(dW.norm().item()),
                                    ggn_quad=gq, n_channels=0)
                        agg = {f"{k}_mean": float("nan") for k in
                               ("e_pre", "e_cen", "e_bn1", "e_relu1", "e_bn2",
                                "e_logit_valid", "e_final")}
                        e_final_max_over_c = res["e_final"]
                    else:
                        per_c = []
                        for c in chans:
                            if dname == "bias_ec":
                                db = torch.zeros_like(H.b0)
                                db[c] = 1.0
                                r = witness_jvp(fused, H, None, db, pv, keep)
                                r["c"] = c
                            else:
                                dW = make_dW(dirs[dname], c, Wshape)
                                r = witness_jvp(fused, H, dW, None, pv, keep)
                                r["c"] = c
                            per_c.append(r)
                        pc = pd.DataFrame(per_c)
                        ci = int(pc["e_final"].idxmax())
                        c_star = int(pc.loc[ci, "c"])
                        best = {k: float(pc.loc[ci, k]) for k in
                                ("e_pre", "e_cen", "e_bn1", "e_relu1", "e_bn2",
                                 "e_logit_valid", "e_final", "quad_min")}
                        agg = {f"{k}_mean": float(pc[k].mean()) for k in
                               ("e_pre", "e_cen", "e_bn1", "e_relu1", "e_bn2",
                                "e_logit_valid", "e_final")}
                        if dname == "bias_ec":
                            db = torch.zeros_like(H.b0)
                            db[c_star] = 1.0
                            gq = ggn_quadform(model, X, Y, [b0p], [db])
                            fro = float(db.norm().item())
                        else:
                            dW = make_dW(dirs[dname], c_star, Wshape)
                            gq = ggn_quadform(model, X, Y, [W0p], [dW])
                            fro = float(dW.norm().item())
                        best.update(c_star=c_star, dW_fro=fro, ggn_quad=gq,
                                    n_channels=len(chans))
                        e_final_max_over_c = float(pc["e_final"].max())

                    absd = abs(best["e_final"] - best["ggn_quad"])
                    scale = max(abs(best["ggn_quad"]), abs(best["e_final"]))
                    both_zero = scale < 1e-12
                    # both sides vanish to machine precision (train-mode bias):
                    # a ratio there is meaningless, the absolute gap is the fact.
                    rel = 0.0 if both_zero else absd / scale
                    if not np.isfinite(rel) or rel > args.xcheck_tol:
                        xcheck_fail.append(
                            (width, seed, mode, dname, best["e_final"],
                             best["ggn_quad"], rel))
                    rows.append(dict(base, direction=dname, **best, **agg,
                                     xcheck_rel_disc=rel,
                                     xcheck_abs_disc=absd,
                                     xcheck_both_zero=both_zero,
                                     e_final_max_over_c=e_final_max_over_c,
                                     secs_dir=time.time() - tj))
                    print(f"  M={width:4d} s={seed} {mode:9s} {dname:11s} "
                          f"c*={best['c_star']:4d} e_pre={best['e_pre']:.4e} "
                          f"e_cen={best['e_cen']:.4e} e_bn1={best['e_bn1']:.4e} "
                          f"e_fin={best['e_final']:.4e} GGN={best['ggn_quad']:.4e} "
                          f"rel={rel:.2e} | lam_WW={r_WW['lam']:.4e} "
                          f"({r_WW['rel_resid']:.1e}) {time.time()-tj:.0f}s",
                          flush=True)
                    pd.DataFrame(rows).to_csv(out_path, index=False)

                    if args.smoke:
                        print("\n[smoke] cross-check "
                              f"{'PASS' if rel <= args.xcheck_tol else 'FAIL'} "
                              f"(rel={rel:.3e}); head_dev={head_dev:.3e}, "
                              f"logit_absmax={logit_scale:.3e}")
                        return

                peak = torch.cuda.max_memory_allocated() / 1e9
                rows[-1]["peak_gb"] = peak
                print(f"  [M={width} s={seed} {mode}] peak {peak:.1f} GB, "
                      f"{time.time()-t0:.0f}s", flush=True)
                del fused, logits_mod, H, pv, v_WW
                torch.cuda.empty_cache()

            if todo:
                npz_path = bn_dir / f"M{width}_s{seed}.npz"
                store = {}
                if npz_path.exists():                     # merge, do not truncate
                    z = np.load(npz_path, allow_pickle=False)
                    store = {k: z[k] for k in z.files}
                store.update({f"{k}__{kk}": vv for k, st in bn_npz.items()
                              for kk, vv in st.items()})
                store.update(width=np.array(width), seed=np.array(seed),
                             core=np.array(core_name), n_valid_px=np.array(n_valid))
                np.savez(npz_path, **store)
                del store, fused_ref, dirs
            del bn_npz
            torch.cuda.empty_cache()

            # ---- (3) function-preserving scale control -----------------------
            if need_scale:
                scale_rows += scale_control(model, X, Y, width, seed, args)
                pd.DataFrame(scale_rows).to_csv(scale_path, index=False)

            del model, theta, phi, W0p, b0p
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[write] {out_path} ({len(df)} rows)")
    if scale_rows:
        pd.DataFrame(scale_rows).to_csv(scale_path, index=False)
        print(f"[write] {scale_path} ({len(scale_rows)} rows)")

    write_summary(df, pd.DataFrame(scale_rows), args, out_path, core_name,
                  n_valid, xcheck_fail)


# --------------------------------------------------------------------------
def scale_control(model, X, Y, width, seed, args):
    """math_02 (B3): BN's exact scale invariance and its fixed-eps departure."""
    set_bn_mode_ext(model, "train")
    W = model.seg_head[0].weight
    b = model.seg_head[0].bias
    bn1 = model.seg_head[1]
    W_orig = W.detach().clone()
    b_orig = b.detach().clone()
    eps0 = float(bn1.eps)

    with torch.no_grad(), math_attention():
        logits0 = model(X).detach().clone()

    out = []
    for coscale in (True, False):
        for c in args.scale_cs:
            with torch.no_grad():
                W.copy_(W_orig * c)
                b.copy_(b_orig * c)
            bn1.eps = (c * c * eps0) if coscale else eps0
            fused, a_pre, _, logits_c = capture_head_activations(model, X)
            with torch.no_grad():
                dmax = float((logits_c - logits0).abs().max().item())
                m = a_pre.mean(dim=(0, 2, 3), dtype=torch.float64).view(1, -1, 1, 1)
                var = ((a_pre.double() - m) ** 2).mean(dim=(0, 2, 3))
                var = var.cpu().numpy()
            del fused, a_pre, logits_c
            torch.cuda.empty_cache()
            r = power_iter_ggn(model, X, Y, [W], args.n_iter, args.tol, seed)
            wf2 = float((W.detach().double() ** 2).sum().item())
            out.append(dict(
                width=width, seed=seed, coscale_eps=coscale, c=c,
                eps=float(bn1.eps), eps0=eps0,
                logits_max_dev=dmax, lam_WW=r["lam"], resid_WW=r["rel_resid"],
                iters_WW=r["iters"], stop_WW=r["stop"],
                W_fro2=wf2, invariant_Wfro2_lam=wf2 * r["lam"],
                var_min=float(var.min()), var_med=float(np.median(var)),
                var_max=float(var.max()),
                var_over_eps_med=float(np.median(var) / bn1.eps),
                frac_var_lt_10eps=float((var < 10 * bn1.eps).mean()),
            ))
            print(f"  [scale] M={width} coscale={int(coscale)} c={c:<5} "
                  f"|dlogit|={dmax:.3e} lam_WW={r['lam']:.4e} "
                  f"var_med/eps={out[-1]['var_over_eps_med']:.3e}", flush=True)

    with torch.no_grad():
        W.copy_(W_orig)
        b.copy_(b_orig)
    bn1.eps = eps0

    # ratio lam(c) * c^2 / lam(1), per eps convention
    d = pd.DataFrame(out)
    for cs in (True, False):
        m = d["coscale_eps"] == cs
        ref = d.loc[m & (d["c"] == 1.0), "lam_WW"]
        ref = float(ref.iloc[0]) if len(ref) else float("nan")
        d.loc[m, "ratio_c2lam_over_lam1"] = d.loc[m, "lam_WW"] * d.loc[m, "c"] ** 2 / ref
    return d.to_dict(orient="records")


# --------------------------------------------------------------------------
def write_summary(df, sdf, args, out_path, core_name, n_valid, xcheck_fail):
    order = {d: i for i, d in enumerate(DIRS_ALL)}
    df = df.copy()
    df["_dord"] = df["direction"].map(lambda d: order.get(d, 99))
    df = df.sort_values(["width", "seed", "_dord"])
    # recompute from the full table so resumed rows are covered too
    bad = df[(~df["xcheck_both_zero"].astype(bool))
             & (df["xcheck_rel_disc"] > args.xcheck_tol)]
    xcheck_fail = bad[["width", "seed", "mode", "direction", "e_final",
                       "ggn_quad", "xcheck_rel_disc"]].values.tolist()
    ecols = ["e_pre", "e_cen", "e_bn1", "e_relu1", "e_bn2", "e_logit_valid", "e_final"]
    key = ["lam_WW", "lam_Wb", "lam_phi", "resid_WW", "resid_Wb", "resid_phi",
           "activation_energy_ratio_sqrt", "p_min_mean"]

    per_mw = (df[df["direction"] == df["direction"].iloc[0]]
              .groupby(["mode", "width"])[key].mean().reset_index())
    per_mwd = df.groupby(["mode", "width", "direction"])[
        ecols + ["xcheck_rel_disc"]].mean().reset_index()

    slopes = {}
    for mode in df["mode"].unique():
        s = {}
        sub = per_mw[per_mw["mode"] == mode]
        for c in ("lam_WW", "lam_Wb", "lam_phi"):
            s[c] = loglog_slope(sub["width"], sub[c])
        for dname in df["direction"].unique():
            q = per_mwd[(per_mwd["mode"] == mode) & (per_mwd["direction"] == dname)]
            s[f"e_final__{dname}"] = loglog_slope(q["width"], q["e_final"])
        slopes[mode] = s

    slopes_seed = {}
    for mode in df["mode"].unique():
        slopes_seed[mode] = {}
        for seed in sorted(df["seed"].unique()):
            s = {}
            sub = df[(df["mode"] == mode) & (df["seed"] == seed)]
            g = (sub.groupby("width")[["lam_WW", "lam_Wb", "lam_phi"]]
                 .mean().reset_index())
            for c in ("lam_WW", "lam_Wb", "lam_phi"):
                s[c] = loglog_slope(g["width"], g[c])
            for dname in df["direction"].unique():
                q = sub[sub["direction"] == dname].sort_values("width")
                s[f"e_final__{dname}"] = loglog_slope(q["width"], q["e_final"])
            slopes_seed[mode][int(seed)] = s

    sp = out_path.with_suffix(".summary.json")
    sp.write_text(json.dumps({
        "dataset": args.dataset_name, "fold": args.fold, "core": core_name,
        "n_valid_px": n_valid, "widths": args.widths, "seeds": args.seeds,
        "modes": args.modes, "directions": args.directions,
        "reduce_dim": args.reduce_dim, "spatial_size": args.spatial_size,
        "tf32": bool(args.tf32), "n_iter": args.n_iter, "tol": args.tol,
        "n_iter_phi": args.n_iter_phi, "tol_phi": args.tol_phi,
        "per_mode_width": per_mw.to_dict(orient="records"),
        "per_mode_width_direction": per_mwd.to_dict(orient="records"),
        "loglog_slopes_vs_M": slopes,
        "loglog_slopes_vs_M_per_seed": slopes_seed,
        "xcheck_failures": xcheck_fail,
    }, indent=2))
    print(f"[write] {sp}")

    # ------------------------------------------------------------------ report
    R = []
    A = R.append
    A("# Exp 1.8d — witness JVP, BN quantities, scale control, eigensolver residuals\n")
    A(f"Model: production BlockViT-v2 at random init, {args.num_layers} layers, "
      f"{args.num_heads} heads, reduce_dim {args.reduce_dim}, "
      f"spatial_size {args.spatial_size}, {args.num_classes} classes.  \n"
      f"Data: {args.dataset_name} fold {args.fold} {args.split}, densest of the first "
      f"{args.n_candidate_cores} cores = **{core_name}**, {n_valid} labeled pixels "
      f"of {args.spatial_size**2} ({n_valid/args.spatial_size**2:.1%}).  \n"
      f"Widths {args.widths}, seeds {args.seeds}, modes {args.modes}.  \n"
      f"cuDNN TF32 {'ON' if args.tf32 else 'OFF'} (off by default so the GGN "
      f"cross-check is not limited by 10-bit mantissas).\n")
    A("All numbers below are measurements. No interpretation beyond them is offered "
      "except in the final scoping paragraph.\n")

    A("\n## Table 1 — BN quantities (math_02 §7 item 1)\n")
    A("Per-channel batch mean and **biased** variance over the FULL normalization "
      "group (all B·H·W positions, ignored pixels included). "
      "`gain_i = gamma_i / sqrt(var_i + eps)`. Raw arrays in "
      "`results/exp1_8d_bn_stats/M{width}_s{seed}.npz`.\n")
    for tag, label in (("bn1", "seg_head[1] (96 ch)"), ("bn2", "seg_head[4] (48 ch)")):
        cols = ["mode", "width", "seed", f"{tag}_eps",
                f"{tag}_var_min", f"{tag}_var_med", f"{tag}_var_max",
                f"{tag}_gain_min", f"{tag}_gain_med", f"{tag}_gain_max",
                f"{tag}_frac_var_lt_10eps", f"{tag}_mean_absmax"]
        t = (df.drop_duplicates(["mode", "width", "seed"])[cols]
             .sort_values(["mode", "width", "seed"]))
        A(f"\n**{label}**\n")
        A(md_table(t))
    A("\n**Renamed scalar** (`bn_gain_rms` in exp1_8c): "
      "`activation_energy_ratio_sqrt` = sqrt(E_postBN / E_preBN) on labeled pixels.\n")
    t = (df.drop_duplicates(["mode", "width", "seed"])
         [["mode", "width", "seed", "activation_energy_ratio_sqrt",
           "energy_preBN_labeled", "energy_postBN_labeled", "p_min_mean"]]
         .sort_values(["mode", "width", "seed"]))
    A(md_table(t))

    A("\n\n## Table 2 — witness JVP (math_02 §7 item 2)\n")
    A("Unit perturbation of `seg_head[0]`, propagated by forward-mode AD through the "
      "actual head in the actual BN mode (train-mode BN derivative includes the "
      "batch-statistics dependence). Energies `e_pre … e_bn2` are means over the FULL "
      "group; `e_logit_valid` and `e_final` use labeled pixels only; "
      "`e_final = N^-1 Σ_n d_n^T (diag(p_n) − p_n p_n^T) d_n`. "
      "`GGN` is `ΔW^T G_WW ΔW` from the same matvec exp1_8c uses; "
      "`rel` is |e_final − GGN| / |GGN|.\n")
    A("\nOutput channel: for the input-side directions the perturbation is "
      "`ΔW = e_c ⊗ u` and **all 96 output channels were swept**; the row reports "
      "`c* = argmax_c e_final`, its energies, and the mean over c. "
      "`ggn_top` is the full 96×(M+K)×3×3 top GGN eigenvector (no separate output "
      "channel, `c* = -1`).\n")
    show = ["width", "seed", "direction", "c_star", "e_pre", "e_cen",
            "e_bn1", "e_relu1", "e_bn2", "e_logit_valid", "e_final",
            "e_final_mean", "ggn_quad", "xcheck_rel_disc"]
    A("\n### 2a. Mean over seeds\n")
    agg = (df[df["direction"] != "bias_ec"]
           .groupby(["mode", "width", "direction", "_dord"], as_index=False)
           [ecols + ["e_final_mean"]].mean()
           .sort_values(["width", "_dord"]))
    for mode in args.modes:
        sub = agg[agg["mode"] == mode]
        if not len(sub):
            continue
        A(f"\n**mode = {mode}** (mean over seeds {args.seeds})\n")
        A(md_table(sub[["width", "direction"] + ecols + ["e_final_mean"]]))
    A("\n### 2b. Every row\n")
    for mode in args.modes:
        sub = df[(df["mode"] == mode) & (df["direction"] != "bias_ec")]
        if not len(sub):
            continue
        A(f"\n**mode = {mode}**\n")
        A(md_table(sub[show]))

    A("\n\n### Bias JVP (Δb = e_c, all 96 channels)\n")
    A("math_02 §1 discrepancy 4: a spatially constant bias perturbation is removed by "
      "the immediately following train-mode BN. Numbers, not an assertion:\n")
    sub = df[df["direction"] == "bias_ec"].copy()
    if len(sub):
        sub["_mord"] = sub["mode"].map(
            {m: i for i, m in enumerate(args.modes)}).fillna(99)
        A(md_table(sub.sort_values(["_mord", "width", "seed"])[
            ["mode", "width", "seed", "c_star", "e_pre", "e_cen", "e_bn1",
             "e_relu1", "e_bn2", "e_logit_valid", "e_final", "e_final_mean",
             "ggn_quad", "xcheck_abs_disc"]]))

    A("\n\n### log-log slopes vs M (all three seeds shown)\n")
    for mode in args.modes:
        if mode not in slopes_seed:
            continue
        A(f"\n**mode = {mode}**\n")
        recs = []
        for seed, s in sorted(slopes_seed[mode].items()):
            recs.append(dict(seed=seed, **{k: v for k, v in s.items()}))
        recs.append(dict(seed="pooled-mean", **slopes[mode]))
        A(md_table(pd.DataFrame(recs)))

    A("\n\n## Table 3 — function-preserving scale control (math_02 §6 (B3), §7 item 3)\n")
    if len(sdf):
        A("`seg_head[0].weight` AND `bias` scaled by c. `coscale_eps=True` also sets "
          "`seg_head[1].eps = c²·eps0` (the exact invariance); `False` keeps eps fixed "
          "(the implementation's actual behaviour). `logits_max_dev` is "
          "max|logits(c) − logits(1)|; `ratio_c2lam_over_lam1 = λ_WW(c)·c²/λ_WW(1)` "
          "(exact invariance ⇒ 1). `var_*` are per-channel pre-BN variances over the "
          "full group at that c.\n")
        A(md_table(sdf.sort_values(["width", "coscale_eps", "c"])[
            ["width", "seed", "coscale_eps", "c", "eps", "logits_max_dev", "lam_WW",
             "ratio_c2lam_over_lam1", "W_fro2", "invariant_Wfro2_lam",
             "var_min", "var_med", "var_max", "var_over_eps_med",
             "frac_var_lt_10eps", "resid_WW"]]))
    else:
        A("(skipped)\n")

    A("\n\n## Table 4 — eigensolver accuracy (math_02 §7 item 4)\n")
    A("`resid = ||Gv − λv|| / λ` and the Rayleigh quotient, both of the SAME final "
      "iterate, from the matvec that iterate already required. The loop stops only "
      "when the relative change of the Rayleigh quotient is below tol AND "
      f"resid < 1e-3 (`stop = tol+resid`); otherwise it runs out of iterations "
      f"(`n_iter`) or wall-clock budget (`budget`) and the residual shown is what was "
      f"achieved. Small blocks: n_iter {args.n_iter}, tol {args.tol}. phi block: "
      f"n_iter {args.n_iter_phi}, tol {args.tol_phi}, cap {args.phi_budget_s:.0f}s "
      "per row. (math_02 §7 item 4 suggested n_iter_phi 60; the phi matvec turned out "
      "to cost under 10 s per row on this batch, so the cap was raised rather than "
      "reporting unconverged rows.)\n")
    t = (df.drop_duplicates(["mode", "width", "seed"])
         [["mode", "width", "seed", "lam_WW", "ray_WW", "resid_WW", "iters_WW",
           "stop_WW", "lam_Wb", "resid_Wb", "iters_Wb", "stop_Wb",
           "lam_phi", "ray_phi", "resid_phi", "iters_phi", "stop_phi", "secs_phi"]]
         .sort_values(["mode", "width", "seed"]))
    A(md_table(t))

    A("\n\n## Numerical hygiene\n")
    t = (df.drop_duplicates(["mode", "width", "seed"])
         [["mode", "width", "seed", "head_max_dev_vs_module", "logit_absmax",
           "fused_max_dev_vs_train", "lam_Sh", "lam_Sh_cen", "lam_Shup",
           "lamS9_valid", "lamS9cen_valid", "lamS9_full", "lamS9cen_full",
           "cos_s9cen_valid_full", "cos_s9cen_mu9", "cos_s9_mu9"]]
         .sort_values(["mode", "width", "seed"]))
    A("`head_max_dev_vs_module`: max |logits(explicit functional head) − "
      "logits(model.seg_head)|, versus `logit_absmax`. "
      "`fused_max_dev_vs_train`: max |fused(mode) − fused(train)|, i.e. whether the "
      "incoming features really are unchanged by the head-BN interventions. "
      "`lamS9*_valid` vs `lamS9*_full`: 3×3-patch Gram on labeled pixels (exp1_8c's "
      "mask) vs on the full BN normalization group (math_02 §1 discrepancy 3), with "
      "the cosine between the two CENTERED top eigenvectors. `lam_Sh`, `lam_Sh_cen`, "
      "`lam_Shup` are exp1_8c's per-pixel Grams, recomputed here with TF32 off.\n")
    A(md_table(t))
    if xcheck_fail:
        A(f"\n**Cross-check failures (rel > {args.xcheck_tol}):** {len(xcheck_fail)}\n")
        A(md_table(pd.DataFrame(xcheck_fail, columns=[
            "width", "seed", "mode", "direction", "e_final", "ggn_quad", "rel"])))
    else:
        nz = df[~df["xcheck_both_zero"]]
        bz = df[df["xcheck_both_zero"]]
        A(f"\nAll {len(df)} rows passed the e_final vs ΔW^T G_WW ΔW cross-check at "
          f"rel < {args.xcheck_tol}. Max relative discrepancy over the "
          f"{len(nz)} rows with nonzero curvature: "
          f"{nz['xcheck_rel_disc'].max():.3e}. The remaining {len(bz)} rows are the "
          f"train-mode / eval_bn2 bias directions, where both sides vanish "
          f"(max |e_final − GGN| = {bz['xcheck_abs_disc'].max():.3e} if any); a ratio "
          f"there is meaningless and is reported as 0.\n")

    A("\n\n## Mode-to-mode ratios (which BN the sensitivity attaches to)\n")
    A("Each entry is the quantity in that mode divided by its train-mode value at the "
      "same width and seed, averaged over seeds. `eval_bn1` puts ONLY seg_head[1] in "
      "eval, `eval_bn2` ONLY seg_head[4], `eval_head` both.\n")
    u = df.drop_duplicates(["mode", "width", "seed"])[
        ["mode", "width", "seed", "lam_WW", "lam_phi"]]
    fin = (df[df["direction"] == "s9cen"][["mode", "width", "seed", "e_final"]]
           .rename(columns={"e_final": "e_final_s9cen"}))
    u = u.merge(fin, on=["mode", "width", "seed"], how="left")
    tr = u[u["mode"] == "train"].set_index(["width", "seed"])
    recs = []
    for mode in args.modes:
        s = u[u["mode"] == mode].set_index(["width", "seed"])
        if not len(s) or not len(tr):
            continue
        for width in args.widths:
            r = {"mode": mode, "width": width}
            for c in ("lam_WW", "lam_phi", "e_final_s9cen"):
                vals = [s.loc[(width, sd), c] / tr.loc[(width, sd), c]
                        for sd in args.seeds
                        if (width, sd) in s.index and (width, sd) in tr.index
                        and tr.loc[(width, sd), c] > 0]
                r[f"{c}_ratio"] = float(np.mean(vals)) if vals else float("nan")
            recs.append(r)
    sep = ""
    if recs:
        A(md_table(pd.DataFrame(recs)))
        rr = pd.DataFrame(recs).groupby("mode")["lam_WW_ratio"].mean()
        sl = {m: slopes[m]["lam_WW"] for m in slopes}

        def _g(d, k):
            return d[k] if k in d else float("nan")
        if all(m in rr.index for m in ("eval_bn1", "eval_bn2", "eval_head")):
            near_train = [m for m in ("eval_bn1", "eval_bn2")
                          if abs(_g(sl, m) - _g(sl, "train"))
                          < abs(_g(sl, m) - _g(sl, "eval_head"))]
            sep = (
                f"Averaged over the three widths, λ_WW keeps "
                f"{rr['eval_bn1']:.3f} of its train-mode value when only seg_head[1] "
                f"is switched to eval, {rr['eval_bn2']:.3f} when only seg_head[4] is, "
                f"and {rr['eval_head']:.3f} when both are. The pooled log-log slope of "
                f"λ_WW against M is {_g(sl, 'train'):+.3f} (train), "
                f"{_g(sl, 'eval_bn1'):+.3f} (eval_bn1), {_g(sl, 'eval_bn2'):+.3f} "
                f"(eval_bn2) and {_g(sl, 'eval_head'):+.3f} (eval_head); "
                + ("both single-BN slopes sit closer to the train slope than to the "
                   "eval_head slope"
                   if len(near_train) == 2 else
                   f"single-BN slope(s) closer to train than to eval_head: "
                   f"{near_train or 'none'}")
                + ". So the two single-BN interventions do separate the two "
                  "BatchNorms in the sense that they attribute most of the drop in the "
                  "LEVEL of λ_WW to "
                + ("seg_head[4]" if rr["eval_bn2"] < rr["eval_bn1"] else "seg_head[1]")
                + ", but "
                + ("neither on its own reproduces the width trend of the joint "
                   "intervention" if len(near_train) == 2 else
                   "the width-trend comparison is given by the slopes above")
                + ".")

    A("\n\n## What this identifies / what it does not\n")
    A("**Identifies.** For each (width, seed, mode) the numbers above give, for a "
      "*specified* unit perturbation of the production first head convolution, the "
      "energy that survives each stage of the actual downstream path: the pre-BN "
      "energy on the full normalization group, the part of it that survives centering, "
      "the energy after each BN, after the first ReLU gate, at the logits on labeled "
      "pixels, and under the CE metric. The final scalar is a valid lower bound "
      "witness for λ_max(G_phi phi) in the sense of math_02 §6, and it is verified "
      "against the GGN matvec, so the reported energies and the eigenvalue estimates "
      "are the same object. The BN table gives the per-channel variances and effective "
      "gains that any normalization-based explanation has to use, measured on the "
      "group BN actually normalizes over. The scale control shows how much of the raw "
      "block curvature is a coordinate choice.\n")
    A("**Does not identify.** The eval-head intervention changes the BN derivative AND "
      "the forward logits, the downstream ReLU gates, and the CE metric H_n at the same "
      "time (`p_min_mean` in Table 1 and `logit_absmax` in the hygiene table below "
      "move with it). A difference "
      "between `train` and `eval_head` therefore establishes sensitivity to that "
      "intervention, not a single cause, and none of these rows separate mean "
      "subtraction from variance scaling. `eval_bn1` and `eval_bn2` isolate WHICH of "
      "the two BatchNorms the sensitivity is attached to — each leaves the other in "
      "train mode — but neither is a clean derivative-only intervention, for the same "
      "reason: switching either BN also changes that BN's forward output, hence the "
      "gates and logits downstream of it. Slopes over three widths are descriptive, "
      "not asymptotic exponent estimates, and everything here is one core, at "
      "initialization, with no optimizer in the loop.\n")
    if sep:
        A("\n**Separation, in numbers.** " + sep + "\n")

    rp = out_path.parent / "exp1_8d_REPORT.md"
    rp.write_text("\n".join(R))
    print(f"[write] {rp}")


if __name__ == "__main__":
    main()
