"""Experiment 1.2 (v5): residual-overlap export for the shallow verified instance.

Answers request #4 of `review_packet/astra/math_01.md` (marked PENDING in
`review_packet/astra/claude_math_reply_01.md` §5, row 4):

    "For the existing shallow verified instance, export the actual residual
     overlap with the fast spatial kernel subspace and the restricted spectral
     output singular directions. If the residual lies mostly outside that
     subspace, stop treating a large top eigenvalue as a rate certificate.
     An instantaneous measurement is a diagnostic, not proof of persistence."

WHICH MODEL.  `prop:verified_instance` (paper/sections/supplement.tex L252) is
the linear per-pixel spectral map f_theta followed by a two-layer ReLU head
g_phi at Kaiming Gaussian init with a FIXED bias variance sigma_b > 0.  In this
repo that is

    CompositionModel(SpectralReduction(S, K),
                     SpatialMLP(K, n_classes, width, init_mode='theorem',
                                sigma_b=0.5))

`code/experiments/exp1_1v3.py` is the runner that instantiates it (arch='mlp',
`--init_mode theorem`), but it measures GGN block eigenvalues at INIT ONLY and
never trains.  `code/experiments/exp1_2v4.py` trains joint-vs-frozen but its
spatial heads are SpatialCNN / SpatialViT, i.e. NOT the verified instance.
This runner puts the two together: exp1_1v3's model, exp1_2v4's training
protocol (data, optimizer, schedule, joint/frozen arms), plus a residual
projection export at log-spaced checkpoints along the trajectory.

WHAT IS MEASURED, on a FIXED probe batch of N supervised sites and C classes
(so the logit space is R^{NC}), at each checkpoint:

  r(t)      = the CE gradient w.r.t. logits, stacked with ggn.py's normalization
              z = N^{-1/2} stack(yhat)  =>  r = N^{-1/2} (p - onehot).
              Then grad_b L = J_b^T r exactly, with J_b = N^{-1/2} d z / d b.
  K_theta   = J_theta J_theta^T   (NC x NC, dense)
  K_phi     = J_phi   J_phi^T     (NC x NC, dense)

  (i)   a_eff   = r^T K_theta r / ||r||^2      (= ||grad_theta L||^2 / ||r||^2)
        kap_eff = r^T K_phi   r / ||r||^2      (= ||grad_phi   L||^2 / ||r||^2)
  (ii)  lam_max(K_theta), lam_max(K_phi), lam_min(K_phi)
  (iii) fraction of ||r||^2 in the top-k eigenspace of K_phi, k in {1,5,20,100}
  (iv)  fraction of ||r||^2 in range(J_theta) and in its top-k left singular
        subspace, k in {1,5,20}
  (v)   instantaneous share r^T K_theta r / (r^T K_theta r + r^T K_phi r),
        the (A1) bound a_eff/(a_eff+kap_eff) -- identical by construction --
        and the two TOP-EIGENVALUE proxies that Counterexample 1.4 is about:
            lam_max(K_theta) / (lam_max(K_theta) + lam_max(K_phi))
            lam_max(K_theta) / (lam_max(K_theta) + lam_min(K_phi))
  (vi)  cumulative discrete share  sum_t ||g_theta||^2 / sum_t (||g_theta||^2 +
        ||g_phi||^2) from the per-step EGR log, up to the checkpoint and over
        the whole run.

  Because CE residual dynamics are r_dot = -H (K_theta + K_phi) r with
  H = blkdiag(diag(p_n) - p_n p_n^T), every Rayleigh quotient and extreme
  eigenvalue above is ALSO reported for the H-weighted kernels H^{1/2} K H^{1/2},
  so a reader can see whether the squared-loss reading is off by more than a
  constant.  Note lam_max(H^{1/2} K_b H^{1/2}) == lam_max(G_bb), the GGN block,
  which is checked against `hessian/ggn.py` at every measurement.

FROZEN ARM.  exp1_2v4 freezes by setting requires_grad=False, which makes the
EGR numerator 0 by construction.  Here the spectral params keep
requires_grad=True but are excluded from the optimizer, so the trajectory is
bit-identical to exp1_2v4's frozen arm while the logged theta gradient is the
COUNTERFACTUAL signal a frozen encoder would have received.  That is what makes
(vi) meaningful on both arms.

OUTPUT
  results/exp1_2v5_residual_export.csv       one row per width x seed x arm x checkpoint
  results/exp1_2v5/egr_{arm}_D{w}_s{seed}.csv    per-step gradient norms
  results/exp1_2v5/metrics_{arm}_D{w}_s{seed}.csv per-epoch train/test

Usage
    python code/experiments/exp1_2v5_residual_export.py
    python code/experiments/exp1_2v5_residual_export.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CODE_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# --- config + helpers imported from the existing runners (no model code copied)
from experiments.exp1_2v4 import (  # noqa: E402
    S, K, H, W, N_CLASSES, N_TRAIN, N_TEST, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, DATA_NOISE, per_pixel_ce, evaluate,
)
from synthetic.data import make_problem  # noqa: E402
from synthetic.models import (  # noqa: E402
    SpectralReduction, SpatialMLP, CompositionModel,
)
from egr.callback import EGRLogger  # noqa: E402
from hessian.ggn import top_eigenvalue_ggn_block  # noqa: E402


RESULTS_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/results")
OUT_DIR = RESULTS_DIR / "exp1_2v5"
SUMMARY_PATH = RESULTS_DIR / "exp1_2v5_residual_export.csv"

# The verified instance's head init: Kaiming Gaussian weights N(0, 2/fan_in),
# biases N(0, sigma_b^2) with sigma_b FIXED across widths (prop:verified_instance
# requires sigma_b > 0; PyTorch's default bias variance Theta(1/fan_in) does not
# satisfy it).
INIT_MODE = "theorem"
SIGMA_B = 0.5

WIDTHS = [128, 512, 2048]
SEEDS = [42, 43, 44]
ARMS = ["joint", "frozen"]

N_PROBE_SITES = 256          # N in the docstring; NC = N_PROBE_SITES * N_CLASSES
N_PROBE_IMAGES = 16          # probe sites are drawn from this many train images
PROBE_RNG_SEED = 12345       # fixed site selection, independent of the run seed
IGNORE_INDEX = 255
CLASS_DIM = 3                # models are channels-LAST: logits (B, H, W, C)
TOPK_EIG = [1, 5, 20, 100]   # for K_phi eigenspaces
TOPK_SV = [1, 5, 20]         # for J_theta left singular subspaces
RANK_RTOL = 1e-10            # numerical-rank cutoff, relative to sigma_max
RANK_RTOL_LOOSE = 1e-6       # a second, stricter-in-practice cutoff


# ------------------------------------------------------------------ #
# model
# ------------------------------------------------------------------ #
def build_verified_instance(width: int) -> CompositionModel:
    """prop:verified_instance: linear f_theta + two-layer ReLU g_phi, Kaiming."""
    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialMLP(K=K, n_classes=N_CLASSES, width=width,
                         init_mode=INIT_MODE, sigma_b=SIGMA_B)
    return CompositionModel(spectral, spatial)


# ------------------------------------------------------------------ #
# probe batch
# ------------------------------------------------------------------ #
def make_probe(X_tr: torch.Tensor, y_tr: torch.Tensor,
               n_sites: int = N_PROBE_SITES, n_images: int = N_PROBE_IMAGES):
    """Fixed probe: `n_sites` supervised sites out of `n_images` train images.

    Non-probe sites are set to IGNORE_INDEX so that hessian/ggn.py, which counts
    valid pixels itself, sees exactly the same N and the same site set.

    Returns (probe_X, probe_y_masked, flat_site_index) with flat_site_index
    indexing the (B*H*W) flattening of the probe images.
    """
    Xp = X_tr[:n_images].clone()
    yp = y_tr[:n_images].clone()
    n_all = Xp.shape[0] * H * W
    if n_sites > n_all:
        raise ValueError(f"n_sites={n_sites} exceeds {n_all} available sites")
    g = torch.Generator().manual_seed(PROBE_RNG_SEED)
    perm = torch.randperm(n_all, generator=g)[:n_sites]
    site_idx = torch.sort(perm).values.to(Xp.device)

    masked = torch.full((n_all,), IGNORE_INDEX, dtype=torch.long, device=Xp.device)
    masked[site_idx] = yp.reshape(-1)[site_idx]
    return Xp, masked.reshape(Xp.shape[0], H, W), site_idx


# ------------------------------------------------------------------ #
# dense Jacobian
# ------------------------------------------------------------------ #
def _dense_jacobian(flat_logits: torch.Tensor, params, chunk: int = 128):
    """Rows of d(flat_logits)/d(params), shape (NC, P). Graph must be alive."""
    NC = flat_logits.numel()
    P = sum(p.numel() for p in params)
    J = torch.empty(NC, P, dtype=flat_logits.dtype, device=flat_logits.device)
    for start in range(0, NC, chunk):
        end = min(start + chunk, NC)
        nb = end - start
        E = torch.zeros(nb, NC, dtype=flat_logits.dtype, device=flat_logits.device)
        E[torch.arange(nb, device=E.device),
          torch.arange(start, end, device=E.device)] = 1.0
        try:
            gs = torch.autograd.grad(flat_logits, params, grad_outputs=E,
                                     is_grads_batched=True, retain_graph=True)
            J[start:end] = torch.cat([g.reshape(nb, -1) for g in gs], dim=1)
        except Exception:                                    # vmap fallback
            for i in range(nb):
                gs = torch.autograd.grad(flat_logits, params,
                                         grad_outputs=E[i], retain_graph=True)
                J[start + i] = torch.cat([g.reshape(-1) for g in gs])
    return J


def _block_diag_from_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """(N, C, C) per-site blocks -> dense (NC, NC) block-diagonal matrix."""
    N, C, _ = blocks.shape
    out = torch.zeros(N * C, N * C, dtype=blocks.dtype, device=blocks.device)
    base = torch.arange(N, device=blocks.device) * C
    for a in range(C):
        for b in range(C):
            out[base + a, base + b] = blocks[:, a, b]
    return out


def _subspace_fraction(basis: torch.Tensor, r: torch.Tensor, r_sq: float) -> float:
    """||basis^T r||^2 / ||r||^2 for an orthonormal-column `basis`."""
    if basis.shape[1] == 0 or r_sq <= 0:
        return float("nan")
    return float((basis.T @ r).pow(2).sum().item() / r_sq)


# ------------------------------------------------------------------ #
# the measurement
# ------------------------------------------------------------------ #
def measure(model, probe_X, probe_y, site_idx, device) -> dict:
    """All quantities (i)-(v) plus H-weighted twins and validation residuals."""
    mdl = copy.deepcopy(model).to(device).double()
    for p in mdl.parameters():
        p.requires_grad_(True)
    Xp = probe_X.to(device).double()
    yp = probe_y.to(device)

    theta_params = list(mdl.spectral.parameters())
    phi_params = list(mdl.spatial.parameters())

    logits = mdl(Xp)                                  # (B, H, W, C)
    C = logits.shape[-1]
    flat = logits.reshape(-1, C)[site_idx]            # (N, C)
    N = flat.shape[0]
    NC = N * C
    flat_logits = flat.reshape(-1)                    # site-major, class-minor

    with torch.no_grad():
        p_sm = torch.softmax(flat.detach(), dim=-1)                   # (N, C)
        y_sel = yp.reshape(-1)[site_idx]
        onehot = F.one_hot(y_sel, num_classes=C).to(p_sm.dtype)
        r = ((p_sm - onehot) / math.sqrt(N)).reshape(-1)              # (NC,)
        r_sq = float(r.pow(2).sum().item())

    # normalized Jacobians  J = N^{-1/2} d(yhat)/d(param)
    Jt = _dense_jacobian(flat_logits, theta_params) / math.sqrt(N)
    Jp = _dense_jacobian(flat_logits, phi_params) / math.sqrt(N)

    Kt = Jt @ Jt.T
    Kp = Jp @ Jp.T

    a_eff = float((r @ (Kt @ r)).item()) / r_sq
    kap_eff = float((r @ (Kp @ r)).item()) / r_sq

    evt = torch.linalg.eigvalsh(0.5 * (Kt + Kt.T))
    ewp, evp = torch.linalg.eigh(0.5 * (Kp + Kp.T))
    lam_max_t = float(evt[-1].item())
    lam_max_p = float(ewp[-1].item())
    lam_min_p = float(ewp[0].item())

    # (iii) residual mass in the top-k eigenspace of K_phi
    order = torch.argsort(ewp, descending=True)
    Qp = evp[:, order]
    frac_eig = {k: _subspace_fraction(Qp[:, :min(k, NC)], r, r_sq)
                for k in TOPK_EIG}

    # (iv) residual mass in range(J_theta) / top-k left singular subspace
    U, Sv, _ = torch.linalg.svd(Jt, full_matrices=False)
    smax = float(Sv[0].item()) if Sv.numel() else 0.0
    rank_tight = int((Sv > smax * RANK_RTOL).sum().item())
    rank_loose = int((Sv > smax * RANK_RTOL_LOOSE).sum().item())
    frac_range = _subspace_fraction(U[:, :rank_tight], r, r_sq)
    frac_range_loose = _subspace_fraction(U[:, :rank_loose], r, r_sq)
    frac_sv = {k: _subspace_fraction(U[:, :min(k, U.shape[1])], r, r_sq)
               for k in TOPK_SV}

    # H-weighted kernels: H^{1/2} K H^{1/2}
    with torch.no_grad():
        Hb = (torch.diag_embed(p_sm)
              - p_sm.unsqueeze(-1) * p_sm.unsqueeze(-2))          # (N, C, C)
        hw, hv = torch.linalg.eigh(0.5 * (Hb + Hb.transpose(-1, -2)))
        hw = hw.clamp_min(0.0).sqrt()
        Hh_blocks = hv @ torch.diag_embed(hw) @ hv.transpose(-1, -2)
        Hh = _block_diag_from_blocks(Hh_blocks)                    # (NC, NC)

    KtH = Hh @ Kt @ Hh
    KpH = Hh @ Kp @ Hh
    a_eff_H = float((r @ (KtH @ r)).item()) / r_sq
    kap_eff_H = float((r @ (KpH @ r)).item()) / r_sq
    evtH = torch.linalg.eigvalsh(0.5 * (KtH + KtH.T))
    ewpH = torch.linalg.eigvalsh(0.5 * (KpH + KpH.T))
    lam_max_tH = float(evtH[-1].item())
    lam_max_pH = float(ewpH[-1].item())
    lam_min_pH = float(ewpH[0].item())

    # ---- validation ---------------------------------------------------- #
    loss = F.cross_entropy(mdl(Xp).reshape(-1, C), yp.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction="mean")
    gt = torch.autograd.grad(loss, theta_params, retain_graph=True)
    gp = torch.autograd.grad(loss, phi_params, retain_graph=False)
    gt_sq = float(sum((g * g).sum() for g in gt).item())
    gp_sq = float(sum((g * g).sum() for g in gp).item())
    chk_theta = abs(a_eff * r_sq - gt_sq) / max(gt_sq, 1e-300)
    chk_phi = abs(kap_eff * r_sq - gp_sq) / max(gp_sq, 1e-300)

    # lam_max(H^{1/2} K_bb H^{1/2}) must equal lam_max(G_bb) from hessian/ggn.py
    lam_ggn_phi = top_eigenvalue_ggn_block(
        mdl, Xp, yp, mdl.spatial, n_iter=300, tol=1e-13,
        class_dim=CLASS_DIM, ignore_index=IGNORE_INDEX, seed=0)
    chk_ggn = abs(lam_max_pH - lam_ggn_phi) / max(abs(lam_ggn_phi), 1e-300)

    denom = a_eff + kap_eff
    denom_H = a_eff_H + kap_eff_H
    row = {
        "N_sites": N, "C_classes": C, "NC": NC,
        "P_theta": Jt.shape[1], "P_phi": Jp.shape[1],
        "loss_probe": float(loss.item()),
        "r_sq": r_sq,
        "a_eff": a_eff, "kappa_eff": kap_eff,
        "lam_max_Ktheta": lam_max_t,
        "lam_max_Kphi": lam_max_p, "lam_min_Kphi": lam_min_p,
        "share_inst": a_eff / denom if denom > 0 else float("nan"),
        "share_A1_bound": a_eff / denom if denom > 0 else float("nan"),
        "share_proxy_lammax": (lam_max_t / (lam_max_t + lam_max_p)
                               if (lam_max_t + lam_max_p) > 0 else float("nan")),
        "share_proxy_lammin": (lam_max_t / (lam_max_t + lam_min_p)
                               if (lam_max_t + lam_min_p) > 0 else float("nan")),
        "frac_range_Jtheta": frac_range,
        "frac_range_Jtheta_rtol1e-6": frac_range_loose,
        "rank_Jtheta": rank_tight, "rank_Jtheta_rtol1e-6": rank_loose,
        # H-weighted twins
        "a_eff_H": a_eff_H, "kappa_eff_H": kap_eff_H,
        "lam_max_Ktheta_H": lam_max_tH,
        "lam_max_Kphi_H": lam_max_pH, "lam_min_Kphi_H": lam_min_pH,
        "share_inst_H": a_eff_H / denom_H if denom_H > 0 else float("nan"),
        "share_proxy_lammax_H": (lam_max_tH / (lam_max_tH + lam_max_pH)
                                 if (lam_max_tH + lam_max_pH) > 0 else float("nan")),
        "share_proxy_lammin_H": (lam_max_tH / (lam_max_tH + lam_min_pH)
                                 if (lam_max_tH + lam_min_pH) > 0 else float("nan")),
        # validation
        "chk_grad_theta_relerr": chk_theta,
        "chk_grad_phi_relerr": chk_phi,
        "chk_ggn_phi_relerr": chk_ggn,
        "lam_max_Gphiphi_ggnpy": float(lam_ggn_phi),
    }
    for k in TOPK_EIG:
        row[f"frac_r_topk{k}_Kphi"] = frac_eig[k]
    for k in TOPK_SV:
        row[f"frac_r_topk{k}_Jtheta_sv"] = frac_sv[k]

    del Jt, Jp, Kt, Kp, KtH, KpH, Hh
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# ------------------------------------------------------------------ #
# training with checkpointed measurement
# ------------------------------------------------------------------ #
def checkpoint_steps(total_steps: int, n_log: int) -> list[int]:
    lg = np.unique(np.round(np.logspace(0, math.log10(total_steps), n_log))
                   ).astype(int)
    return sorted(set([0] + [int(v) for v in lg] + [int(total_steps)]))


def run_one(width: int, seed: int, arm: str, device: torch.device,
            epochs: int, n_log: int,
            n_sites: int = N_PROBE_SITES, n_images: int = N_PROBE_IMAGES
            ) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_verified_instance(width).to(device)

    X_tr, y_tr = make_problem(N_TRAIN, S=S, H=H, W=W, n_classes=N_CLASSES,
                              alpha=1.0, beta=1.0, noise=DATA_NOISE, seed=seed)
    X_te, y_te = make_problem(N_TEST, S=S, H=H, W=W, n_classes=N_CLASSES,
                              alpha=1.0, beta=1.0, noise=DATA_NOISE,
                              seed=seed + 1000)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)

    probe_X, probe_y, site_idx = make_probe(X_tr, y_tr, n_sites, n_images)

    # frozen == spectral excluded from the optimizer (trajectory identical to
    # exp1_2v4's requires_grad=False arm) but grads still computed so the EGR
    # numerator is the counterfactual signal, not a structural zero.
    if arm == "frozen":
        trainable = [p for p in model.spatial.parameters()]
    else:
        trainable = [p for p in model.parameters()]
    opt = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    logger = EGRLogger(model)

    steps_per_epoch = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = epochs * steps_per_epoch
    ckpts = set(checkpoint_steps(total_steps, n_log))

    rows: list[dict] = []
    metrics_rows: list[dict] = []
    egr_sq_theta = 0.0
    egr_sq_phi = 0.0
    global_step = 0

    def take_measurement(step, epoch):
        te_loss, te_acc = evaluate(model, X_te, y_te, device)
        row = measure(model, probe_X, probe_y, site_idx, device)
        cum = (egr_sq_theta / (egr_sq_theta + egr_sq_phi)
               if (egr_sq_theta + egr_sq_phi) > 0 else float("nan"))
        row.update({"width": width, "seed": seed, "arm": arm,
                    "step": int(step), "epoch": int(epoch),
                    "total_steps": int(total_steps),
                    "share_cum_discrete": cum,
                    "test_loss": te_loss, "test_acc": te_acc})
        rows.append(row)
        print(f"    [D={width} s={seed} {arm}] step {step:5d}  "
              f"a_eff={row['a_eff']:.3e}  kap_eff={row['kappa_eff']:.3e}  "
              f"share={row['share_inst']:.4f}  "
              f"proxy_max={row['share_proxy_lammax']:.4f}  "
              f"top1={row['frac_r_topk1_Kphi']:.4f}  "
              f"top20={row['frac_r_topk20_Kphi']:.4f}", flush=True)

    for epoch in range(epochs):
        idx = torch.randperm(N_TRAIN, device=device)
        epoch_losses = []
        for s in range(steps_per_epoch):
            if global_step in ckpts:
                take_measurement(global_step, epoch)
            batch_idx = idx[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            xb, yb = X_tr[batch_idx], y_tr[batch_idx]
            # model.zero_grad, NOT opt.zero_grad: in the frozen arm the spectral
            # params are deliberately absent from the optimizer, and
            # opt.zero_grad() only clears params it owns -- their .grad would
            # otherwise ACCUMULATE over all steps and the EGR numerator would be
            # a running sum instead of a per-step gradient.
            model.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = per_pixel_ce(logits, yb)
            loss.backward()
            rec = logger.log_step(global_step)
            egr_sq_theta += rec["grad_theta_norm"] ** 2
            egr_sq_phi += rec["grad_phi_norm"] ** 2
            opt.step()
            epoch_losses.append(loss.item())
            global_step += 1
        te_loss, te_acc = evaluate(model, X_te, y_te, device)
        metrics_rows.append({"epoch": epoch,
                             "train_loss": float(np.mean(epoch_losses)),
                             "test_loss": te_loss, "test_acc": te_acc})

    take_measurement(global_step, epochs)

    egr_df = logger.to_dataframe()
    full_cum = (egr_sq_theta / (egr_sq_theta + egr_sq_phi)
                if (egr_sq_theta + egr_sq_phi) > 0 else float("nan"))
    for r in rows:
        r["share_cum_discrete_fullrun"] = full_cum
    return rows, egr_df, pd.DataFrame(metrics_rows)


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--arms", type=str, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_log", type=int, default=None,
                    help="log-spaced checkpoints in addition to t=0")
    ap.add_argument("--n_sites", type=int, default=N_PROBE_SITES,
                    help="N: supervised probe sites; logit space is R^{N*C}")
    ap.add_argument("--n_probe_images", type=int, default=N_PROBE_IMAGES)
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for per-run artifact filenames")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dflt = dict(widths=WIDTHS, seeds=SEEDS, arms=ARMS, epochs=EPOCHS, n_log=10,
                out_csv=str(SUMMARY_PATH))
    if args.smoke:
        dflt.update(widths=[128], seeds=[42], arms=["joint"], epochs=3, n_log=3,
                    out_csv=str(RESULTS_DIR / "exp1_2v5_residual_export_smoke.csv"))
    for key, val in dflt.items():          # explicit CLI values always win
        if getattr(args, key) is None:
            setattr(args, key, val)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[exp1_2v5] device={device}  init_mode={INIT_MODE} sigma_b={SIGMA_B}")
    print(f"[exp1_2v5] model = SpectralReduction(S={S},K={K}) + "
          f"SpatialMLP(K={K},C={N_CLASSES},D=.) == prop:verified_instance")
    print(f"[exp1_2v5] probe: N={args.n_sites} sites, C={N_CLASSES} classes "
          f"-> NC={args.n_sites * N_CLASSES}")
    print(f"[exp1_2v5] widths={args.widths} seeds={args.seeds} arms={args.arms} "
          f"epochs={args.epochs}")

    all_rows: list[dict] = []
    t0 = time.time()
    for arm in args.arms:
        for width in args.widths:
            for seed in args.seeds:
                print(f"[exp1_2v5] arm={arm} D={width} seed={seed}", flush=True)
                rows, egr_df, met_df = run_one(
                    width, seed, arm, device, args.epochs, args.n_log,
                    args.n_sites, args.n_probe_images)
                all_rows.extend(rows)
                egr_df.to_csv(
                    OUT_DIR / f"egr_{arm}_D{width}_s{seed}{args.tag}.csv",
                    index=False)
                met_df.to_csv(
                    OUT_DIR / f"metrics_{arm}_D{width}_s{seed}{args.tag}.csv",
                    index=False)
                pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    lead = ["arm", "width", "seed", "step", "epoch", "total_steps"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(args.out_csv, index=False)
    print(f"\n[exp1_2v5] wrote {args.out_csv}  ({len(df)} rows, "
          f"{time.time() - t0:.1f} s)")
    print(f"[exp1_2v5] max validation rel-errors: "
          f"grad_theta={df['chk_grad_theta_relerr'].max():.3e}  "
          f"grad_phi={df['chk_grad_phi_relerr'].max():.3e}  "
          f"ggn_phi={df['chk_ggn_phi_relerr'].max():.3e}")


if __name__ == "__main__":
    main()
