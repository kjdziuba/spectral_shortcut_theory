#!/usr/bin/env python3
"""
Experiment 3B (v1, refocus 2026-09-10): real centre spectra with constructed
context. Design: paper/REFOCUS_PLAN_2026-09-10.md §5 (Astra refocus_01 §3).

Honest one-sentence description (Astra): "We test contextual competition using
measured centre spectra and their recorded labels, with a synthetic
class-associated spatial intensity pattern along a training-estimated dominant
spectral direction in the neighbourhood."

Data: breast QCL, fold-0 patient-level split (115 train / 28 val / 26 test
cores), per-core NPZ with X_raw/X_d1/X_d2 (H, W, 314), y in {0..4}
(1 NormalEpi, 2 NormalStroma, 3 CancerEpi, 4 CAS; 0 unlabelled), tissue_mask.
Binary task on a preselected pair (default CancerEpi (+1) vs CAS (-1)).
Feature vector = [raw, d1, d2] (942), standardized per feature with TRAIN
statistics; v1 = first principal direction of the standardized TRAIN pixels
(frozen). Everything below is estimated on training patients only.

Patch: centre = a real pixel (standardized spectrum, recorded label); eight
neighbours = real DONOR spectra sampled uniformly from the same split side
(other cores), independent of the centre label, plus the constructed cue
    gamma * (code * s + tau * eta) * v1        per neighbour,
code = y_centre (informative, s = +1), -y_centre (reversed, s = -1), or an
independent random label (uninformative; same marginal amplitude and noise).
gamma = g * sqrt(lambda_1) (g fixed), tau calibrated on training-side
calibration data so that the neighbourhood oracle (LDA on the eight v1
projections) matches the centre-only oracle (ridge LDA on the 942-dim centre).

Stages
  --build-cache   subsample labelled pixels per core/split -> results/exp3/cache_fold0.npz
  --calibrate     standardization, v1, oracles, tau, random-encoder readiness -> calibration.json
  (training arms are added after the readiness table is reviewed)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "exp3"
CACHE = OUT_DIR / "cache_fold0.npz"
CALIB = OUT_DIR / "calibration.json"
DATA_DIR = Path("/mnt/hdd2/u37314kd/data_breast_v2_pca23")
SPLITS = DATA_DIR / "splits_fold0.json"

# ---- design constants (REFOCUS_PLAN §5) ----
PAIR = {3: +1.0, 4: -1.0}          # CancerEpi -> +1, CAS -> -1   [author to confirm]
N_PER_CORE_CLASS = {"train": 600, "val": 300, "test": 300}
S_FEAT = 942
K = 12
G_AMP = 3.0                         # gamma = G_AMP * sqrt(lambda_1): "large" relative to natural v1 variation
CAL_SEED = 0
RIDGE = 1e-2                        # relative ridge for the 942-dim LDA


def load_split_ids() -> dict:
    return json.load(open(SPLITS))


def build_cache() -> None:
    splits = load_split_ids()
    rng = np.random.default_rng(1234)
    Xs, ys, cores, sides = [], [], [], []
    t0 = time.time()
    for side in ("train", "val", "test"):
        for cid in splits[side]:
            f = DATA_DIR / f"{cid}.npz"
            if not f.exists():
                print(f"[cache] missing {f.name}", flush=True)
                continue
            z = np.load(f)
            y = z["y"]; mask = z["tissue_mask"].astype(bool)
            feats = None
            for lab, sgn in PAIR.items():
                idx = np.argwhere((y == lab) & mask)
                if len(idx) == 0:
                    continue
                take = min(len(idx), N_PER_CORE_CLASS[side])
                sel = idx[rng.choice(len(idx), take, replace=False)]
                if feats is None:
                    feats = np.concatenate([z["X_raw"], z["X_d1"], z["X_d2"]], axis=-1)  # (H, W, 942)
                Xs.append(feats[sel[:, 0], sel[:, 1]].astype(np.float32))
                ys.append(np.full(take, sgn, dtype=np.float32))
                cores.append(np.full(take, cid))
                sides.append(np.full(take, side))
            print(f"[cache] {side} {cid}: {sum(len(a) for a in ys[-len(PAIR):]) if feats is not None else 0} px "
                  f"({time.time() - t0:.0f}s)", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys); core = np.concatenate(cores); side = np.concatenate(sides)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, core=core, side=side)
    for s in ("train", "val", "test"):
        m = side == s
        print(f"[cache] {s}: {m.sum()} px, +1: {(y[m] > 0).sum()}, -1: {(y[m] < 0).sum()}, cores {len(np.unique(core[m]))}")
    print(f"[cache] wrote {CACHE} ({X.nbytes / 1e9:.2f} GB raw)")


# ------------------------------------------------------------------ helpers ---
def lda_fit(F: torch.Tensor, y: torch.Tensor, ridge_rel: float):
    F = F.double(); pos = y > 0
    mu_p = F[pos].mean(0); mu_n = F[~pos].mean(0)
    Xc = torch.cat([F[pos] - mu_p, F[~pos] - mu_n])
    Sigma = Xc.T @ Xc / Xc.shape[0]
    d = Sigma.shape[0]
    Sigma = Sigma + ridge_rel * Sigma.diagonal().mean() * torch.eye(d, dtype=Sigma.dtype, device=Sigma.device)
    w = torch.linalg.solve(Sigma, mu_p - mu_n)
    thr = w @ (mu_p + mu_n) / 2.0
    return w, thr


def lda_eval(w, thr, F: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.sign(F.double() @ w - thr)
    return float((pred == y.double()).double().mean())


def context_features(donors_v1: torch.Tensor, y_code: torch.Tensor, gamma: float, tau: float,
                     eta: torch.Tensor) -> torch.Tensor:
    """donors_v1: (n, 8) v1-projections of the donor spectra; y_code: (n,) code in {-1,+1};
    eta: (n, 8) N(0,1). Returns the eight neighbour v1-projections with the constructed cue."""
    return donors_v1 + gamma * (y_code[:, None] + tau * eta)


def calibrate(device: torch.device) -> dict:
    z = np.load(CACHE, allow_pickle=True)
    X = torch.from_numpy(z["X"]); y = torch.from_numpy(z["y"]); side = z["side"]; core = z["core"]
    tr = torch.from_numpy(side == "train")
    Xtr = X[tr]
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-8
    Xs = ((X - mean) / std).to(device)             # standardized, all sides
    ytr_all = y[tr].to(device); Xtr_s = Xs[tr.to(device)]
    # v1 from a training subsample (frozen)
    g = torch.Generator().manual_seed(CAL_SEED)
    sub = torch.randperm(Xtr_s.shape[0], generator=g)[:40000].to(device)
    A = Xtr_s[sub].double()
    A = A - A.mean(0)
    U, Sv, Vh = torch.linalg.svd(A, full_matrices=False)
    v1 = Vh[0].float()
    lam = (Sv ** 2) / (A.shape[0] - 1)
    var_frac_v1 = float(lam[0] / lam.sum())
    lam1 = float(lam[0])
    # class contrast alignment with v1 (descriptive)
    mu_p = Xtr_s[ytr_all > 0].mean(0); mu_n = Xtr_s[ytr_all < 0].mean(0)
    contrast = mu_p - mu_n
    cos_contrast_v1 = float((contrast @ v1) / (contrast.norm() + 1e-12))
    print(f"[calib] train px {Xtr_s.shape[0]}; v1 variance fraction {var_frac_v1:.3f}; "
          f"lambda_1 {lam1:.2f}; |cos(contrast, v1)| {abs(cos_contrast_v1):.3f}")

    # training-side calibration halves (by core, so halves are patient-disjoint)
    tr_cores = np.unique(core[side == "train"])
    rng = np.random.default_rng(CAL_SEED)
    rng.shuffle(tr_cores)
    half_a = set(tr_cores[: len(tr_cores) // 2])
    in_a = torch.from_numpy(np.array([c in half_a for c in core[side == "train"]])).to(device)
    Fa, ya = Xtr_s[in_a], ytr_all[in_a]; Fb, yb = Xtr_s[~in_a], ytr_all[~in_a]

    # centre-only oracle: ridge LDA on the 942-dim standardized spectrum (fit A, eval B; and val)
    w, thr = lda_fit(Fa, ya, RIDGE)
    acc_centre_B = lda_eval(w, thr, Fb, yb)
    val = torch.from_numpy(side == "val").to(device)
    acc_centre_val = lda_eval(w, thr, Xs[val], y.to(device)[val])
    print(f"[calib] centre-only oracle (ridge LDA, 942-d): calib-B {acc_centre_B:.4f}, val {acc_centre_val:.4f}")

    # constructed context: donors independent of the centre label (uniform over the other half's pixels)
    def make_ctx(F_side: torch.Tensor, y_side: torch.Tensor, seed: int, tau: float, gamma: float,
                 mode: str = "informative"):
        gg = torch.Generator(device="cpu").manual_seed(seed)
        n = F_side.shape[0]
        donor_idx = torch.randint(0, n, (n, 8), generator=gg).to(device)
        donors_v1 = (F_side @ v1)[donor_idx]            # (n, 8) natural v1 projections of donors
        eta = torch.randn(n, 8, generator=gg).to(device)
        if mode == "informative":
            code = y_side
        elif mode == "reversed":
            code = -y_side
        else:
            code = (torch.randint(0, 2, (n,), generator=gg).float() * 2 - 1).to(device)
        return context_features(donors_v1, code, gamma, tau, eta)

    gamma = G_AMP * math.sqrt(lam1)
    target = acc_centre_B
    # neighbourhood oracle: LDA on the eight v1 projections (fit A, eval B); bisection on tau
    def ctx_oracle(tau: float) -> float:
        Ca = make_ctx(Fa, ya, 11, tau, gamma); Cb = make_ctx(Fb, yb, 12, tau, gamma)
        w2, t2 = lda_fit(Ca, ya, 1e-6)
        return lda_eval(w2, t2, Cb, yb)
    acc0 = ctx_oracle(0.0)
    print(f"[calib] gamma = {G_AMP} sqrt(lambda_1) = {gamma:.2f}; context oracle at tau=0: {acc0:.4f}; target {target:.4f}")
    lo, hi = 0.0, 8.0
    tau = None
    if acc0 <= target:
        tau = 0.0
        print("[calib] context oracle cannot reach the centre-only oracle even without context noise; tau = 0 "
              "(report the imbalance)")
    else:
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            a = ctx_oracle(mid)
            if a > target:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
    acc_ctx = ctx_oracle(tau)
    print(f"[calib] tau = {tau:.4f} -> context oracle {acc_ctx:.4f}")

    # random-encoder readiness (K x 942, N(0, 1/942)): centre probe (K feats) and patch probe (9K feats)
    readiness = {}
    for s in (0, 1, 2):
        torch.manual_seed(s)
        Wenc = torch.randn(K, S_FEAT, device=device) / math.sqrt(S_FEAT)
        Za, Zb = Fa @ Wenc.T, Fb @ Wenc.T
        wc, tc = lda_fit(Za, ya, 1e-4)
        acc_c = lda_eval(wc, tc, Zb, yb)
        # patch: encoded donors + cue along v1 -> encoded neighbours = W(donor + gamma(code+tau eta)v1)
        def enc_patch(F_side, y_side, seed):
            gg = torch.Generator(device="cpu").manual_seed(seed)
            n = F_side.shape[0]
            donor_idx = torch.randint(0, n, (n, 8), generator=gg).to(device)
            eta = torch.randn(n, 8, generator=gg).to(device)
            Zd = (F_side @ Wenc.T)[donor_idx]                       # (n, 8, K)
            cue = gamma * (y_side[:, None] + tau * eta)              # (n, 8)
            Zn = Zd + cue[:, :, None] * (Wenc @ v1)[None, None, :]   # (n, 8, K)
            return torch.cat([(F_side @ Wenc.T)[:, None, :], Zn], dim=1).reshape(n, 9 * K)
        Pa, Pb = enc_patch(Fa, ya, 21), enc_patch(Fb, yb, 22)
        wp, tp = lda_fit(Pa, ya, 1e-4)
        acc_p = lda_eval(wp, tp, Pb, yb)
        # patch with centre only (neighbours' cue removed) and neighbours only
        readiness[s] = dict(centre=acc_c, patch=acc_p)
        print(f"[calib] random encoder seed {s}: centre probe {acc_c:.4f}, patch probe {acc_p:.4f}")

    rec = dict(pair={str(k): v for k, v in PAIR.items()}, S=S_FEAT, K=K, g_amp=G_AMP, gamma=gamma, tau=tau,
               lambda_1=lam1, var_frac_v1=var_frac_v1, cos_contrast_v1=cos_contrast_v1,
               centre_oracle_calibB=acc_centre_B, centre_oracle_val=acc_centre_val,
               context_oracle=acc_ctx, context_oracle_tau0=acc0, readiness_by_seed=readiness,
               n_train=int(tr.sum()), n_val=int((side == "val").sum()), n_test=int((side == "test").sum()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(rec, indent=2))
    np.savez(OUT_DIR / "preprocessing_fold0.npz", mean=mean.numpy(), std=std.numpy(), v1=v1.cpu().numpy())
    print(f"[calib] wrote {CALIB}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    if args.build_cache:
        build_cache()
    if args.calibrate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        calibrate(device)


if __name__ == "__main__":
    main()
