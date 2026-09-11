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
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path("/mnt/hdd2/u37314kd/data_breast_v2_pca23")
FOLD = int(os.environ.get("EXP_FOLD", "0"))        # §12: five-fold extension; fold 0 = the registered paths
SPLITS = DATA_DIR / f"splits_fold{FOLD}.json"
N_PER_CORE_CLASS = {"train": 600, "val": 300, "test": 300}
K = 12

# ---- dataset switch (REFOCUS_PLAN §5 breast = default; §9 paviau = public replication) ----
DATASET = os.environ.get("EXP3_DATASET", "breast")
_CFG = {
    "breast": dict(out="exp3", pair={3: +1.0, 4: -1.0},          # CancerEpi -> +1, CAS -> -1   [author to confirm]
                   classes={1: "NormalEpi", 2: "NormalStroma", 3: "CancerEpi", 4: "CAS"}, s_feat=942,
                   scan_pairs={"CancerEpi-vs-CAS": {3: +1.0, 4: -1.0}, "NormalStroma-vs-CAS": {2: +1.0, 4: -1.0},
                               "NormalEpi-vs-CancerEpi": {1: +1.0, 3: -1.0}, "NormalStroma-vs-CancerEpi": {2: +1.0, 3: -1.0}},
                   sizes=dict(n_train_patch=24_000, n_probe_fit=6_000, n_eval_max=6_000)),
    "paviau": dict(out="exp3_paviau", pair={2: +1.0, 4: -1.0},   # Meadows -> +1, Trees -> -1 (§9, fixed before any run)
                   classes={1: "Asphalt", 2: "Meadows", 3: "Gravel", 4: "Trees", 5: "PaintedMetal", 6: "BareSoil",
                            7: "Bitumen", 8: "Bricks", 9: "Shadows"}, s_feat=103,
                   scan_pairs={"Meadows-vs-Trees": {2: +1.0, 4: -1.0}, "Gravel-vs-Bricks": {3: +1.0, 8: -1.0},
                               "Asphalt-vs-Bricks": {1: +1.0, 8: -1.0}, "Meadows-vs-BareSoil": {2: +1.0, 6: -1.0}},
                   sizes=dict(n_train_patch=2_400, n_probe_fit=400, n_eval_max=6_000)),
}
CFG = _CFG[DATASET]
OUT_DIR = REPO / "results" / (CFG["out"] if FOLD == 0 else f"{CFG['out']}_fold{FOLD}")
CACHE = OUT_DIR / f"cache_fold{FOLD}.npz"
CALIB = OUT_DIR / "calibration.json"
PAIR = CFG["pair"]
CLASSES = CFG["classes"]
CACHE_CLASSES = tuple(CLASSES)     # the cache holds all classes; PAIR selects the task
S_FEAT = CFG["s_feat"]
SCAN_PAIRS = CFG["scan_pairs"]
SIZES = CFG["sizes"]
G_AMP = 3.0                         # gamma = G_AMP * sqrt(lambda_1): "large" relative to natural v1 variation
CAL_SEED = 0
RIDGE = 1e-2                        # relative ridge for the 942-dim LDA


def load_split_ids() -> dict:
    return json.load(open(SPLITS))


def build_cache() -> None:
    if DATASET != "breast":
        raise SystemExit("build hyperspectral caches with code/experiments/hsi_data.py")
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
            n_core = 0
            for lab in CACHE_CLASSES:
                idx = np.argwhere((y == lab) & mask)
                if len(idx) == 0:
                    continue
                take = min(len(idx), N_PER_CORE_CLASS[side])
                sel = idx[rng.choice(len(idx), take, replace=False)]
                if feats is None:
                    feats = np.concatenate([z["X_raw"], z["X_d1"], z["X_d2"]], axis=-1)  # (H, W, 942)
                Xs.append(feats[sel[:, 0], sel[:, 1]].astype(np.float32))
                ys.append(np.full(take, lab, dtype=np.int16))          # ORIGINAL label 1..4
                cores.append(np.full(take, cid))
                sides.append(np.full(take, side))
                n_core += take
            print(f"[cache] {side} {cid}: {n_core} px ({time.time() - t0:.0f}s)", flush=True)
    X = np.concatenate(Xs); lab = np.concatenate(ys); core = np.concatenate(cores); side = np.concatenate(sides)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, label=lab, core=core, side=side)
    for s in ("train", "val", "test"):
        m = side == s
        counts = {CLASSES[c]: int((lab[m] == c).sum()) for c in CACHE_CLASSES}
        print(f"[cache] {s}: {m.sum()} px, {counts}, cores {len(np.unique(core[m]))}")
    print(f"[cache] wrote {CACHE} ({X.nbytes / 1e9:.2f} GB raw)")


def load_pair(pair: dict):
    """Cache -> (X float32 (n, 942), y in {-1,+1}, core, side) restricted to `pair`."""
    z = np.load(CACHE, allow_pickle=True)
    lab = z["label"]
    keep = np.isin(lab, list(pair.keys()))
    y = np.array([pair[int(l)] for l in lab[keep]], dtype=np.float32)
    return z["X"][keep], y, z["core"][keep], z["side"][keep]


def readiness_scan(device: torch.device) -> None:
    """For several class pairs, bottleneck sizes K and preprocessing choices:
    centre-only oracle (ridge LDA, 942-d), random-encoder centre probe (K-d), and
    the alignment of the class contrast with the top principal directions.
    Fit on training half A (cores), evaluate on half B. Writes results/exp3/readiness_scan.csv."""
    import pandas as pd
    pairs = SCAN_PAIRS
    rows = []
    for pname, pair in pairs.items():
        X, y, core, side = load_pair(pair)
        tr = side == "train"
        Xt = torch.from_numpy(X[tr]); yt = torch.from_numpy(y[tr])
        mean = Xt.mean(0); std = Xt.std(0) + 1e-8
        Xs = ((Xt - mean) / std).to(device); ys_ = yt.to(device)
        tr_cores = np.unique(core[tr]); rng = np.random.default_rng(CAL_SEED); rng.shuffle(tr_cores)
        half_a = set(tr_cores[: len(tr_cores) // 2])
        in_a = torch.from_numpy(np.array([c in half_a for c in core[tr]])).to(device)
        # PCA on half A (standardized) for whitening and contrast alignment
        A = Xs[in_a].double(); A = A - A.mean(0)
        U, Sv, Vh = torch.linalg.svd(A, full_matrices=False)
        lam = (Sv ** 2) / (A.shape[0] - 1)
        contrast = (Xs[in_a][ys_[in_a] > 0].mean(0) - Xs[in_a][ys_[in_a] < 0].mean(0)).double()
        cos1 = float(abs(contrast @ Vh[0]) / contrast.norm())
        top10 = float(((Vh[:10] @ contrast) ** 2).sum().sqrt() / contrast.norm())
        for prep in ("standardized", "whitened"):
            if prep == "standardized":
                F = Xs.double()
            else:
                # PCA whitening on half-A statistics (ridge on small eigenvalues)
                Fc = Xs.double() - A.mean(0) * 0 - Xs[in_a].double().mean(0)
                scale = 1.0 / torch.sqrt(lam + 1e-3 * lam.mean())
                F = (Fc @ Vh.T) * scale
            Fa, ya = F[in_a], ys_[in_a]; Fb, yb = F[~in_a], ys_[~in_a]
            w, thr = lda_fit(Fa, ya, RIDGE)
            acc_oracle = lda_eval(w, thr, Fb, yb)
            for Kb in (2, 4, 12, 32):
                accs = []
                for s in (0, 1, 2):
                    torch.manual_seed(s)
                    Wenc = torch.randn(Kb, F.shape[1], device=device, dtype=torch.float64) / math.sqrt(F.shape[1])
                    wc, tc = lda_fit(Fa @ Wenc.T, ya, 1e-4)
                    accs.append(lda_eval(wc, tc, Fb @ Wenc.T, yb))
                rows.append(dict(pair=pname, n_train=int(tr.sum()), prep=prep, K=Kb, oracle=acc_oracle,
                                 random_probe_mean=float(np.mean(accs)), random_probe_min=float(np.min(accs)),
                                 random_probe_max=float(np.max(accs)), cos_contrast_v1=cos1,
                                 contrast_frac_top10=top10, var_frac_v1=float(lam[0] / lam.sum())))
                print(f"[scan] {pname:28s} {prep:12s} K={Kb:3d}: oracle {acc_oracle:.3f}  random probe "
                      f"{np.mean(accs):.3f} [{np.min(accs):.3f}, {np.max(accs):.3f}]  |cos(contrast,v1)| {cos1:.2f}  "
                      f"top10 frac {top10:.2f}", flush=True)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "readiness_scan.csv", index=False)
    print(f"[scan] wrote {OUT_DIR / 'readiness_scan.csv'}")


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
    Xn, yn, core, side = load_pair(PAIR)
    X = torch.from_numpy(Xn); y = torch.from_numpy(yn)
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
    ap.add_argument("--readiness-scan", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.build_cache:
        build_cache()
    if args.readiness_scan:
        readiness_scan(device)
    if args.calibrate:
        calibrate(device)


if __name__ == "__main__":
    main()
