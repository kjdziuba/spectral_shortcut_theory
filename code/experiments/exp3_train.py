#!/usr/bin/env python3
"""
Experiment 3B training (v1, 2026-09-10): real centre spectra with constructed
context, two readiness regimes. Design: paper/REFOCUS_PLAN_2026-09-10.md §5
(readiness scan + two-regime decision). Data cache and readiness scan:
code/experiments/exp3_real_context.py.

Regimes (same real spectra, same labels, pair CancerEpi (+1) vs CAS (-1), K=12):
  ready    per-feature standardization (train statistics); a random encoder
           already exposes the centre's class (random probe ~0.955 vs oracle 0.969)
  unready  PCA whitening (train statistics, ridge on small eigenvalues); random
           probe ~0.54 vs oracle 0.944

Patch = centre (real pixel, recorded label) + eight real donor spectra sampled
uniformly from the same split side (other cores), independent of the centre
label, each carrying the constructed cue gamma*(code*s + tau*eta) along the
first principal coordinate c (standardized regime: c = v1, gamma = 3 sqrt(lambda_1);
whitened regime: c = e_1, gamma = 30). tau is calibrated per regime so that the
neighbourhood oracle (LDA on the eight c-projections) matches the centre-only
oracle (ridge LDA on the preprocessed centre), on training-side halves.

Model: linear encoder W (K x 942, no bias) applied to the nine spectra ->
ReLU head h = relu(W1 vec(z_patch) + b1) (width M) -> logit w2.h. Margin loss,
full-batch GD, one global rate; whole-head multiplier kappa (arm headlr).
Matched fit at L* = 0.30 with snapshots at 0.6..0.10 (as Exp 2). Paired
evaluation conditions on held-out patients (val cores; test cores reported
separately): iid, reversed, ctx_random, spec_only (cue removed, donors kept).
Spectral probe: ridge LDA on the encoder output of centre pixels, fitted on a
disjoint training-side probe set, evaluated on val (and test) centres.

Usage
  python code/experiments/exp3_train.py --calibrate                # both regimes
  python code/experiments/exp3_train.py --regimes unready --arms sp --widths 32 --seeds 0 --max-steps 100   # smoke
  python code/experiments/exp3_train.py                            # full grid
  python code/experiments/exp3_train.py --collect
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from experiments.exp3_real_context import (  # noqa: E402
    CACHE, OUT_DIR, PAIR, S_FEAT, K, RIDGE, CAL_SEED, load_pair, lda_fit, lda_eval,
)

SUMMARY = OUT_DIR.parent / "exp3_summary.csv"

# ---- pre-registered constants (REFOCUS_PLAN §5; mirrors Exp 2) ----
LR = 1e-3
MAX_STEPS = 40_000
LOSS_STAR = 0.30
THRESHOLDS = [0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.10]
STOP_LOSS = 0.10
DIVERGE_LOSS = 3.0
LOG_EVERY = 25
N_TRAIN_PATCH = 24_000
N_PROBE_FIT = 6_000
N_EVAL_MAX = 6_000
WIDTHS = [8, 32, 128, 512]
HEAD_MULTS = [1, 1 / 16, 1 / 256]
HEADLR_WIDTH = 32
SEEDS = [0, 1, 2]
G_AMP_STD = 3.0      # gamma = 3 sqrt(lambda_1) in the standardized regime
G_AMP_WHITE = 30.0   # gamma = 30 in whitened units (unit natural variance along c)
WHITEN_RIDGE = 1e-3  # relative ridge on eigenvalues for whitening
CONDITIONS = ("iid", "reversed", "ctx_random", "spec_only")

ARMS = {
    "sp":      dict(widths=WIDTHS,          train_cond="iid",        head_mults=[1],        frozen=False),
    "headlr":  dict(widths=[HEADLR_WIDTH],  train_cond="iid",        head_mults=HEAD_MULTS, frozen=False),
    "ctxfree": dict(widths=[8, 128],        train_cond="ctx_random", head_mults=[1],        frozen=False),
    "frozen":  dict(widths=[HEADLR_WIDTH],  train_cond="iid",        head_mults=[1],        frozen=True),
}
REGIMES = ("ready", "unready")


# ------------------------------------------------------------ preprocessing ---
class Preproc:
    """Fit on training pixels; transform any pixels. Holds the cue direction c
    (unit vector in the preprocessed space) and the natural std of donors along c."""

    def __init__(self, regime: str, Xtr: torch.Tensor):
        self.regime = regime
        self.mean = Xtr.mean(0); self.std = Xtr.std(0) + 1e-8
        Z = (Xtr - self.mean) / self.std
        self.mu = Z.mean(0)
        Zc = (Z - self.mu).double()
        sub = Zc[torch.randperm(Zc.shape[0], generator=torch.Generator().manual_seed(CAL_SEED))[:40000]]
        U, Sv, Vh = torch.linalg.svd(sub, full_matrices=False)
        lam = (Sv ** 2) / (sub.shape[0] - 1)
        self.Vh = Vh.float(); self.lam = lam.float()
        # Eight cue directions, one per neighbour offset (as in Exp 2, so that a random
        # head's responses to the eight copies do not add coherently): the top eight
        # principal coordinates. C: (8, S) unit rows; gamma: (8,) amplitudes.
        if regime == "ready":
            self.C = self.Vh[:8].clone()                                # v_1..v_8 (standardized coords)
            self.c_std = torch.sqrt(self.lam[:8])                       # natural donor std along each
            self.gamma = G_AMP_STD * self.c_std
        elif regime == "unready":
            self.scale = 1.0 / torch.sqrt(self.lam + WHITEN_RIDGE * self.lam.mean())
            self.C = torch.zeros(8, S_FEAT)
            for d in range(8):
                self.C[d, d] = 1.0                                      # e_1..e_8 (whitened coords)
            self.c_std = torch.ones(8)
            self.gamma = G_AMP_WHITE * torch.ones(8)
        else:
            raise ValueError(regime)
        self.c = self.C[0]  # kept for backward compatibility of saved files

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        Z = (X - self.mean) / self.std - self.mu
        if self.regime == "ready":
            return Z
        return (Z @ self.Vh.T) * self.scale


# ---------------------------------------------------------------- patches ---
def sample_donors(n: int, pool_core: np.ndarray, centre_core: np.ndarray, g: torch.Generator) -> torch.Tensor:
    """(n, 8) donor indices into the pool, uniform, resampled once where the donor's core
    equals the centre's core."""
    npool = len(pool_core)
    idx = torch.randint(0, npool, (n, 8), generator=g)
    same = torch.from_numpy(pool_core[idx.numpy()] == centre_core[:, None])
    if same.any():
        idx2 = torch.randint(0, npool, (n, 8), generator=g)
        idx = torch.where(same, idx2, idx)
    return idx


def make_patches(Fp: torch.Tensor, y: torch.Tensor, core: np.ndarray, centres: torch.Tensor,
                 pool: torch.Tensor, pre: Preproc, tau: float, seed: int, cond: str, device):
    """Fp: preprocessed pixels (n_pix, S) on device; centres/pool: index tensors (CPU).
    Returns X (n, 9, S) on device and y (n,). Draw order is condition-independent:
    the four conditions from one seed share centres, donors and eta (paired)."""
    g = torch.Generator().manual_seed(seed)
    n = len(centres)
    donors = sample_donors(n, core[pool.numpy()], core[centres.numpy()], g)      # (n, 8) into pool
    eta = torch.randn(n, 8, generator=g)
    y_alt = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    yc = y[centres]
    if cond == "iid":
        code = yc
    elif cond == "reversed":
        code = -yc
    elif cond == "ctx_random":
        code = y_alt
    elif cond == "spec_only":
        code = None
    else:
        raise ValueError(cond)
    donor_idx = pool[donors]                                                    # (n, 8) into Fp
    Xn = Fp[donor_idx.to(device)]                                               # (n, 8, S)
    if code is not None:
        cue = pre.gamma[None, :] * (code[:, None] + tau * eta)                  # (n, 8): neighbour d on direction d
        Xn = Xn + cue.to(device)[:, :, None] * pre.C.to(device)[None, :, :]
    Xc = Fp[centres.to(device)][:, None, :]
    return torch.cat([Xc, Xn], dim=1).contiguous(), yc.to(device)


# ------------------------------------------------------------------ model ---
class Encoder(nn.Module):
    def __init__(self, S: int, K: int):
        super().__init__()
        self.proj = nn.Linear(S, K, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(S))

    def forward(self, X):  # (n, 9, S) -> (n, 9, K)
        return self.proj(X)


class PatchHead(nn.Module):
    """h = relu(W1 vec(z) + b1), logit = w2.h; W1: (M, 9K) default nn.Linear init."""

    def __init__(self, K: int, M: int):
        super().__init__()
        self.fc1 = nn.Linear(9 * K, M)
        self.fc2 = nn.Linear(M, 1, bias=False)

    def forward(self, Z):  # (n, 9, K) -> (n,)
        return self.fc2(F.relu(self.fc1(Z.reshape(Z.shape[0], -1)))).squeeze(-1)


def margin_loss(logits, y):
    return F.softplus(-y * logits).mean()


# ------------------------------------------------------------ calibration ---
def calibrate(device) -> dict:
    Xn, yn, core, side = load_pair(PAIR)
    X = torch.from_numpy(Xn); y = torch.from_numpy(yn)
    tr = side == "train"
    rec_all = {}
    for regime in REGIMES:
        pre = Preproc(regime, X[tr])
        Fp = pre.transform(X).to(device)
        ydev = y.to(device)
        tr_idx = np.where(tr)[0]
        tr_cores = np.unique(core[tr]); rng = np.random.default_rng(CAL_SEED); rng.shuffle(tr_cores)
        half_a = set(tr_cores[: len(tr_cores) // 2])
        a_mask = np.array([c in half_a for c in core[tr]])
        ia = torch.from_numpy(tr_idx[a_mask]); ib = torch.from_numpy(tr_idx[~a_mask])
        Fa, ya = Fp[ia.to(device)], ydev[ia.to(device)]; Fb, yb = Fp[ib.to(device)], ydev[ib.to(device)]
        w, thr = lda_fit(Fa, ya, RIDGE)
        acc_centre = lda_eval(w, thr, Fb, yb)
        val = torch.from_numpy(np.where(side == "val")[0])
        acc_centre_val = lda_eval(w, thr, Fp[val.to(device)], ydev[val.to(device)])

        def ctx_oracle(tau):
            Ca, _ = make_patches(Fp, y, core, ia, ia, pre, tau, 11, "iid", device)
            Cb, _ = make_patches(Fp, y, core, ib, ib, pre, tau, 12, "iid", device)
            Cd = pre.C.to(device)
            pa = torch.einsum("nds,ds->nd", Ca[:, 1:, :], Cd); pb = torch.einsum("nds,ds->nd", Cb[:, 1:, :], Cd)   # (n, 8)
            w2, t2 = lda_fit(pa, ya, 1e-6)
            return lda_eval(w2, t2, pb, yb)
        acc0 = ctx_oracle(0.0)
        lo, hi = 0.0, 8.0
        if acc0 <= acc_centre:
            tau = 0.0
        else:
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                if ctx_oracle(mid) > acc_centre:
                    lo = mid
                else:
                    hi = mid
            tau = 0.5 * (lo + hi)
        acc_ctx = ctx_oracle(tau)
        readiness = {}
        for s in (0, 1, 2):
            torch.manual_seed(s)
            enc = Encoder(S_FEAT, K).to(device)
            with torch.no_grad():
                Za, Zb = enc.proj(Fa), enc.proj(Fb)
            wc, tc = lda_fit(Za, ya, 1e-4); acc_c = lda_eval(wc, tc, Zb, yb)
            Pa, _ = make_patches(Fp, y, core, ia, ia, pre, tau, 21, "iid", device)
            Pb, _ = make_patches(Fp, y, core, ib, ib, pre, tau, 22, "iid", device)
            with torch.no_grad():
                Qa = enc(Pa).reshape(len(ia), -1); Qb = enc(Pb).reshape(len(ib), -1)
            wp, tp = lda_fit(Qa, ya, 1e-4); acc_p = lda_eval(wp, tp, Qb, yb)
            readiness[s] = dict(centre=acc_c, patch=acc_p)
        rec = dict(regime=regime, gamma=[float(g) for g in pre.gamma], tau=tau, c_std=[float(s) for s in pre.c_std],
                   lambda_1=float(pre.lam[0]), var_frac_v1=float(pre.lam[0] / pre.lam.sum()),
                   centre_oracle=acc_centre, centre_oracle_val=acc_centre_val,
                   context_oracle=acc_ctx, context_oracle_tau0=acc0, readiness_by_seed=readiness)
        print(f"[calib3] {regime}: gamma {[round(float(g), 2) for g in pre.gamma]} tau {tau:.4f}; centre oracle {acc_centre:.4f} (val {acc_centre_val:.4f}); "
              f"context oracle {acc_ctx:.4f} (tau=0: {acc0:.4f}); readiness {readiness}", flush=True)
        rec_all[regime] = rec
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration3.json").write_text(json.dumps(rec_all, indent=2))
    print(f"[calib3] wrote {OUT_DIR / 'calibration3.json'}")
    return rec_all


# ----------------------------------------------------------------- one run ---
def run_one(regime: str, arm: str, width: int, head_mult: float, seed: int, tau: float, lr: float,
            max_steps: int, device) -> pd.DataFrame:
    cfg = ARMS[arm]
    Xn, yn, core, side = load_pair(PAIR)
    X = torch.from_numpy(Xn); y = torch.from_numpy(yn)
    tr = side == "train"
    pre = Preproc(regime, X[tr])
    Fp = pre.transform(X).to(device)
    g = torch.Generator().manual_seed(7000 + seed)
    tr_idx = torch.from_numpy(np.where(tr)[0])
    perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
    # balanced centres for training; disjoint probe-fit pixels; donors = all train pixels
    pos = perm[y[perm] > 0]; neg = perm[y[perm] < 0]
    nh = min(N_TRAIN_PATCH // 2, len(pos) - N_PROBE_FIT // 2, len(neg) - N_PROBE_FIT // 2)
    centres = torch.cat([pos[:nh], neg[:nh]])
    probe_fit = torch.cat([pos[nh:nh + N_PROBE_FIT // 2], neg[nh:nh + N_PROBE_FIT // 2]])
    Xtr, ytr = make_patches(Fp, y, core, centres, tr_idx, pre, tau, 8000 + seed, cfg["train_cond"], device)
    evals = {}
    for sd in ("val", "test"):
        idx = torch.from_numpy(np.where(side == sd)[0])
        gs = torch.Generator().manual_seed(9000 + seed)
        idx = idx[torch.randperm(len(idx), generator=gs)[:N_EVAL_MAX]]
        for cond in CONDITIONS:
            evals[(sd, cond)] = make_patches(Fp, y, core, idx, torch.from_numpy(np.where(side == sd)[0]),
                                             pre, tau, 9500 + seed, cond, device)
        evals[(sd, "centres")] = idx
    yd = y.to(device)

    torch.manual_seed(seed); np.random.seed(seed)
    enc = Encoder(S_FEAT, K).to(device); head = PatchHead(K, width).to(device)
    if cfg["frozen"]:
        enc.proj.weight.requires_grad_(False)
    W0 = enc.proj.weight.detach().clone()
    groups = []
    if not cfg["frozen"]:
        groups.append({"params": [enc.proj.weight], "lr": lr})
    groups.append({"params": list(head.parameters()), "lr": lr * head_mult})
    opt = torch.optim.SGD(groups, momentum=0.0)

    h = f"_h{head_mult:g}" if head_mult != 1 else ""
    tag = f"{regime}_{arm}_M{width}{h}_s{seed}"
    pending = sorted(THRESHOLDS, reverse=True)
    traj_rows, snap_rows, W_snaps = [], [], {}
    t0 = time.time(); status = "running"; gth = gph = float("nan")

    @torch.no_grad()
    def probe(sd: str) -> float:
        zf = enc.proj(Fp[probe_fit.to(device)]); yf = yd[probe_fit.to(device)]
        idx = evals[(sd, "centres")].to(device)
        w, thr = lda_fit(zf, yf, 1e-4)
        return lda_eval(w, thr, enc.proj(Fp[idx]), yd[idx])

    @torch.no_grad()
    def acc(Xe, ye) -> float:
        return float((torch.sign(head(enc(Xe))) == ye).float().mean())

    def snapshot(label, step, loss_v, acc_v):
        Wt = enc.proj.weight.detach()
        row = dict(regime=regime, arm=arm, width=width, head_mult=head_mult, seed=seed, threshold=label,
                   step=step, loss=loss_v, acc_train=acc_v, gnorm_theta=gth, gnorm_phi=gph,
                   disp=float((Wt - W0).norm() / W0.norm()), wall_s=time.time() - t0,
                   probe_val=probe("val"), probe_test=probe("test"))
        for (sd, cond), val in evals.items():
            if cond == "centres":
                continue
            row[f"acc_{cond}_{sd}"] = acc(*val)
        snap_rows.append(row)
        W_snaps[f"W_{label}"] = Wt.cpu().numpy().copy()
        print(f"  [{tag}] {label!s:>8} step {step:6d} loss {loss_v:.4f} acc {acc_v:.3f} probe_val {row['probe_val']:.3f} "
              f"val iid {row['acc_iid_val']:.3f} rev {row['acc_reversed_val']:.3f} ctxrnd {row['acc_ctx_random_val']:.3f} "
              f"spec {row['acc_spec_only_val']:.3f}", flush=True)

    step = 0
    while True:
        opt.zero_grad(set_to_none=True)
        logits = head(enc(Xtr)); loss = margin_loss(logits, ytr)
        loss_v = float(loss); acc_v = float((torch.sign(logits) == ytr).float().mean())
        if not math.isfinite(loss_v) or loss_v > DIVERGE_LOSS:
            status = "diverged"; snapshot("diverged", step, loss_v, acc_v); break
        loss.backward()
        gth = float(enc.proj.weight.grad.norm()) if enc.proj.weight.grad is not None else 0.0
        gph = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in head.parameters() if p.grad is not None))
        if step == 0:
            snapshot("init", 0, loss_v, acc_v)
        while pending and loss_v < pending[0]:
            snapshot(pending.pop(0), step, loss_v, acc_v)
        if loss_v < STOP_LOSS:
            status = "stopped"; snapshot("final", step, loss_v, acc_v); break
        if step >= max_steps:
            status = "max_steps"; snapshot("final", step, loss_v, acc_v); break
        if step % LOG_EVERY == 0:
            traj_rows.append(dict(step=step, loss=loss_v, acc_train=acc_v, gnorm_theta=gth, gnorm_phi=gph,
                                  disp=float((enc.proj.weight.detach() - W0).norm() / W0.norm())))
        opt.step(); step += 1

    snaps = pd.DataFrame(snap_rows); snaps["status"] = status; snaps["lr"] = lr; snaps["tau"] = tau
    snaps["reached_star"] = bool((snaps["threshold"] == LOSS_STAR).any())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(traj_rows).to_csv(OUT_DIR / f"traj_{tag}.csv", index=False)
    snaps.to_csv(OUT_DIR / f"snap_{tag}.csv", index=False)
    np.savez_compressed(OUT_DIR / f"enc_{tag}.npz", W0=W0.cpu().numpy(), C=pre.C.numpy(), **W_snaps)
    print(f"[{tag}] {status} at step {step}, {time.time() - t0:.1f}s", flush=True)
    return snaps


def collect() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob("snap_*.csv"))
    if not files:
        raise SystemExit("no snapshot files")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["regime", "arm", "width", "head_mult", "seed"]
    init = df[df["threshold"].astype(str) == "init"].set_index(key)
    for col in ("probe_val", "probe_test"):
        df[f"{col}_gain"] = df[col].values - init[col].reindex(pd.MultiIndex.from_frame(df[key])).values
    df.to_csv(SUMMARY, index=False)
    print(f"[collect] {len(files)} runs -> {SUMMARY}")
    d = df[df["threshold"].astype(str) == str(LOSS_STAR)]
    cols = ["regime", "arm", "width", "head_mult", "seed", "step", "probe_val_gain", "acc_iid_val",
            "acc_reversed_val", "acc_ctx_random_val", "acc_spec_only_val"]
    print(d[cols].sort_values(["regime", "arm", "width", "head_mult", "seed"]).round(3).to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--widths", nargs="*", type=int, default=None)
    ap.add_argument("--head-mults", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.calibrate:
        calibrate(device); return
    if args.collect:
        collect(); return
    cal_path = OUT_DIR / "calibration3.json"
    if not cal_path.exists():
        raise SystemExit("run --calibrate first")
    cal = json.loads(cal_path.read_text())
    for regime in args.regimes:
        tau = float(cal[regime]["tau"])
        for arm in args.arms:
            cfg = ARMS[arm]
            for width in (args.widths or cfg["widths"]):
                for hm in (args.head_mults or cfg["head_mults"]):
                    for seed in args.seeds:
                        h = f"_h{hm:g}" if hm != 1 else ""
                        tag = f"{regime}_{arm}_M{width}{h}_s{seed}"
                        if args.skip_existing and (OUT_DIR / f"snap_{tag}.csv").exists():
                            print(f"[exp3] skip {tag}"); continue
                        print(f"[exp3] run {tag} lr={args.lr:g} tau={tau:.4f}", flush=True)
                        run_one(regime, arm, width, hm, seed, tau, args.lr, args.max_steps, device)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    collect()


if __name__ == "__main__":
    main()
