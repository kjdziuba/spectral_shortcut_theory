#!/usr/bin/env python3
"""
Experiment 2 (v1, refocus 2026-09-10): contextual-pathway speed versus
spectral learning at matched training fit, beyond replicated coordinates.

Pre-registration: paper/REFOCUS_PLAN_2026-09-10.md, section 4 (data, model,
arms, thresholds, predictions P1-P5). Constants below are the ones fixed
there; tau comes from results/exp2/calibration.json, produced by
`--calibrate` BEFORE any training run.

Model: linear per-pixel encoder W (K x S, no bias, N(0,1/S) init) -> ReLU CNN
head Conv3x3(K->M) - ReLU - Conv3x3(M->1), circular padding, single logit,
loss = mean_p log(1 + exp(-y_p F_p)). Full-batch gradient descent, one global
learning rate for every parameter (the theorem's unit rates), no momentum,
no clipping, no weight decay.

Arms (see ARMS): sp (standard parameterization, width sweep), mup (readout
M^{-1/2} gamma with O(1) gamma, same function at init: the theorem's
normalization control), ctxfree (training context uninformative: removes
the proposed cause), lrmult (M = 32, readout learning-rate multiplier:
effective speed without parameter count), frozen (encoder fixed at init).

Matched fit: the first step with training loss < L* = 0.30 (primary), with
snapshots at 0.6, 0.5, 0.4, 0.2, 0.15, 0.10. At every snapshot: encoder
alignment with u, context alignment, displacement, LDA spectral probe on
the encoder output, block gradient norms, accuracies on the five test
conditions (iid, reversed, ctx_random, spec_only, ctx_only).

Usage
  python code/experiments/exp2_intervention.py --calibrate
  python code/experiments/exp2_intervention.py --stability-check
  python code/experiments/exp2_intervention.py --arms sp --widths 8 2048 --seeds 0   # pilot
  python code/experiments/exp2_intervention.py                                        # full grid
  python code/experiments/exp2_intervention.py --collect --plot
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

from synthetic.data_v2 import (  # noqa: E402
    CONDITIONS, ProblemSpec, lda_accuracy, make_directions, make_problem_v2,
    unfold_window,
)

REPO = CODE_DIR.parent
OUT_DIR = REPO / "results" / "exp2"
SUMMARY = REPO / "results" / "exp2_summary.csv"
CALIB = OUT_DIR / "calibration.json"

# ---------------------------------------------------------------------------
# Pre-registered constants (REFOCUS_PLAN section 4)
# ---------------------------------------------------------------------------
S, K, H, W = 256, 12, 16, 16
N_TRAIN, N_TEST = 256, 64
ALPHA, BETA, SIGMA = 1.645, 5.0, 1.0
TARGET_ORACLE = 0.95
LR_DEFAULT = 1e-3          # stability check 2026-09-10: largest monotone rate at M=2048
MAX_STEPS = 40_000
LOSS_STAR = 0.30
THRESHOLDS = [0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.10]
STOP_LOSS = 0.10
DIVERGE_LOSS = 3.0
LOG_EVERY = 25
# Amended 2026-09-10 20:00 after the one-seed pilot (REFOCUS_PLAN 4.4a):
# widths 2 and 4 added (O(1)-speed regime); fractional readout multipliers
# added; 2048 dropped from the control arms for cost.
WIDTHS_FULL = [2, 4, 8, 32, 128, 512, 2048]
WIDTHS_CTRL = [2, 8, 128, 512]
LR_MULTS = [1 / 16, 1 / 4, 1, 4, 16]
LRMULT_WIDTH = 32
SEEDS = [0, 1, 2]

ARMS = {
    "sp":      dict(param="sp",  widths=WIDTHS_FULL,    train_cond="iid",        mults=[1],      frozen=False),
    "mup":     dict(param="mup", widths=WIDTHS_FULL,    train_cond="iid",        mults=[1],      frozen=False),
    "ctxfree": dict(param="sp",  widths=WIDTHS_CTRL,    train_cond="ctx_random", mults=[1],      frozen=False),
    "lrmult":  dict(param="sp",  widths=[LRMULT_WIDTH], train_cond="iid",        mults=LR_MULTS, frozen=False),
    "frozen":  dict(param="sp",  widths=WIDTHS_CTRL,    train_cond="iid",        mults=[1],      frozen=True),
}

# Seed layout: directions 1000+s; training data 2000+s; test condition i
# 3000+10s+i; probe fit 4000+s; probe eval 5000+s; model init = s.


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, S: int, K: int) -> None:
        super().__init__()
        self.proj = nn.Linear(S, K, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(S))

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # (B,H,W,S) -> (B,H,W,K)
        return self.proj(X)


class CNNHead(nn.Module):
    """Conv3x3(K->M) - ReLU - Conv3x3(M->1), circular padding, single logit.

    param='sp'  : PyTorch default init for both convs (readout entries O(1/sqrt(M))).
    param='mup' : readout = M^{-1/2} * gamma with gamma = sqrt(M) * (default init),
                  i.e. O(1) entries and the SAME function at initialization; trained
                  at the same learning rate, its effective speed is 1/M of 'sp'
                  (the theorem's alpha_M = M^{-1/2} control).
    """

    def __init__(self, K: int, M: int, param: str, use_conv: bool = False) -> None:
        super().__init__()
        if param not in ("sp", "mup"):
            raise ValueError(param)
        self.M, self.param = M, param
        self.use_conv = use_conv  # True: nn.Conv2d path (reference); False: matmul path (fast, identical)
        self.conv1 = nn.Conv2d(K, M, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(M, 1, 3, padding=1, padding_mode="circular", bias=False)
        if param == "mup":
            with torch.no_grad():
                self.conv2.weight.mul_(math.sqrt(M))
            self.scale = M ** -0.5
        else:
            self.scale = 1.0

    @staticmethod
    def _windows(Z: torch.Tensor) -> torch.Tensor:
        """(B,H,W,K) -> (B,H,W,K,9): [..., k, 3*iy+ix] = Z[p + (iy-1, ix-1), k] on the torus,
        the cross-correlation convention of nn.Conv2d with circular padding."""
        cols = [torch.roll(Z, shifts=(-(iy - 1), -(ix - 1)), dims=(1, 2))
                for iy in range(3) for ix in range(3)]
        return torch.stack(cols, dim=-1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:  # (B,H,W,K) -> (B,H,W)
        if self.use_conv:
            Zc = Z.permute(0, 3, 1, 2).contiguous()
            h = F.relu(self.conv1(Zc))
            return (self.conv2(h) * self.scale).squeeze(1)
        B, H_, W_, Kc = Z.shape
        zw = self._windows(Z).reshape(B * H_ * W_, Kc * 9)          # (k, iy, ix) order
        W1 = self.conv1.weight.reshape(self.M, Kc * 9)               # (M, K, 3, 3) -> same order
        h = F.relu(zw @ W1.T + self.conv1.bias)                       # (BHW, M)
        W2 = self.conv2.weight.reshape(self.M, 9)                     # (1, M, 3, 3) -> (M, (iy, ix))
        G = (h @ W2).reshape(B, H_, W_, 9)
        out = torch.zeros(B, H_, W_, device=Z.device, dtype=Z.dtype)
        i = 0
        for iy in range(3):
            for ix in range(3):
                out = out + torch.roll(G[..., i], shifts=(-(iy - 1), -(ix - 1)), dims=(1, 2))
                i += 1
        return out * self.scale


def selftest_head(device) -> None:
    """The matmul path must reproduce nn.Conv2d with circular padding (outputs and gradients)."""
    # cuDNN convolutions use TF32 by default (relative precision ~1e-3); compare in float64.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    for param in ("sp", "mup"):
        for M in (2, 37, 512):
            ref = CNNHead(K, M, param, use_conv=True).to(device).double()
            fast = CNNHead(K, M, param, use_conv=False).to(device).double()
            fast.load_state_dict(ref.state_dict())
            Z = torch.randn(3, H, W, K, device=device, dtype=torch.float64)
            y = torch.sign(torch.randn(3, H, W, device=device, dtype=torch.float64))
            o_ref = ref(Z); o_fast = fast(Z)
            l_ref = margin_loss(o_ref, y); l_fast = margin_loss(o_fast, y)
            g_ref = torch.autograd.grad(l_ref, list(ref.parameters()))
            g_fast = torch.autograd.grad(l_fast, list(fast.parameters()))
            d_out = float((o_ref - o_fast).abs().max() / (o_ref.abs().max() + 1e-12))
            d_grad = max(float((a - b).abs().max() / (a.abs().max() + 1e-12)) for a, b in zip(g_ref, g_fast))
            print(f"[selftest] param={param} M={M}: max rel |out diff| {d_out:.2e}, max rel |grad diff| {d_grad:.2e}")
            assert d_out < 1e-9 and d_grad < 1e-9, "matmul head disagrees with Conv2d"
    print("[selftest] OK")


def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.softplus(-y * logits).mean()


# ---------------------------------------------------------------------------
# Data bundle per seed
# ---------------------------------------------------------------------------
def build_data(seed: int, spec: ProblemSpec, train_cond: str, device: torch.device):
    u, V = make_directions(spec.S, spec.n_ctx, seed=1000 + seed)
    Xtr, ytr = make_problem_v2(N_TRAIN, spec, u, V, seed=2000 + seed, condition=train_cond)
    tests = {}
    for i, c in enumerate(CONDITIONS):
        Xte, yte = make_problem_v2(N_TEST, spec, u, V, seed=3000 + 10 * seed + i, condition=c)
        tests[c] = (Xte.to(device), yte.to(device))
    pf = make_problem_v2(N_TEST, spec, u, V, seed=4000 + seed, condition="spec_only")
    pe = make_problem_v2(N_TEST, spec, u, V, seed=5000 + seed, condition="spec_only")
    return dict(
        u=u.to(device), V=V.to(device),
        Xtr=Xtr.to(device), ytr=ytr.to(device), tests=tests,
        probe_fit=(pf[0].to(device), pf[1].to(device)),
        probe_eval=(pe[0].to(device), pe[1].to(device)),
    )


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
@torch.no_grad()
def accuracy(enc: nn.Module, head: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    logits = head(enc(X))
    return float((torch.sign(logits) == y).float().mean())


@torch.no_grad()
def spectral_probe(enc: nn.Module, data: dict) -> float:
    Xf, yf = data["probe_fit"]; Xe, ye = data["probe_eval"]
    zf = enc(Xf).reshape(-1, K); ze = enc(Xe).reshape(-1, K)
    return lda_accuracy(zf, yf.reshape(-1), ze, ye.reshape(-1))


@torch.no_grad()
def encoder_stats(enc: nn.Module, W0: torch.Tensor, u: torch.Tensor, V: torch.Tensor) -> dict:
    Wt = enc.proj.weight
    wn2 = float((Wt ** 2).sum())
    wu = Wt @ u
    wV = Wt @ V.T  # (K, n_ctx)
    return dict(
        align_u=float((wu ** 2).sum()) / wn2,
        align_V=float((wV ** 2).sum()) / wn2,
        gain_u=float(wu.norm()),
        gain_V=float(wV.norm()),
        w_norm=math.sqrt(wn2),
        disp=float((Wt - W0).norm() / W0.norm()),
    )


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def mult_tag(mult: float) -> str:
    return f"{mult:g}"


def run_one(arm: str, width: int, mult: float, seed: int, tau: float, lr: float,
            max_steps: int, device: torch.device, quiet: bool = False) -> pd.DataFrame:
    cfg = ARMS[arm]
    spec = ProblemSpec(S=S, H=H, W=W, alpha=ALPHA, beta=BETA, tau=tau, sigma=SIGMA)
    data = build_data(seed, spec, cfg["train_cond"], device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder(S, K).to(device)
    head = CNNHead(K, width, cfg["param"]).to(device)
    if cfg["frozen"]:
        enc.proj.weight.requires_grad_(False)
    W0 = enc.proj.weight.detach().clone()

    groups = []
    if not cfg["frozen"]:
        groups.append({"params": [enc.proj.weight], "lr": lr})
    groups.append({"params": list(head.conv1.parameters()), "lr": lr})
    groups.append({"params": [head.conv2.weight], "lr": lr * mult})
    opt = torch.optim.SGD(groups, momentum=0.0)

    Xtr, ytr = data["Xtr"], data["ytr"]
    tag = f"{arm}_M{width}_x{mult_tag(mult)}_s{seed}"
    pending = sorted(THRESHOLDS, reverse=True)
    traj_rows, snap_rows = [], []
    t0 = time.time()
    status = "running"
    gth = gph = float("nan")

    def snapshot(label, step, loss_v, acc_v):
        st = encoder_stats(enc, W0, data["u"], data["V"])
        row = dict(arm=arm, param=cfg["param"], train_cond=cfg["train_cond"], width=width,
                   mult=mult, seed=seed, threshold=label, step=step, loss=loss_v,
                   acc_train=acc_v, gnorm_theta=gth, gnorm_phi=gph,
                   probe_acc=spectral_probe(enc, data), wall_s=time.time() - t0, **st)
        for c, (Xc, yc) in data["tests"].items():
            row[f"acc_{c}"] = accuracy(enc, head, Xc, yc)
        snap_rows.append(row)
        if not quiet:
            print(f"  [{tag}] {label!s:>8} step {step:6d} loss {loss_v:.4f} acc {acc_v:.3f} "
                  f"a_u {st['align_u']:.4f} probe {row['probe_acc']:.3f} "
                  f"iid {row['acc_iid']:.3f} rev {row['acc_reversed']:.3f} "
                  f"ctxrnd {row['acc_ctx_random']:.3f} spec {row['acc_spec_only']:.3f}",
                  flush=True)

    step = 0
    while True:
        logits = head(enc(Xtr))
        loss = margin_loss(logits, ytr)
        loss_v = float(loss)
        acc_v = float((torch.sign(logits) == ytr).float().mean())

        if not math.isfinite(loss_v) or loss_v > DIVERGE_LOSS:
            status = "diverged"
            snapshot("diverged", step, loss_v, acc_v)
            break
        if step == 0:
            # gradient norms at init are filled in below (after backward)
            pass
        while pending and loss_v < pending[0]:
            snapshot(pending.pop(0), step, loss_v, acc_v)
        if loss_v < STOP_LOSS:
            status = "stopped"
            snapshot("final", step, loss_v, acc_v)
            break
        if step >= max_steps:
            status = "max_steps"
            snapshot("final", step, loss_v, acc_v)
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gth = float(enc.proj.weight.grad.norm()) if enc.proj.weight.grad is not None else 0.0
        gph = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in head.parameters() if p.grad is not None))
        if step == 0:
            snapshot("init", 0, loss_v, acc_v)
        if step % LOG_EVERY == 0:
            st = encoder_stats(enc, W0, data["u"], data["V"])
            traj_rows.append(dict(step=step, loss=loss_v, acc_train=acc_v, gnorm_theta=gth,
                                  gnorm_phi=gph, **st))
        opt.step()
        step += 1

    traj = pd.DataFrame(traj_rows)
    snaps = pd.DataFrame(snap_rows)
    snaps["status"] = status
    snaps["lr"] = lr
    snaps["tau"] = tau
    snaps["reached_star"] = bool((snaps["threshold"] == LOSS_STAR).any())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj.to_csv(OUT_DIR / f"traj_{tag}.csv", index=False)
    snaps.to_csv(OUT_DIR / f"snap_{tag}.csv", index=False)
    if not quiet:
        print(f"[{tag}] {status} at step {step}, {time.time() - t0:.1f}s", flush=True)
    return snaps


# ---------------------------------------------------------------------------
# Calibration (run before any training; writes results/exp2/calibration.json)
# ---------------------------------------------------------------------------
def oracle_context_acc(spec: ProblemSpec, u, V, device) -> float:
    Xf, yf = make_problem_v2(N_TEST, spec, u, V, seed=6001, condition="iid")
    Xe, ye = make_problem_v2(N_TEST, spec, u, V, seed=6002, condition="iid")
    Xf, yf, Xe, ye = Xf.to(device), yf.to(device), Xe.to(device), ye.to(device)
    Vd = V.to(device)
    Pf = torch.einsum("bhws,ds->bdhw", Xf, Vd); Pe = torch.einsum("bhws,ds->bdhw", Xe, Vd)
    Ff = unfold_window(Pf, r=1); Fe = unfold_window(Pe, r=1)
    return lda_accuracy(Ff, yf.reshape(-1), Fe, ye.reshape(-1))


def oracle_spectral_acc(spec: ProblemSpec, u, V, device) -> float:
    Xf, yf = make_problem_v2(N_TEST, spec, u, V, seed=6001, condition="iid")
    Xe, ye = make_problem_v2(N_TEST, spec, u, V, seed=6002, condition="iid")
    ud = u.to(device)
    Ff = (Xf.to(device) @ ud).reshape(-1, 1); Fe = (Xe.to(device) @ ud).reshape(-1, 1)
    return lda_accuracy(Ff, yf.reshape(-1).to(device), Fe, ye.reshape(-1).to(device))


def readiness(spec: ProblemSpec, seed: int, device) -> dict:
    """LDA accuracies through a RANDOM encoder W ~ N(0, 1/S): the initial
    readability of each cue (the analogue of a_0 versus v_0)."""
    u, V = make_directions(spec.S, spec.n_ctx, seed=1000 + seed)
    torch.manual_seed(seed)
    enc = Encoder(S, K).to(device)
    out = {}
    for name, cond, window in (("spec", "spec_only", 0), ("ctx", "ctx_only", 1), ("both", "iid", 1)):
        Xf, yf = make_problem_v2(N_TEST, spec, u, V, seed=6001, condition=cond)
        Xe, ye = make_problem_v2(N_TEST, spec, u, V, seed=6002, condition=cond)
        with torch.no_grad():
            zf = enc(Xf.to(device)).permute(0, 3, 1, 2); ze = enc(Xe.to(device)).permute(0, 3, 1, 2)
        if window:
            Ff = unfold_window(zf, r=1); Fe = unfold_window(ze, r=1)
        else:
            Ff = zf.permute(0, 2, 3, 1).reshape(-1, K); Fe = ze.permute(0, 2, 3, 1).reshape(-1, K)
        out[name] = lda_accuracy(Ff, yf.reshape(-1).to(device), Fe, ye.reshape(-1).to(device))
    return out


def calibrate(device) -> dict:
    spec0 = ProblemSpec(S=S, H=H, W=W, alpha=ALPHA, beta=BETA, tau=0.0, sigma=SIGMA)
    u, V = make_directions(S, spec0.n_ctx, seed=1000 + 0)
    lo, hi = 0.0, 6.0
    acc_lo = oracle_context_acc(spec0, u, V, device)
    print(f"[calib] oracle context accuracy at tau=0: {acc_lo:.4f}")
    if acc_lo < TARGET_ORACLE:
        raise SystemExit("context cue cannot reach the target even without context noise")
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        spec = ProblemSpec(S=S, H=H, W=W, alpha=ALPHA, beta=BETA, tau=mid, sigma=SIGMA)
        acc = oracle_context_acc(spec, u, V, device)
        print(f"[calib]   tau={mid:.4f} -> oracle context acc {acc:.4f}")
        if acc > TARGET_ORACLE:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    spec = ProblemSpec(S=S, H=H, W=W, alpha=ALPHA, beta=BETA, tau=tau, sigma=SIGMA)
    rec = dict(S=S, K=K, H=H, W=W, alpha=ALPHA, beta=BETA, sigma=SIGMA, tau=tau,
               target_oracle=TARGET_ORACLE, n_ctx=spec.n_ctx,
               oracle_ctx_by_seed={}, oracle_spec_by_seed={}, readiness_by_seed={})
    for s in SEEDS:
        us, Vs = make_directions(S, spec.n_ctx, seed=1000 + s)
        rec["oracle_ctx_by_seed"][s] = oracle_context_acc(spec, us, Vs, device)
        rec["oracle_spec_by_seed"][s] = oracle_spectral_acc(spec, us, Vs, device)
        rec["readiness_by_seed"][s] = readiness(spec, s, device)
        print(f"[calib] seed {s}: oracle spec {rec['oracle_spec_by_seed'][s]:.4f}  "
              f"oracle ctx {rec['oracle_ctx_by_seed'][s]:.4f}  random-encoder readiness "
              f"{rec['readiness_by_seed'][s]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(rec, indent=2))
    print(f"[calib] tau = {tau:.4f} written to {CALIB}")
    return rec


def stability_check(device, tau: float, lrs=(1e-4, 3e-4, 1e-3, 3e-3), width=2048, steps=300):
    """Full-batch GD at the largest width: report whether the loss decreases
    monotonically over the first `steps` steps for each candidate lr."""
    spec = ProblemSpec(S=S, H=H, W=W, alpha=ALPHA, beta=BETA, tau=tau, sigma=SIGMA)
    data = build_data(0, spec, "iid", device)
    for lr in lrs:
        torch.manual_seed(0)
        enc = Encoder(S, K).to(device); head = CNNHead(K, width, "sp").to(device)
        opt = torch.optim.SGD(list(enc.parameters()) + list(head.parameters()), lr=lr, momentum=0.0)
        losses = []
        for _ in range(steps):
            loss = margin_loss(head(enc(data["Xtr"])), data["ytr"])
            losses.append(float(loss))
            if not math.isfinite(losses[-1]) or losses[-1] > DIVERGE_LOSS:
                break
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        l = np.array(losses)
        n_up = int((np.diff(l) > 1e-6).sum())
        print(f"[stability] M={width} lr={lr:.0e}: loss {l[0]:.4f} -> {l[-1]:.4f} after {len(l)} steps; "
              f"{n_up} increases; max jump {np.max(np.diff(l)) if len(l) > 1 else float('nan'):+.4f}")


# ---------------------------------------------------------------------------
# Collect + plot
# ---------------------------------------------------------------------------
def collect() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob("snap_*.csv"))
    if not files:
        raise SystemExit("no snapshot files")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(SUMMARY, index=False)
    print(f"[collect] {len(files)} runs, {len(df)} snapshot rows -> {SUMMARY}")
    return df


def spearman(x, y) -> float:
    xr = pd.Series(x).rank().values; yr = pd.Series(y).rank().values
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def evaluate_predictions(df: pd.DataFrame) -> str:
    """Pre-registered P1-P5 at L* (REFOCUS_PLAN 4.4). Descriptive; prints a table."""
    d = df[df["threshold"].astype(str) == str(LOSS_STAR)].copy()
    lines = [f"# Pre-registered predictions at L* = {LOSS_STAR}", ""]
    for arm in ("sp", "mup", "ctxfree", "frozen"):
        a = d[d["arm"] == arm]
        if a.empty:
            continue
        lines.append(f"## {arm}")
        lines.append("| seed | widths | a_u(L*) | probe(L*) | acc_reversed(L*) | rho(a_u,M) | rho(rev,M) |")
        lines.append("|---|---|---|---|---|---|---|")
        for s, g in a.groupby("seed"):
            g = g.sort_values("width")
            lines.append(
                f"| {s} | {list(g['width'])} | {[round(v, 4) for v in g['align_u']]} | "
                f"{[round(v, 3) for v in g['probe_acc']]} | {[round(v, 3) for v in g['acc_reversed']]} | "
                f"{spearman(g['width'], g['align_u']):+.2f} | {spearman(g['width'], g['acc_reversed']):+.2f} |")
        lines.append("")
    a = d[d["arm"] == "lrmult"]
    if not a.empty:
        lines.append("## lrmult (M = 32)")
        lines.append("| seed | mults | a_u(L*) | probe(L*) | acc_reversed(L*) | rho(a_u,mult) | rho(rev,mult) |")
        lines.append("|---|---|---|---|---|---|---|")
        for s, g in a.groupby("seed"):
            g = g.sort_values("mult")
            lines.append(
                f"| {s} | {list(g['mult'])} | {[round(v, 4) for v in g['align_u']]} | "
                f"{[round(v, 3) for v in g['probe_acc']]} | {[round(v, 3) for v in g['acc_reversed']]} | "
                f"{spearman(g['mult'], g['align_u']):+.2f} | {spearman(g['mult'], g['acc_reversed']):+.2f} |")
        lines.append("")
    miss = df[(df["threshold"] == "final") & (~df["reached_star"])]
    if not miss.empty:
        lines.append("## Runs that never reached L*")
        for _, r in miss.iterrows():
            lines.append(f"- {r['arm']} M={r['width']} x{r['mult']} s{r['seed']}: status {r['status']}, "
                         f"final loss {r['loss']:.4f} at step {r['step']}")
    return "\n".join(lines)


def plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["threshold"].astype(str) == str(LOSS_STAR)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    colors = {"sp": "C3", "mup": "C0", "ctxfree": "C2", "frozen": "C7"}
    for ax, metric, ylabel in zip(axes[:3], ("align_u", "probe_acc", "acc_reversed"),
                                  ("encoder energy on u at L*", "spectral probe acc at L*",
                                   "accuracy under context reversal at L*")):
        for arm in ("sp", "mup", "ctxfree", "frozen"):
            a = d[d["arm"] == arm]
            if a.empty:
                continue
            for s, g in a.groupby("seed"):
                g = g.sort_values("width")
                ax.plot(g["width"], g[metric], "o-", color=colors[arm], alpha=0.35, lw=1)
            m = a.groupby("width")[metric].mean()
            ax.plot(m.index, m.values, "s-", color=colors[arm], lw=2.5, label=arm)
        ax.set_xscale("log", base=2); ax.set_xlabel("head width M"); ax.set_ylabel(ylabel)
        if metric == "acc_reversed":
            ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.grid(True, alpha=0.3)
    axes[0].set_yscale("log"); axes[0].legend()
    ax = axes[3]
    a = d[d["arm"] == "lrmult"]
    if not a.empty:
        for s, g in a.groupby("seed"):
            g = g.sort_values("mult")
            ax.plot(g["mult"], g["align_u"], "o-", color="C3", alpha=0.35, lw=1)
            ax.plot(g["mult"], g["acc_reversed"], "^-", color="C1", alpha=0.35, lw=1)
        m = a.groupby("mult")[["align_u", "acc_reversed"]].mean()
        ax.plot(m.index, m["align_u"], "s-", color="C3", lw=2.5, label="a_u at L*")
        ax.plot(m.index, m["acc_reversed"], "D-", color="C1", lw=2.5, label="acc reversed at L*")
        ax.set_xscale("log", base=4); ax.set_xlabel("readout lr multiplier (M = 32)")
        ax.axhline(0.5, color="k", ls=":", lw=1); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle(f"Exp 2 v1: matched fit L* = {LOSS_STAR}; full-batch GD; linear encoder + ReLU CNN head")
    fig.tight_layout()
    out = OUT_DIR / "fig_exp2_v1.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] wrote {out}")
    plot_traj()


def plot_traj(seed: int = 0) -> None:
    """Trajectories a_u(t) and train loss for sp/mup/ctxfree at several widths (one seed):
    is the suppression transient (encoder aligns after the context fit) or persistent?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    cmap = plt.get_cmap("viridis")
    for ax, arm in zip(axes, ("sp", "mup", "ctxfree")):
        files = sorted(OUT_DIR.glob(f"traj_{arm}_M*_x1_s{seed}.csv"),
                       key=lambda f: int(f.stem.split("_M")[1].split("_")[0]))
        if not files:
            continue
        for i, f in enumerate(files):
            Mv = int(f.stem.split("_M")[1].split("_")[0])
            t = pd.read_csv(f)
            c = cmap(i / max(1, len(files) - 1))
            ax.plot(t["step"] + 1, t["align_u"], color=c, lw=1.8, label=f"M={Mv}")
            ax.plot(t["step"] + 1, t["loss"], color=c, lw=0.9, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("GD step"); ax.set_title(f"{arm} (seed {seed}): a_u solid, train loss dashed")
        ax.axhline(LOSS_STAR, color="k", ls=":", lw=1)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("encoder energy on u  /  loss")
    fig.tight_layout()
    out = OUT_DIR / f"fig_exp2_v1_traj_s{seed}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] wrote {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--stability-check", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--widths", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--mults", nargs="*", type=float, default=None)
    ap.add_argument("--lr", type=float, default=LR_DEFAULT)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 convolutions/matmuls (speed; ~1e-3 relative precision)")
    ap.add_argument("--benchmark", action="store_true",
                    help="time forward+backward at M=2048 with/without TF32")
    ap.add_argument("--selftest", action="store_true",
                    help="check the matmul head against nn.Conv2d (outputs and gradients)")
    args = ap.parse_args()
    if args.selftest:
        selftest_head(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[exp2] device {device}  tf32={args.tf32}")

    if args.benchmark:
        tau_b = float(json.loads(CALIB.read_text())["tau"]) if CALIB.exists() else 1.7
        spec = ProblemSpec(S=S, H=H, W=W, alpha=ALPHA, beta=BETA, tau=tau_b, sigma=SIGMA)
        data = build_data(0, spec, "iid", device)
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            for width in (512, 2048):
                torch.manual_seed(0)
                enc = Encoder(S, K).to(device); head = CNNHead(K, width, "sp").to(device)
                ps = list(enc.parameters()) + list(head.parameters())
                for _ in range(5):  # warm-up (cudnn autotune)
                    loss = margin_loss(head(enc(data["Xtr"])), data["ytr"]); loss.backward()
                    for p in ps: p.grad = None
                torch.cuda.synchronize(); t0 = time.time(); n = 30
                for _ in range(n):
                    loss = margin_loss(head(enc(data["Xtr"])), data["ytr"]); loss.backward()
                    for p in ps: p.grad = None
                torch.cuda.synchronize()
                print(f"[bench] tf32={tf32} M={width}: {(time.time() - t0) / n * 1e3:.1f} ms/step "
                      f"(loss {float(loss):.5f})")
        return

    if args.calibrate:
        calibrate(device)
        return
    if not CALIB.exists():
        raise SystemExit("run --calibrate first (tau must be fixed before training)")
    tau = float(json.loads(CALIB.read_text())["tau"])
    print(f"[exp2] tau = {tau:.4f} (from {CALIB})")

    if args.stability_check:
        stability_check(device, tau)
        return
    if args.collect or args.plot:
        df = collect()
        print(evaluate_predictions(df))
        (OUT_DIR / "PREDICTIONS.md").write_text(evaluate_predictions(df))
        if args.plot:
            plot(df)
        return

    total = 0
    for arm in args.arms:
        cfg = ARMS[arm]
        widths = args.widths or cfg["widths"]
        mults = args.mults or cfg["mults"]
        for width in widths:
            for mult in mults:
                for seed in args.seeds:
                    total += 1
                    tag = f"{arm}_M{width}_x{mult_tag(mult)}_s{seed}"
                    if args.skip_existing and (OUT_DIR / f"snap_{tag}.csv").exists():
                        print(f"[exp2] skip {tag}")
                        continue
                    print(f"[exp2] run {tag}  lr={args.lr:g}", flush=True)
                    run_one(arm, width, mult, seed, tau, args.lr, args.max_steps, device)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    print(f"[exp2] done: {total} runs")
    df = collect()
    print(evaluate_predictions(df))
    (OUT_DIR / "PREDICTIONS.md").write_text(evaluate_predictions(df))


if __name__ == "__main__":
    main()
