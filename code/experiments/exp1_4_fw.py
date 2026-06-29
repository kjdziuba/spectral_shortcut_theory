"""
Experiment 1.4 fixed-width follow-up.

Earlier we showed EGR depth correlates strongly with final test accuracy
across our capacity sweep (r = -0.78). Adversarial verification revealed
that ~77% of the pooled correlation was driven by capacity (partial r
controlling for log(width) collapsed to -0.17). The honest framing is
"capacity-aware diagnostic" — informative only when capacity is held
constant.

This experiment tests whether a within-bin signal exists at meaningful
power. Fix CNN width D = 256. Vary three hyperparameters that the theory
predicts should ALSO modulate EGR depth:
  - data noise         (more noise -> harder to overfit -> shallower EGR)
  - learning rate      (higher LR -> faster CE saturation -> deeper EGR)
  - weight decay       (more decay -> milder overfit -> shallower EGR)

with N seeds per config. If EGR depth still correlates with final test
accuracy at fixed width across these knobs, the diagnostic is genuinely
capacity-independent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CODE_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from synthetic.data import make_problem  # noqa: E402
from synthetic.models import (  # noqa: E402
    SpectralReduction, SpatialCNN, CompositionModel,
)
from egr.callback import EGRLogger  # noqa: E402


RESULTS_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Config — D is fixed
# ------------------------------------------------------------------ #
S = 64
K = 16
H = W = 16
N_CLASSES = 2
N_TRAIN = 512
N_TEST = 128
BATCH_SIZE = 32
EPOCHS = 100
WIDTH = 256              # FIXED

ALPHA = 1.0
BETA = 1.0

# Hyperparameter grid — vary axes other than capacity
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]
LRS = [3e-4, 1e-3, 3e-3]
WEIGHT_DECAYS = [0.0, 1e-3]
SEEDS = [42, 43, 44, 45, 46, 47]    # 6 seeds

# Total: 4 * 3 * 2 * 6 = 144 runs ~= 70 min at ~30s each

EARLY_WINDOW = (50, 200)
MID_WINDOW = (400, 800)
LATE_WINDOW = (1200, 1600)
FINAL_LAST_K = 10


@dataclass
class RunResult:
    noise: float
    lr: float
    weight_decay: float
    seed: int
    egr_early: float
    egr_mid: float
    egr_late: float
    egr_min: float
    egr_depth: float
    final_test_acc: float
    peak_test_acc: float
    overfit_gap: float
    final_train_loss: float
    final_test_loss: float


def per_pixel_ce(logits, y):
    C = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, C), y.reshape(-1), reduction="mean")


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        y_dev = y.to(device)
        loss = per_pixel_ce(logits, y_dev).item()
        acc = (logits.argmax(dim=-1) == y_dev).float().mean().item()
    model.train()
    return loss, acc


def train_one(noise: float, lr: float, weight_decay: float, seed: int,
              device: torch.device) -> RunResult:
    torch.manual_seed(seed); np.random.seed(seed)
    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialCNN(K=K, n_classes=N_CLASSES, width=WIDTH)
    model = CompositionModel(spectral, spatial).to(device)

    X_tr, y_tr = make_problem(N_TRAIN, S=S, H=H, W=W, n_classes=N_CLASSES,
                              alpha=ALPHA, beta=BETA, noise=noise, seed=seed)
    X_te, y_te = make_problem(N_TEST, S=S, H=H, W=W, n_classes=N_CLASSES,
                              alpha=ALPHA, beta=BETA, noise=noise, seed=seed + 1000)
    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_te = X_te.to(device); y_te = y_te.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger = EGRLogger(model)

    n_train = X_tr.shape[0]
    steps_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    global_step = 0
    epoch_rows = []
    for epoch in range(EPOCHS):
        idx = torch.randperm(n_train, device=device)
        epoch_losses = []
        for s in range(steps_per_epoch):
            batch_idx = idx[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            xb = X_tr[batch_idx]; yb = y_tr[batch_idx]
            opt.zero_grad()
            logits = model(xb)
            loss = per_pixel_ce(logits, yb)
            loss.backward()
            logger.log_step(global_step)
            opt.step()
            epoch_losses.append(loss.item())
            global_step += 1
        train_loss = float(np.mean(epoch_losses))
        test_loss, test_acc = evaluate(model, X_te, y_te, device)
        epoch_rows.append({"epoch": epoch, "train_loss": train_loss,
                           "test_loss": test_loss, "test_acc": test_acc})

    egr_df = logger.to_dataframe()
    def winmean(lo, hi):
        w = egr_df[(egr_df["step"] >= lo) & (egr_df["step"] <= hi)]
        return float(w["egr"].mean()) if not w.empty else float("nan")
    egr_early = winmean(*EARLY_WINDOW)
    egr_mid = winmean(*MID_WINDOW)
    egr_late = winmean(*LATE_WINDOW)
    egr_min = float(egr_df["egr"].rolling(window=20, min_periods=1).mean().min())
    egr_depth = egr_early - egr_min

    final_acc = float(np.mean([r["test_acc"] for r in epoch_rows[-FINAL_LAST_K:]]))
    peak_acc = float(np.max([r["test_acc"] for r in epoch_rows]))
    final_train_loss = epoch_rows[-1]["train_loss"]
    final_test_loss = float(np.mean([r["test_loss"] for r in epoch_rows[-FINAL_LAST_K:]]))

    return RunResult(
        noise=noise, lr=lr, weight_decay=weight_decay, seed=seed,
        egr_early=egr_early, egr_mid=egr_mid, egr_late=egr_late,
        egr_min=egr_min, egr_depth=egr_depth,
        final_test_acc=final_acc, peak_test_acc=peak_acc,
        overfit_gap=peak_acc - final_acc,
        final_train_loss=final_train_loss, final_test_loss=final_test_loss,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    configs = [(n, lr, wd) for n in NOISE_LEVELS for lr in LRS for wd in WEIGHT_DECAYS]
    total = len(configs) * len(SEEDS)
    print(f"[exp1_4_fw] device = {device}")
    print(f"[exp1_4_fw] D = {WIDTH} (fixed)")
    print(f"[exp1_4_fw] total runs = {total}")

    i = 0
    for noise, lr, wd in configs:
        for seed in SEEDS:
            i += 1
            print(f"[exp1_4_fw] [{i}/{total}] noise={noise} lr={lr} "
                  f"wd={wd} seed={seed}", flush=True)
            r = train_one(noise, lr, wd, seed, device)
            rows.append({
                "noise": r.noise, "lr": r.lr, "weight_decay": r.weight_decay,
                "seed": r.seed,
                "egr_early": r.egr_early, "egr_mid": r.egr_mid,
                "egr_late": r.egr_late, "egr_min": r.egr_min,
                "egr_depth": r.egr_depth,
                "final_test_acc": r.final_test_acc,
                "peak_test_acc": r.peak_test_acc,
                "overfit_gap": r.overfit_gap,
                "final_train_loss": r.final_train_loss,
                "final_test_loss": r.final_test_loss,
            })
            print(f"          EGR depth={r.egr_depth:.3f}  "
                  f"final_acc={r.final_test_acc:.3f}  "
                  f"overfit_gap={r.overfit_gap:.3f}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp1_4_fw_summary.csv", index=False)

    # Compute headline correlations
    from scipy.stats import pearsonr, spearmanr
    for col in ["egr_early", "egr_mid", "egr_late", "egr_min", "egr_depth"]:
        for target in ["final_test_acc", "overfit_gap"]:
            x = df[col].values; y = df[target].values
            finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if finite.sum() < 3:
                continue
            r_p, p_p = pearsonr(np.log(x[finite] + 1e-12), y[finite])
            r_s, p_s = spearmanr(x[finite], y[finite])
            print(f"  {col:>12s} vs {target:>15s}: "
                  f"Pearson r={r_p:+.3f} (p={p_p:.2e})  "
                  f"Spearman ρ={r_s:+.3f} (p={p_s:.2e})")

    print(f"\nWrote {RESULTS_DIR / 'exp1_4_fw_summary.csv'}")


if __name__ == "__main__":
    main()
