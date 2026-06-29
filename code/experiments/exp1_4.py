"""
Experiment 1.4: EGR as a predictor of final test accuracy.

Validates the EGR diagnostic from Section 6: at training step t,
EGR(t) = ||grad_theta L|| / ||grad_phi L|| reflects the gradient
asymmetry between modules. A low EGR early in training should predict
poor final test accuracy because the spectral module is being starved
by the spatial module's rapid loss saturation.

We sweep CNN width D (controls capacity ratio) and data noise (controls
SNR), 3 seeds each, log EGR throughout training, then correlate
early-window EGR with final test accuracy.

Predictions:
  - Negative correlation: low EGR(early) -> low final acc
  - Capacity ratio C_g / C_f modulates the effect
  - Effect is monotonic across the sweep
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
# Config
# ------------------------------------------------------------------ #
S = 64
K = 16
H = W = 16
N_CLASSES = 2
N_TRAIN = 512
N_TEST = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

WIDTHS = [16, 64, 256, 1024]
NOISE_LEVELS = [0.05, 0.10]
SEEDS = [42, 43, 44]
ALPHA = 1.0
BETA = 1.0

# Windows for EGR statistics (in steps). Total steps ~ 1600 for 100 epochs.
EARLY_WINDOW = (50, 200)
MID_WINDOW = (400, 800)
LATE_WINDOW = (1200, 1600)
# Final accuracy: average over last 10 epochs
FINAL_LAST_K = 10


@dataclass
class RunResult:
    width: int
    noise: float
    seed: int
    n_spatial: int
    n_spectral: int
    # multiple EGR statistics — we'll see which correlates with final acc
    egr_early_mean: float
    egr_mid_mean: float
    egr_late_mean: float
    egr_min: float          # deepest collapse over training
    egr_depth: float        # = egr_early_mean - egr_min (collapse depth)
    final_test_acc: float
    peak_test_acc: float
    overfit_gap: float      # peak_test_acc - final_test_acc
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
        preds = logits.argmax(dim=-1)
        acc = (preds == y_dev).float().mean().item()
    model.train()
    return loss, acc


def train_one(width: int, noise: float, seed: int, device: torch.device) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialCNN(K=K, n_classes=N_CLASSES, width=width)
    model = CompositionModel(spectral, spatial).to(device)

    X_tr, y_tr = make_problem(N_TRAIN, S=S, H=H, W=W,
                              n_classes=N_CLASSES,
                              alpha=ALPHA, beta=BETA,
                              noise=noise, seed=seed)
    X_te, y_te = make_problem(N_TEST, S=S, H=H, W=W,
                              n_classes=N_CLASSES,
                              alpha=ALPHA, beta=BETA,
                              noise=noise, seed=seed + 1000)
    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_te = X_te.to(device); y_te = y_te.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger = EGRLogger(model)

    n_train = X_tr.shape[0]
    steps_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    global_step = 0
    epoch_metrics = []
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
        epoch_metrics.append({
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "test_acc": test_acc,
        })

    egr_df = logger.to_dataframe()
    # Save full EGR DataFrame per run (post-hoc reanalysis)
    egr_path = RESULTS_DIR / f"exp1_4_egr_D{width}_n{noise}_s{seed}.csv"
    egr_df.to_csv(egr_path, index=False)

    def window_mean(lo, hi):
        w = egr_df[(egr_df["step"] >= lo) & (egr_df["step"] <= hi)]
        return float(w["egr"].mean()) if not w.empty else float("nan")

    egr_early = window_mean(*EARLY_WINDOW)
    egr_mid = window_mean(*MID_WINDOW)
    egr_late = window_mean(*LATE_WINDOW)
    egr_min = float(egr_df["egr"].rolling(window=20, min_periods=1).mean().min())
    egr_depth = egr_early - egr_min

    # Final metrics: average over last K epochs
    final_acc = float(np.mean([m["test_acc"] for m in epoch_metrics[-FINAL_LAST_K:]]))
    peak_acc = float(np.max([m["test_acc"] for m in epoch_metrics]))
    final_train_loss = epoch_metrics[-1]["train_loss"]
    final_test_loss = float(np.mean([m["test_loss"] for m in epoch_metrics[-FINAL_LAST_K:]]))
    overfit_gap = peak_acc - final_acc

    return RunResult(
        width=width, noise=noise, seed=seed,
        n_spatial=sum(p.numel() for p in spatial.parameters()),
        n_spectral=sum(p.numel() for p in spectral.parameters()),
        egr_early_mean=egr_early,
        egr_mid_mean=egr_mid,
        egr_late_mean=egr_late,
        egr_min=egr_min,
        egr_depth=egr_depth,
        final_test_acc=final_acc,
        peak_test_acc=peak_acc,
        overfit_gap=overfit_gap,
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
    )


def plot_results(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    metrics = [
        ("egr_early_mean", "EGR (early: steps 50–200)"),
        ("egr_mid_mean",   "EGR (mid: steps 400–800)"),
        ("egr_late_mean",  "EGR (late: steps 1200–1600)"),
        ("egr_min",        "EGR (minimum over training)"),
        ("egr_depth",      "EGR depth (early − min)"),
    ]
    targets = [
        ("final_test_acc", "final test accuracy"),
        ("overfit_gap",    "overfit gap (peak − final)"),
    ]

    widths = sorted(df["width"].unique())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(widths)))
    width_to_color = {w: c for w, c in zip(widths, cmap)}

    fig, axes = plt.subplots(len(targets), len(metrics),
                             figsize=(3.5 * len(metrics), 4 * len(targets)),
                             squeeze=False)
    correlations = {}
    for i, (target_col, target_label) in enumerate(targets):
        for j, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[i][j]
            x = df[metric_col].values
            y = df[target_col].values
            finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
            xs = x[finite]
            ys = y[finite]
            try:
                pearson, _ = pearsonr(np.log(xs + 1e-12), ys)
                spearman, _ = spearmanr(xs, ys)
            except Exception:
                pearson = float("nan"); spearman = float("nan")
            correlations[(target_col, metric_col)] = (pearson, spearman)

            for w in widths:
                sub = df[(df["width"] == w) & finite]
                ax.scatter(sub[metric_col], sub[target_col],
                           s=50, color=width_to_color[w], alpha=0.85,
                           edgecolor="black", linewidth=0.3,
                           label=f"D = {w}" if (i == 0 and j == 0) else None)
            ax.set_xscale("log")
            ax.set_xlabel(metric_label, fontsize=9)
            if j == 0:
                ax.set_ylabel(target_label)
            ax.set_title(f"Pearson r={pearson:+.2f}, ρ={spearman:+.2f}",
                         fontsize=9)
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(title="CNN width", loc="lower right", fontsize=8)

    fig.suptitle("Exp 1.4: EGR statistics as predictors")
    fig.tight_layout()
    out = RESULTS_DIR / "exp1_4_correlation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_4] wrote {out}")

    # Print top correlations
    print()
    print("=== Correlations ===")
    print(f"{'target':<20} {'metric':<20} {'Pearson':>10} {'Spearman':>10}")
    for (t, m), (p, s) in correlations.items():
        print(f"{t:<20} {m:<20} {p:>+10.3f} {s:>+10.3f}")
    return correlations


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(WIDTHS) * len(NOISE_LEVELS) * len(SEEDS)
    print(f"[exp1_4] device = {device}")
    print(f"[exp1_4] {total} runs across widths {WIDTHS}, noises {NOISE_LEVELS}, seeds {SEEDS}")

    rows = []
    i = 0
    for w in WIDTHS:
        for n in NOISE_LEVELS:
            for seed in SEEDS:
                i += 1
                print(f"[exp1_4] [{i}/{total}] D={w} noise={n} seed={seed} ...",
                      flush=True)
                r = train_one(w, n, seed, device)
                rows.append({
                    "width": r.width, "noise": r.noise, "seed": r.seed,
                    "C_g": r.n_spatial, "C_f": r.n_spectral,
                    "ratio": r.n_spatial / r.n_spectral,
                    "egr_early_mean": r.egr_early_mean,
                    "egr_mid_mean": r.egr_mid_mean,
                    "egr_late_mean": r.egr_late_mean,
                    "egr_min": r.egr_min,
                    "egr_depth": r.egr_depth,
                    "final_test_acc": r.final_test_acc,
                    "peak_test_acc": r.peak_test_acc,
                    "overfit_gap": r.overfit_gap,
                    "final_train_loss": r.final_train_loss,
                    "final_test_loss": r.final_test_loss,
                })
                print(f"          EGR(e/m/l/min)={r.egr_early_mean:.3f}/"
                      f"{r.egr_mid_mean:.3f}/{r.egr_late_mean:.3f}/{r.egr_min:.3f}  "
                      f"final={r.final_test_acc:.3f}  peak={r.peak_test_acc:.3f}  "
                      f"overfit={r.overfit_gap:.3f}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp1_4_summary.csv", index=False)
    print(f"\n=== Exp 1.4 summary ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    plot_results(df)


if __name__ == "__main__":
    main()
