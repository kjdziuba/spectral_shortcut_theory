"""
Experiment 1.2 (v3): Two-Timescale Dynamics — with a real spatial pathway.

CRITICAL change from v1/v2: the spatial model is now SpatialCNN, a small
2-layer Conv2d stack. This is the first version with an actual spatial
receptive field (5x5 after the two 3x3 convs) — without spatial mixing,
the framing of "joint training discovers spatial shortcuts" cannot be
demonstrated.

We compare joint training vs frozen-random spectral. Adam optimizer
matches the typical setup in the spectroscopy ML literature.

Outputs:
  - per-condition diagnostic plots (loss, accuracy, gradient norms, EGR)
  - cross-width comparison overlays
  - joint-vs-frozen test-accuracy comparison
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from synthetic.data import make_problem  # noqa: E402
from synthetic.models import (  # noqa: E402
    SpectralReduction,
    SpatialCNN,
    CompositionModel,
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
EPOCHS = 150
LEARNING_RATE = 1e-3        # Adam default — same as spec lit setup
DATA_NOISE = 0.10           # back to standard noise level

# CNN widths: parameter count grows ~163 * D (9-kernel x K + bias x 2 layers)
WIDTHS = [16, 64, 256]
SEEDS = [42, 43, 44]
CONDITIONS = ["joint", "frozen"]
TAG = "v3_cnn"               # suffix on output filenames


@dataclass
class RunResult:
    condition: str
    width: int
    seed: int
    egr_df: pd.DataFrame
    metrics_df: pd.DataFrame
    final_train_loss: float
    final_test_loss: float
    final_test_acc: float


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


def train_one(condition: str, width: int, seed: int, device: torch.device) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialCNN(K=K, n_classes=N_CLASSES, width=width)
    model = CompositionModel(spectral, spatial).to(device)

    if condition == "frozen":
        for p in spectral.parameters():
            p.requires_grad = False

    X_tr, y_tr = make_problem(N_TRAIN, S=S, H=H, W=W,
                              n_classes=N_CLASSES,
                              alpha=1.0, beta=1.0,
                              noise=DATA_NOISE, seed=seed)
    X_te, y_te = make_problem(N_TEST, S=S, H=H, W=W,
                              n_classes=N_CLASSES,
                              alpha=1.0, beta=1.0,
                              noise=DATA_NOISE, seed=seed + 1000)
    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_te = X_te.to(device); y_te = y_te.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    logger = EGRLogger(model)

    n_train = X_tr.shape[0]
    steps_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    global_step = 0

    metrics_rows = []
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
        metrics_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

    egr_df = logger.to_dataframe()
    metrics_df = pd.DataFrame(metrics_rows)
    final = metrics_rows[-1]
    return RunResult(
        condition=condition, width=width, seed=seed,
        egr_df=egr_df, metrics_df=metrics_df,
        final_train_loss=final["train_loss"],
        final_test_loss=final["test_loss"],
        final_test_acc=final["test_acc"],
    )


def _smooth(series, win=20):
    return pd.Series(series).rolling(window=win, min_periods=1, center=True).mean()


def plot_per_condition(results: list[RunResult], condition: str, width: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in results if r.condition == condition and r.width == width]
    if not runs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for r in runs:
        ax.plot(r.metrics_df["epoch"], r.metrics_df["train_loss"],
                color="C0", alpha=0.7)
        ax.plot(r.metrics_df["epoch"], r.metrics_df["test_loss"],
                color="C3", alpha=0.7, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("train (solid) and test (dashed) loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for r in runs:
        ax.plot(r.metrics_df["epoch"], r.metrics_df["test_acc"], alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("test accuracy")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i, r in enumerate(runs):
        ax.plot(r.egr_df["step"], _smooth(r.egr_df["grad_phi_norm"]),
                color="C0", alpha=0.7,
                label=r"$\|\nabla_\phi L\|$" if i == 0 else None)
        ax.plot(r.egr_df["step"], _smooth(r.egr_df["grad_theta_norm"]),
                color="C3", alpha=0.7,
                label=r"$\|\nabla_\theta L\|$" if i == 0 else None)
    ax.set_xlabel("step")
    ax.set_ylabel("gradient L2 norm")
    ax.set_yscale("log")
    ax.set_title("gradient norms (smoothed)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    for r in runs:
        if condition == "joint":
            ax.plot(r.egr_df["step"], _smooth(r.egr_df["egr"]), alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("EGR")
    ax.set_yscale("log")
    ax.set_title("EGR (smoothed)" if condition == "joint" else "frozen: EGR undefined")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Exp 1.2 {TAG}, {condition}, CNN width D = {width}  "
                 f"(seeds: {[r.seed for r in runs]})")
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2{TAG}_{condition}_D{width}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 {TAG}] wrote {out}")


def plot_comparison(results: list[RunResult]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = sorted({r.width for r in results})

    # (A) joint vs frozen test accuracy by width
    fig, axes = plt.subplots(1, len(widths), figsize=(5 * len(widths), 5),
                              sharey=True)
    if len(widths) == 1:
        axes = [axes]
    for ax, w in zip(axes, widths):
        for cond, color in [("joint", "C3"), ("frozen", "C0")]:
            runs = [r for r in results if r.width == w and r.condition == cond]
            if not runs:
                continue
            accs = np.stack([r.metrics_df["test_acc"].values for r in runs])
            mean = accs.mean(0); std = accs.std(0)
            epochs = np.arange(len(mean))
            ax.plot(epochs, mean, color=color, lw=2, label=cond)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("epoch")
        ax.set_title(f"CNN width D = {w}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("test accuracy (mean ± 1σ)")
    fig.suptitle("Exp 1.2 v3 (CNN): joint vs frozen spectral, by spatial width")
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2{TAG}_joint_vs_frozen.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 {TAG}] wrote {out}")

    # (B) EGR overlay across widths (joint only)
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["C0", "C2", "C3", "C4", "C5"]
    for i, w in enumerate(widths):
        runs = [r for r in results if r.width == w and r.condition == "joint"]
        smoothed = [_smooth(r.egr_df["egr"], win=20).values for r in runs]
        n_min = min(len(s) for s in smoothed)
        stacked = np.stack([s[:n_min] for s in smoothed])
        mean = stacked.mean(0); std = stacked.std(0)
        steps = np.arange(n_min)
        ax.plot(steps, mean, color=palette[i], lw=2, label=f"D = {w}")
        ax.fill_between(steps,
                        np.clip(mean - std, 1e-12, None),
                        mean + std, color=palette[i], alpha=0.18)
    ax.set_xlabel("step")
    ax.set_ylabel("EGR")
    ax.set_yscale("log")
    ax.set_title("Exp 1.2 v3: EGR trajectories (CNN, joint training)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2{TAG}_egr_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 {TAG}] wrote {out}")

    # (C) gradient norms at largest width, joint
    largest = max(widths)
    runs = [r for r in results if r.width == largest and r.condition == "joint"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        ax.plot(r.egr_df["step"], _smooth(r.egr_df["grad_phi_norm"]),
                color="C0", alpha=0.7, lw=1.2)
        ax.plot(r.egr_df["step"], _smooth(r.egr_df["grad_theta_norm"]),
                color="C3", alpha=0.7, lw=1.2)
    ax.plot([], [], color="C0", lw=2, label=r"$\|\nabla_\phi L\|$")
    ax.plot([], [], color="C3", lw=2, label=r"$\|\nabla_\theta L\|$")
    ax.set_xlabel("step")
    ax.set_ylabel("gradient L2 norm")
    ax.set_yscale("log")
    ax.set_title(f"Exp 1.2 v3: gradient norms at D={largest} (CNN, joint)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2{TAG}_grad_norms_D{largest}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 {TAG}] wrote {out}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_2 {TAG}] device = {device}")
    print(f"[exp1_2 {TAG}] architecture = SpatialCNN (5x5 receptive field)")
    print(f"[exp1_2 {TAG}] optimizer = Adam(lr={LEARNING_RATE})")
    print(f"[exp1_2 {TAG}] data noise = {DATA_NOISE}")
    print(f"[exp1_2 {TAG}] widths: {WIDTHS}, seeds: {SEEDS}, epochs: {EPOCHS}")
    print(f"[exp1_2 {TAG}] conditions: {CONDITIONS}")

    all_results: list[RunResult] = []
    summary_rows = []
    for cond in CONDITIONS:
        for w in WIDTHS:
            for seed in SEEDS:
                print(f"[exp1_2 {TAG}] cond={cond} D={w} seed={seed} ...",
                      flush=True)
                r = train_one(cond, w, seed, device)
                all_results.append(r)
                r.egr_df.to_csv(
                    RESULTS_DIR / f"exp1_2{TAG}_{cond}_D{w}_seed{seed}_egr.csv",
                    index=False,
                )
                r.metrics_df.to_csv(
                    RESULTS_DIR / f"exp1_2{TAG}_{cond}_D{w}_seed{seed}_metrics.csv",
                    index=False,
                )
                summary_rows.append({
                    "condition": cond, "width": w, "seed": seed,
                    "final_train_loss": r.final_train_loss,
                    "final_test_loss": r.final_test_loss,
                    "final_test_acc": r.final_test_acc,
                })
                print(f"             train={r.final_train_loss:.4f}  "
                      f"test={r.final_test_loss:.4f}  "
                      f"acc={r.final_test_acc:.4f}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / f"exp1_2{TAG}_summary.csv", index=False
    )

    for cond in CONDITIONS:
        for w in WIDTHS:
            plot_per_condition(all_results, cond, w)
    plot_comparison(all_results)

    print(f"\n=== Exp 1.2 {TAG} summary ===")
    print(pd.DataFrame(summary_rows).to_string(
        index=False, float_format=lambda v: f"{v:.4g}"
    ))


if __name__ == "__main__":
    main()
