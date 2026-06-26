"""
Experiment 1.2 (v4): Two-Timescale Dynamics — universality across architectures.

Combines two things into one sweep:
  (a) extending SpatialCNN to wider widths (up to D=1024) to confirm the
      joint-vs-frozen gap continues widening as capacity grows;
  (b) adding SpatialViT (a small per-pixel-token transformer) so we can show
      the pattern is not CNN-specific.

Setup:
  architecture in {cnn, vit}
  conditions   in {joint, frozen}
  seeds        in {42, 43, 44}
  CNN widths   in {16, 64, 256, 1024}
  ViT widths   in {64, 128, 256}

We use Adam (matches typical spectroscopy ML setup) and noise=0.1 (the
"interesting" regime where joint training discovers shortcuts).
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
    SpatialViT,
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
LEARNING_RATE = 1e-3
DATA_NOISE = 0.10

ARCH_WIDTHS = {
    "cnn": [16, 64, 256, 1024],
    "vit": [64, 128, 256],
}
SEEDS = [42, 43, 44]
CONDITIONS = ["joint", "frozen"]

TAG = "v4_universal"


def build_spatial(arch: str, width: int) -> nn.Module:
    if arch == "cnn":
        return SpatialCNN(K=K, n_classes=N_CLASSES, width=width)
    elif arch == "vit":
        return SpatialViT(K=K, n_classes=N_CLASSES, width=width, H=H, W=W)
    raise ValueError(arch)


@dataclass
class RunResult:
    arch: str
    condition: str
    width: int
    seed: int
    n_spectral: int
    n_spatial: int
    egr_df: pd.DataFrame
    metrics_df: pd.DataFrame
    final_train_loss: float
    final_test_loss: float
    final_test_acc: float
    peak_test_acc: float


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


def train_one(arch: str, condition: str, width: int, seed: int, device: torch.device) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spectral = SpectralReduction(S=S, K=K)
    spatial = build_spatial(arch, width)
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
        arch=arch, condition=condition, width=width, seed=seed,
        n_spectral=sum(p.numel() for p in spectral.parameters()),
        n_spatial=sum(p.numel() for p in spatial.parameters()),
        egr_df=egr_df, metrics_df=metrics_df,
        final_train_loss=final["train_loss"],
        final_test_loss=final["test_loss"],
        final_test_acc=final["test_acc"],
        peak_test_acc=float(max(r["test_acc"] for r in metrics_rows)),
    )


def _smooth(series, win=20):
    return pd.Series(series).rolling(window=win, min_periods=1, center=True).mean()


def plot_joint_vs_frozen(results: list[RunResult], arch: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = sorted({r.width for r in results if r.arch == arch})

    fig, axes = plt.subplots(1, len(widths), figsize=(5 * len(widths), 4.5),
                              sharey=True)
    if len(widths) == 1:
        axes = [axes]
    for ax, w in zip(axes, widths):
        for cond, color in [("joint", "C3"), ("frozen", "C0")]:
            runs = [r for r in results
                    if r.arch == arch and r.width == w and r.condition == cond]
            if not runs:
                continue
            accs = np.stack([r.metrics_df["test_acc"].values for r in runs])
            mean = accs.mean(0); std = accs.std(0)
            epochs = np.arange(len(mean))
            ax.plot(epochs, mean, color=color, lw=2, label=cond)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=color, alpha=0.2)
        ax.set_xlabel("epoch")
        ax.set_title(f"D = {w}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("test accuracy (mean ± 1σ)")
    fig.suptitle(f"Exp 1.2 v4: joint vs frozen test accuracy ({arch.upper()})")
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2{TAG}_{arch}_joint_vs_frozen.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 {TAG}] wrote {out}")


def plot_egr_overlay(results: list[RunResult], arch: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = sorted({r.width for r in results if r.arch == arch})

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["C0", "C2", "C3", "C4", "C5"]
    for i, w in enumerate(widths):
        runs = [r for r in results
                if r.arch == arch and r.width == w and r.condition == "joint"]
        if not runs:
            continue
        smoothed = [_smooth(r.egr_df["egr"], win=20).values for r in runs]
        n_min = min(len(s) for s in smoothed)
        stacked = np.stack([s[:n_min] for s in smoothed])
        mean = stacked.mean(0); std = stacked.std(0)
        steps = np.arange(n_min)
        ax.plot(steps, mean, color=palette[i % len(palette)], lw=2,
                label=f"D = {w}")
        ax.fill_between(steps,
                        np.clip(mean - std, 1e-12, None),
                        mean + std,
                        color=palette[i % len(palette)], alpha=0.18)
    ax.set_xlabel("step")
    ax.set_ylabel("EGR")
    ax.set_yscale("log")
    ax.set_title(f"Exp 1.2 v4: EGR trajectories ({arch.upper()}, joint)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2{TAG}_{arch}_egr_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 {TAG}] wrote {out}")


def plot_universality(results: list[RunResult]) -> None:
    """Side-by-side: CNN and ViT joint-vs-frozen gap at the largest width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = sorted(ARCH_WIDTHS.keys())
    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 5),
                              sharey=True)
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        w = max(r.width for r in results if r.arch == arch)
        for cond, color in [("joint", "C3"), ("frozen", "C0")]:
            runs = [r for r in results
                    if r.arch == arch and r.width == w and r.condition == cond]
            if not runs:
                continue
            accs = np.stack([r.metrics_df["test_acc"].values for r in runs])
            mean = accs.mean(0); std = accs.std(0)
            epochs = np.arange(len(mean))
            ax.plot(epochs, mean, color=color, lw=2, label=cond)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=color, alpha=0.2)
        ax.set_xlabel("epoch")
        ax.set_title(f"{arch.upper()} (largest D = {w})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("test accuracy (mean ± 1σ)")
    fig.suptitle("Exp 1.2 v4: universality across architectures (joint vs frozen)")
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2{TAG}_universality.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 {TAG}] wrote {out}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_2 {TAG}] device = {device}")
    print(f"[exp1_2 {TAG}] arch_widths = {ARCH_WIDTHS}")
    print(f"[exp1_2 {TAG}] seeds = {SEEDS}  conditions = {CONDITIONS}")
    total = sum(len(w) for w in ARCH_WIDTHS.values()) * len(SEEDS) * len(CONDITIONS)
    print(f"[exp1_2 {TAG}] total runs = {total}")
    print()

    all_results: list[RunResult] = []
    summary_rows = []
    run_i = 0
    for arch, widths in ARCH_WIDTHS.items():
        for cond in CONDITIONS:
            for w in widths:
                for seed in SEEDS:
                    run_i += 1
                    print(f"[exp1_2 {TAG}] [{run_i}/{total}] "
                          f"arch={arch} cond={cond} D={w} seed={seed} ...",
                          flush=True)
                    r = train_one(arch, cond, w, seed, device)
                    all_results.append(r)
                    summary_rows.append({
                        "arch": arch, "condition": cond, "width": w, "seed": seed,
                        "n_spectral": r.n_spectral,
                        "n_spatial": r.n_spatial,
                        "final_train_loss": r.final_train_loss,
                        "final_test_loss": r.final_test_loss,
                        "final_test_acc": r.final_test_acc,
                        "peak_test_acc": r.peak_test_acc,
                    })
                    print(f"             train={r.final_train_loss:.4f}  "
                          f"test={r.final_test_loss:.4f}  "
                          f"acc={r.final_test_acc:.4f}  "
                          f"peak={r.peak_test_acc:.4f}  "
                          f"C_g={r.n_spatial}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / f"exp1_2{TAG}_summary.csv", index=False)

    for arch in ARCH_WIDTHS.keys():
        plot_joint_vs_frozen(all_results, arch)
        plot_egr_overlay(all_results, arch)
    plot_universality(all_results)

    print(f"\n=== Exp 1.2 {TAG} summary ===")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
