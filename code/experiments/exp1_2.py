"""
Experiment 1.2: Two-Timescale Dynamics.

Validates Theorem 2: after the per-pixel cross-entropy loss saturates (the
spatial block g_phi has fit the dominant pattern), the gradient norm to the
spectral parameters theta collapses exponentially. This produces a clear
two-timescale separation between

  ||grad_phi  L||   (fast, drives initial loss decay)
  ||grad_theta L||  (slow, collapses to ~0 after saturation)

We train the same compositional model F(X) = g_phi(f_theta(X)) at two
spatial widths D in {32, 256} so the reader can see that the timescale
separation is qualitative (it shows up at both capacities) but its strength
scales with the spatial capacity, consistent with Exp 1.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make `synthetic`, `egr`, ... importable.
CODE_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from synthetic.data import make_problem  # noqa: E402
from synthetic.models import (  # noqa: E402
    SpectralReduction,
    SpatialMLP,
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
WIDTHS = [32, 256]
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
SEED = 42


def _per_pixel_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy across all pixels.

    logits: (B, H, W, C)   y: (B, H, W)
    """
    B, Hh, Ww, C = logits.shape
    return F.cross_entropy(logits.reshape(-1, C), y.reshape(-1))


@torch.no_grad()
def _eval(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          batch_size: int) -> tuple[float, float]:
    """Return (mean per-pixel CE loss, mean per-pixel accuracy) over X, y."""
    model.eval()
    n = X.shape[0]
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)  # (B, H, W, C)
        loss = _per_pixel_ce(logits, yb)
        bs = xb.shape[0]
        # CE is mean over all pixels in the batch; weight by pixel count.
        n_pix = bs * H * W
        total_loss += float(loss.item()) * n_pix
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == yb).sum().item())
        total_pixels += n_pix
    model.train()
    return total_loss / total_pixels, total_correct / total_pixels


def run_one(D: int, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one width D end-to-end. Returns (egr_df, metrics_df)."""
    # Reset RNG so widths share the same data and init scheme.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------ Model ------------------
    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialMLP(K=K, n_classes=N_CLASSES, width=D)
    model = CompositionModel(spectral, spatial).to(device)

    # ------------------ Data -------------------
    # Use distinct seeds for train and test so they are independent draws.
    X_train, y_train = make_problem(
        n_samples=N_TRAIN, S=S, H=H, W=W, n_classes=N_CLASSES,
        alpha=1.0, beta=1.0, seed=SEED,
    )
    X_test, y_test = make_problem(
        n_samples=N_TEST, S=S, H=H, W=W, n_classes=N_CLASSES,
        alpha=1.0, beta=1.0, seed=SEED + 1,
    )
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # ------------------ Optimiser + EGR ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    egr = EGRLogger(model)

    # ------------------ Train loop ---------------
    metrics_rows: list[dict] = []
    step = 0
    n_train = X_train.shape[0]
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        # Shuffle indices each epoch with the global torch RNG.
        perm = torch.randperm(n_train, device=device)

        epoch_loss_sum = 0.0
        epoch_pixels = 0

        model.train()
        for bi in range(n_batches):
            idx = perm[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            xb = X_train[idx]
            yb = y_train[idx]

            logits = model(xb)
            loss = _per_pixel_ce(logits, yb)

            loss.backward()
            egr.log_step(step)           # BEFORE optimizer.step / zero_grad
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            bs = xb.shape[0]
            n_pix = bs * H * W
            epoch_loss_sum += float(loss.item()) * n_pix
            epoch_pixels += n_pix
            step += 1

        train_loss = epoch_loss_sum / epoch_pixels
        test_loss, test_acc = _eval(model, X_test, y_test, BATCH_SIZE)

        metrics_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"[exp1_2 D={D:3d}] epoch={epoch:3d}  "
                f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  "
                f"test_acc={test_acc:.4f}"
            )

    egr_df = egr.to_dataframe()
    metrics_df = pd.DataFrame(metrics_rows)

    egr_path = RESULTS_DIR / f"exp1_2_D{D}_egr.csv"
    metrics_path = RESULTS_DIR / f"exp1_2_D{D}_metrics.csv"
    egr_df.to_csv(egr_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[exp1_2 D={D}] wrote {egr_path}")
    print(f"[exp1_2 D={D}] wrote {metrics_path}")

    return egr_df, metrics_df


def plot_one(D: int, egr_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Per-width 4-panel figure: losses, acc, grad norms, EGR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) train / test loss vs epoch
    ax = axes[0, 0]
    ax.plot(metrics_df["epoch"], metrics_df["train_loss"], label="train", color="C0")
    ax.plot(metrics_df["epoch"], metrics_df["test_loss"], label="test", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-pixel CE loss")
    ax.set_title(f"(a) Loss vs epoch  [D={D}]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (b) test accuracy vs epoch
    ax = axes[0, 1]
    ax.plot(metrics_df["epoch"], metrics_df["test_acc"], color="C2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-pixel accuracy")
    ax.set_title(f"(b) Test accuracy vs epoch  [D={D}]")
    ax.grid(True, alpha=0.3)

    # (c) grad norms vs step (log y)
    ax = axes[1, 0]
    ax.semilogy(egr_df["step"], egr_df["grad_theta_norm"],
                label=r"$\|\nabla_\theta L\|$", color="C3")
    ax.semilogy(egr_df["step"], egr_df["grad_phi_norm"],
                label=r"$\|\nabla_\phi L\|$", color="C0")
    ax.set_xlabel("step")
    ax.set_ylabel("grad L2 norm  (log)")
    ax.set_title(f"(c) Block gradient norms vs step  [D={D}]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # (d) EGR vs step (log y)
    ax = axes[1, 1]
    ax.semilogy(egr_df["step"], egr_df["egr"], color="C4")
    ax.set_xlabel("step")
    ax.set_ylabel(r"EGR = $\|\nabla_\theta L\| / \|\nabla_\phi L\|$  (log)")
    ax.set_title(f"(d) EGR vs step  [D={D}]")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_2_dynamics_D{D}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2 D={D}] wrote {out}")


def plot_comparison(results: dict[int, pd.DataFrame]) -> None:
    """Overlay EGR trajectories for all widths on a single log-y plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (D, egr_df) in enumerate(sorted(results.items())):
        ax.semilogy(
            egr_df["step"], egr_df["egr"],
            label=f"D = {D}", color=f"C{i}",
        )
    ax.set_xlabel("step")
    ax.set_ylabel(r"EGR = $\|\nabla_\theta L\| / \|\nabla_\phi L\|$  (log)")
    ax.set_title("Exp 1.2: EGR trajectories across spatial capacity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = RESULTS_DIR / "exp1_2_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_2] wrote {out}")


def _summary_row(D: int, egr_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict:
    """Compute a one-line summary for the printed table."""
    final = metrics_df.iloc[-1]
    # Take last 10% of steps for the "post-saturation" averages.
    n_steps = len(egr_df)
    tail = max(1, n_steps // 10)
    tail_df = egr_df.iloc[-tail:]
    return {
        "D": D,
        "final_train_loss": float(final["train_loss"]),
        "final_test_loss": float(final["test_loss"]),
        "final_test_acc": float(final["test_acc"]),
        "init_grad_theta": float(egr_df["grad_theta_norm"].iloc[0]),
        "final_grad_theta": float(tail_df["grad_theta_norm"].mean()),
        "init_grad_phi": float(egr_df["grad_phi_norm"].iloc[0]),
        "final_grad_phi": float(tail_df["grad_phi_norm"].mean()),
        "init_egr": float(egr_df["egr"].iloc[0]),
        "final_egr": float(tail_df["egr"].mean()),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_2] device = {device}")

    egr_results: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for D in WIDTHS:
        print(f"\n=== Running D = {D} ===")
        egr_df, metrics_df = run_one(D, device)
        plot_one(D, egr_df, metrics_df)
        egr_results[D] = egr_df
        summary_rows.append(_summary_row(D, egr_df, metrics_df))

    plot_comparison(egr_results)

    summary = pd.DataFrame(summary_rows)
    print("\n=== Exp 1.2 summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
