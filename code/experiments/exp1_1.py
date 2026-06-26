"""
Experiment 1.1: Capacity Ablation.

Validates Theorem 1: lambda_max(H_phi_phi) scales with C_g, the spatial
(downstream) capacity. We hold the spectral reduction fixed (S=64, K=16)
and sweep the width D of the spatial MLP, measuring:

  - C_f: spectral parameter count (constant across sweep)
  - C_g: spatial parameter count (grows with D)
  - lambda_theta = lambda_max(H_theta_theta) (spectral Hessian block)
  - lambda_phi   = lambda_max(H_phi_phi)     (spatial Hessian block)
  - kappa        = lambda_phi / lambda_theta (curvature ratio)

Predicted relationship: log(lambda_phi) ~ log(C_g) with slope ~ 1 and
log(kappa) ~ log(C_g / C_f) with slope ~ 1 (reference line y = x).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Make `synthetic`, `hessian`, ... importable.
CODE_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from synthetic.data import make_problem  # noqa: E402
from synthetic.models import (  # noqa: E402
    SpectralReduction,
    SpatialMLP,
    CompositionModel,
)
from hessian.eigenvalues import top_eigenvalue_block, count_params  # noqa: E402


RESULTS_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run() -> pd.DataFrame:
    # -------------------------------------------------------------------- #
    # Configuration
    # -------------------------------------------------------------------- #
    S = 64
    K = 16
    H = W = 16
    n_classes = 2
    n_samples = 64
    capacities = [16, 32, 64, 128, 256, 512]
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_1] device = {device}")

    rows: list[dict] = []

    for D in capacities:
        # Reset RNG state so the only thing that varies is width D.
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ------------------------------------------------------------- #
        # Model
        # ------------------------------------------------------------- #
        spectral = SpectralReduction(S=S, K=K)
        spatial = SpatialMLP(K=K, n_classes=n_classes, width=D)
        model = CompositionModel(spectral, spatial).to(device)

        # ------------------------------------------------------------- #
        # Data
        # ------------------------------------------------------------- #
        X, y = make_problem(
            n_samples=n_samples,
            S=S,
            H=H,
            W=W,
            n_classes=n_classes,
            alpha=1.0,
            beta=1.0,
            seed=seed,
        )
        X = X.to(device)
        y = y.to(device)

        # ------------------------------------------------------------- #
        # Hessian block top eigenvalues
        # ------------------------------------------------------------- #
        lambda_theta = top_eigenvalue_block(
            model, X, y, block="spectral", n_iter=20, device=device,
        )
        lambda_phi = top_eigenvalue_block(
            model, X, y, block="spatial", n_iter=20, device=device,
        )

        C_f = count_params(spectral)
        C_g = count_params(spatial)
        ratio = C_g / C_f
        kappa = lambda_phi / lambda_theta if lambda_theta > 0 else float("nan")

        row = {
            "D": D,
            "C_f": C_f,
            "C_g": C_g,
            "ratio": ratio,
            "lambda_theta": float(lambda_theta),
            "lambda_phi": float(lambda_phi),
            "kappa": float(kappa),
        }
        rows.append(row)
        print(
            f"[exp1_1] D={D:4d}  C_f={C_f:7d}  C_g={C_g:8d}  "
            f"ratio={ratio:8.3f}  lambda_theta={lambda_theta:.4e}  "
            f"lambda_phi={lambda_phi:.4e}  kappa={kappa:.4e}"
        )

    df = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "exp1_1_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[exp1_1] wrote {csv_path}")

    return df


def _fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of log y = slope * log x + intercept."""
    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def plot(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --------------------------------------------------------------- #
    # Plot 1: lambda_phi vs C_g (log-log)
    # --------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 5))
    Cg = df["C_g"].to_numpy(dtype=float)
    lphi = df["lambda_phi"].to_numpy(dtype=float)

    ax.loglog(Cg, lphi, "o-", color="C0", label=r"measured $\lambda_\phi$")

    slope, intercept = _fit_loglog_slope(Cg, lphi)
    fit_x = np.array([Cg.min(), Cg.max()])
    fit_y = np.exp(intercept) * fit_x**slope
    ax.loglog(
        fit_x,
        fit_y,
        "--",
        color="C1",
        label=f"fit slope = {slope:.2f}",
    )

    # Reference: predicted slope = 1, anchored at first data point.
    ref_y = lphi[0] * (fit_x / Cg[0])
    ax.loglog(fit_x, ref_y, ":", color="grey", label="predicted slope = 1")

    ax.set_xlabel(r"spatial capacity $C_g$")
    ax.set_ylabel(r"$\lambda_{\max}(H_{\phi\phi})$")
    ax.set_title(r"Exp 1.1: $\lambda_\phi$ vs spatial capacity $C_g$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out1 = RESULTS_DIR / "exp1_1_lambda_vs_Cg.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"[exp1_1] wrote {out1}  (fit slope = {slope:.3f})")

    # --------------------------------------------------------------- #
    # Plot 2: kappa vs ratio (log-log) with y = x reference
    # --------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 5))
    ratio = df["ratio"].to_numpy(dtype=float)
    kappa = df["kappa"].to_numpy(dtype=float)

    ax.loglog(ratio, kappa, "o-", color="C2", label=r"measured $\kappa$")

    slope_k, intercept_k = _fit_loglog_slope(ratio, kappa)
    fit_x = np.array([ratio.min(), ratio.max()])
    fit_y = np.exp(intercept_k) * fit_x**slope_k
    ax.loglog(
        fit_x,
        fit_y,
        "--",
        color="C1",
        label=f"fit slope = {slope_k:.2f}",
    )

    # Reference line y = x (predicted slope = 1 through origin in log-log).
    ax.loglog(fit_x, fit_x, ":", color="grey", label="y = x (predicted)")

    ax.set_xlabel(r"$C_g / C_f$")
    ax.set_ylabel(r"$\kappa = \lambda_\phi / \lambda_\theta$")
    ax.set_title(r"Exp 1.1: curvature ratio vs capacity ratio")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out2 = RESULTS_DIR / "exp1_1_kappa_vs_ratio.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"[exp1_1] wrote {out2}  (fit slope = {slope_k:.3f})")


def main() -> None:
    df = run()
    print("\n=== Exp 1.1 results ===")
    print(df.to_string(index=False))
    plot(df)


if __name__ == "__main__":
    main()
