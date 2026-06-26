"""
Experiment 1.1: Capacity Ablation (multi-seed, extended capacity).

Validates Theorem 1: lambda_max(H_phi_phi) grows with C_g, the spatial
(downstream) capacity. We hold the spectral reduction fixed (S=64, K=16)
and sweep the width D of the spatial MLP, measuring:

  - C_f: spectral parameter count (constant across sweep)
  - C_g: spatial parameter count (grows with D)
  - lambda_theta = lambda_max(H_theta_theta) (spectral Hessian block)
  - lambda_phi   = lambda_max(H_phi_phi)     (spatial Hessian block)
  - kappa        = lambda_phi / lambda_theta (curvature ratio)

This version runs N_SEEDS per width and reports mean +/- std, so we can
distinguish a real scaling slope from initialization noise. Widths extend
to D = 8192 to probe whether the empirical scaling slope approaches the
predicted asymptote of 1 at larger widths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
S = 64
K = 16
H = W = 16
N_CLASSES = 2
N_SAMPLES = 64
CAPACITIES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SEEDS = [42, 43, 44, 45, 46]
N_ITER_POWER = 20


def measure_one(D: int, seed: int, device: torch.device) -> dict:
    """Build a model at width D, seed it, measure block eigenvalues."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialMLP(K=K, n_classes=N_CLASSES, width=D)
    model = CompositionModel(spectral, spatial).to(device)

    X, y = make_problem(
        n_samples=N_SAMPLES,
        S=S,
        H=H,
        W=W,
        n_classes=N_CLASSES,
        alpha=1.0,
        beta=1.0,
        seed=seed,
    )
    X = X.to(device)
    y = y.to(device)

    lambda_theta = top_eigenvalue_block(
        model, X, y, block="spectral", n_iter=N_ITER_POWER, device=device,
    )
    lambda_phi = top_eigenvalue_block(
        model, X, y, block="spatial", n_iter=N_ITER_POWER, device=device,
    )

    C_f = count_params(spectral)
    C_g = count_params(spatial)
    kappa = lambda_phi / lambda_theta if lambda_theta > 0 else float("nan")

    # Free the GPU resources before the next run.
    del model, spectral, spatial, X, y
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "D": D,
        "seed": seed,
        "C_f": C_f,
        "C_g": C_g,
        "ratio": C_g / C_f,
        "lambda_theta": float(lambda_theta),
        "lambda_phi": float(lambda_phi),
        "kappa": float(kappa),
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (raw_df, agg_df) where raw_df has one row per (D, seed) and
    agg_df has one row per D with mean and std summaries."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_1] device = {device}")
    print(f"[exp1_1] widths: {CAPACITIES}")
    print(f"[exp1_1] seeds:  {SEEDS}")
    print()

    rows: list[dict] = []
    for D in CAPACITIES:
        for seed in SEEDS:
            row = measure_one(D, seed, device)
            rows.append(row)
            print(
                f"[exp1_1] D={D:5d}  seed={seed}  "
                f"C_g={row['C_g']:8d}  ratio={row['ratio']:8.3f}  "
                f"l_theta={row['lambda_theta']:.3e}  "
                f"l_phi={row['lambda_phi']:.3e}  kappa={row['kappa']:.3e}"
            )

    raw_df = pd.DataFrame(rows)
    raw_path = RESULTS_DIR / "exp1_1_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n[exp1_1] wrote {raw_path}")

    # ---- aggregate over seeds for each width ----------------------- #
    agg_df = raw_df.groupby("D", sort=True).agg(
        C_f=("C_f", "first"),
        C_g=("C_g", "first"),
        ratio=("ratio", "first"),
        lambda_theta_mean=("lambda_theta", "mean"),
        lambda_theta_std=("lambda_theta", "std"),
        lambda_phi_mean=("lambda_phi", "mean"),
        lambda_phi_std=("lambda_phi", "std"),
        kappa_mean=("kappa", "mean"),
        kappa_std=("kappa", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    agg_path = RESULTS_DIR / "exp1_1_aggregated.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"[exp1_1] wrote {agg_path}")

    return raw_df, agg_df


def _fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of log y = slope * log x + intercept."""
    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def plot(agg_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Cg = agg_df["C_g"].to_numpy(dtype=float)
    Cf = agg_df["C_f"].to_numpy(dtype=float)
    ratio = agg_df["ratio"].to_numpy(dtype=float)
    lphi_mean = agg_df["lambda_phi_mean"].to_numpy(dtype=float)
    lphi_std = agg_df["lambda_phi_std"].to_numpy(dtype=float)
    kappa_mean = agg_df["kappa_mean"].to_numpy(dtype=float)
    kappa_std = agg_df["kappa_std"].to_numpy(dtype=float)

    # --------------------------------------------------------------- #
    # Plot 1: lambda_phi vs C_g (log-log) with shaded +/- 1 std
    # --------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))

    # All seeds as faint dots
    ax.scatter(
        raw_df["C_g"], raw_df["lambda_phi"],
        s=20, color="C0", alpha=0.3, label="individual seeds",
    )

    # Mean line
    ax.plot(
        Cg, lphi_mean, "o-",
        color="C0", lw=2, label=r"mean $\lambda_\phi$",
    )

    # +/- 1 std shaded band (in linear, before going to log-log)
    ax.fill_between(
        Cg,
        np.clip(lphi_mean - lphi_std, 1e-12, None),
        lphi_mean + lphi_std,
        color="C0", alpha=0.18, label=r"$\pm 1\sigma$ across seeds",
    )

    # Fit slope
    slope, intercept = _fit_loglog_slope(Cg, lphi_mean)
    fit_x = np.array([Cg.min(), Cg.max()])
    fit_y = np.exp(intercept) * fit_x**slope
    ax.plot(fit_x, fit_y, "--", color="C1", lw=2,
            label=f"fit slope = {slope:.2f}")

    # Reference line slope = 1 (predicted)
    ref_y = lphi_mean[0] * (fit_x / Cg[0])
    ax.plot(fit_x, ref_y, ":", color="grey", lw=2,
            label="predicted slope = 1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"spatial parameter count $C_g$")
    ax.set_ylabel(r"$\lambda_{\max}(H_{\phi\phi})$ (mean over 5 seeds)")
    ax.set_title(
        r"Exp 1.1: spatial Hessian top eigenvalue vs capacity"
        + f"\n(widths D=16..8192, fit slope = {slope:.2f})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out1 = RESULTS_DIR / "exp1_1_lambda_vs_Cg.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"[exp1_1] wrote {out1}  (slope = {slope:.3f})")

    # --------------------------------------------------------------- #
    # Plot 2: kappa vs ratio (log-log)
    # --------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        raw_df["ratio"], raw_df["kappa"],
        s=20, color="C2", alpha=0.3, label="individual seeds",
    )

    ax.plot(
        ratio, kappa_mean, "o-",
        color="C2", lw=2, label=r"mean $\kappa$",
    )
    ax.fill_between(
        ratio,
        np.clip(kappa_mean - kappa_std, 1e-12, None),
        kappa_mean + kappa_std,
        color="C2", alpha=0.18, label=r"$\pm 1\sigma$",
    )

    slope_k, intercept_k = _fit_loglog_slope(ratio, kappa_mean)
    fit_x = np.array([ratio.min(), ratio.max()])
    fit_y = np.exp(intercept_k) * fit_x**slope_k
    ax.plot(fit_x, fit_y, "--", color="C1", lw=2,
            label=f"fit slope = {slope_k:.2f}")

    ax.plot(fit_x, fit_x, ":", color="grey", lw=2,
            label="y = x (predicted)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"capacity ratio $C_g / C_f$")
    ax.set_ylabel(r"$\kappa = \lambda_\phi / \lambda_\theta$")
    ax.set_title(
        r"Exp 1.1: condition number across modules vs capacity ratio"
        + f"\n(widths D=16..8192, fit slope = {slope_k:.2f})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out2 = RESULTS_DIR / "exp1_1_kappa_vs_ratio.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"[exp1_1] wrote {out2}  (slope = {slope_k:.3f})")

    # --------------------------------------------------------------- #
    # Plot 3: lambda_theta and lambda_phi together (paired view)
    # --------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))

    lth_mean = agg_df["lambda_theta_mean"].to_numpy(dtype=float)
    lth_std = agg_df["lambda_theta_std"].to_numpy(dtype=float)

    ax.plot(Cg, lphi_mean, "o-", color="C0", lw=2,
            label=r"$\lambda_\phi$ (spatial)")
    ax.fill_between(
        Cg,
        np.clip(lphi_mean - lphi_std, 1e-12, None),
        lphi_mean + lphi_std,
        color="C0", alpha=0.18,
    )

    ax.plot(Cg, lth_mean, "s-", color="C3", lw=2,
            label=r"$\lambda_\theta$ (spectral)")
    ax.fill_between(
        Cg,
        np.clip(lth_mean - lth_std, 1e-12, None),
        lth_mean + lth_std,
        color="C3", alpha=0.18,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"spatial parameter count $C_g$")
    ax.set_ylabel("top block Hessian eigenvalue")
    ax.set_title(r"Exp 1.1: paired view of spectral and spatial curvature")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out3 = RESULTS_DIR / "exp1_1_paired_eigenvalues.png"
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"[exp1_1] wrote {out3}")


def main() -> None:
    raw_df, agg_df = run()

    print("\n=== Exp 1.1 results (aggregated over seeds) ===")
    cols = [
        "D", "C_f", "C_g", "ratio",
        "lambda_theta_mean", "lambda_phi_mean",
        "kappa_mean", "kappa_std", "n_seeds",
    ]
    print(agg_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    plot(agg_df, raw_df)


if __name__ == "__main__":
    main()
