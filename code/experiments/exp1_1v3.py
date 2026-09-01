"""
Experiment 1.1 v3: Capacity ablation on the TRUE GGN blocks (audit A24/A6/A64).

Differences from exp1_1.py (kept untouched for provenance):

  1. Measures lambda_max of the Gauss-Newton blocks G_thetatheta / G_phiphi
     (G = J^T H_tau J / N, softmax-CE), NOT the full-Hessian blocks. The old
     top_eigenvalue_block measured the loss Hessian, which differs by the
     uncontrolled functional-Hessian term at init (A24). Uses the validated
     GGNBlockOperator + lanczos_topk (full reorthogonalization, top-1).
  2. Reports and plots against the width M directly (A6), not only C_g.
  3. Adds a second architecture class, SpatialDeepMLP: fixed depth, hidden
     layers all M x M, so C_g = Theta(M^2). This tests the parameter-count
     corollary in its own capacity class (A64). SpatialMLP has C_g = Theta(M).
  4. The curvature ratio is branded D_curv in code and CSVs:
         D_curv = lambda_max(G_phiphi) / lambda_max(G_thetatheta).

VALIDATION GATE: before any sweep, GGNBlockOperator's lambda_max is
cross-checked on tiny configs (both architectures, both blocks) against a
brute-force densely materialized J^T H_tau J eigendecomposition; the script
asserts relative error < 1e-6 and aborts otherwise.

All measurements at random init, float64, on one fixed batch per seed.
Output: results/exp1_1_v3_ggn.csv (+ aggregated CSV + plots).

Usage:
    python exp1_1v3.py            # validation gate, then full sweep
    python exp1_1v3.py --smoke    # validation gate + 1 width/seed per arch,
                                  # writes exp1_1_v3_ggn_smoke.csv
    python exp1_1v3.py --init_mode theorem
                                  # theorem-matching init (He weights, fixed
                                  # sigma_b=0.5 biases) on the SPATIAL module
                                  # only; writes exp1_1_v3_ggn_theoreminit.csv

--init_mode default (the default) reproduces the original run bit-exact; the
spectral module's init is never touched in either mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from synthetic.data import make_problem  # noqa: E402
from synthetic.models import (  # noqa: E402
    SpectralReduction,
    SpatialMLP,
    SpatialDeepMLP,
    CompositionModel,
)
from hessian.ggn import _brute_force_ggn_block  # noqa: E402
from hessian.lanczos import GGNBlockOperator, lanczos_topk  # noqa: E402
from hessian.eigenvalues import count_params  # noqa: E402


RESULTS_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #
# Configuration (sweep skeleton matches exp1_1.py)
# --------------------------------------------------------------------- #
S = 64
K = 16
H = W = 16
N_CLASSES = 2
N_SAMPLES = 64
CLASS_DIM = 3            # models are channels-LAST: logits (B, H, W, C)

ARCH_WIDTHS = {
    # SpatialMLP: one hidden layer of width M       -> C_g = Theta(M)
    "mlp": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    # SpatialDeepMLP: 3 hidden layers, all M x M    -> C_g = Theta(M^2)
    "deep_mlp": [16, 32, 64, 128, 256, 512, 1024],
}
SEEDS = [42, 43, 44, 45, 46]
LANCZOS_M = 80           # Lanczos steps (top-1 needed; generous margin)
VALIDATION_TOL = 1e-6


def build_spatial(arch: str, M: int, init_mode: str = "default") -> torch.nn.Module:
    if arch == "mlp":
        return SpatialMLP(K=K, n_classes=N_CLASSES, width=M, init_mode=init_mode)
    if arch == "deep_mlp":
        return SpatialDeepMLP(K=K, n_classes=N_CLASSES, width=M, n_hidden=3,
                              init_mode=init_mode)
    raise ValueError(arch)


def _top1_ggn(model, X, y, block_module, seed: int) -> float:
    """lambda_max of the GGN block for `block_module`'s params via Lanczos."""
    params = [p for p in block_module.parameters() if p.requires_grad]
    op = GGNBlockOperator(model, X, y, params, class_dim=CLASS_DIM)
    ritz_vals, _, info = lanczos_topk(op, m=LANCZOS_M, k=1, seed=seed, n_vectors=1)
    lam = float(ritz_vals[0].item())
    # Residual check: ||G u - lam u|| must be tiny relative to lam.
    res = info["residuals"][0] if info["residuals"] else float("nan")
    if not (res <= 1e-6 * max(abs(lam), 1e-30)):
        print(f"    WARNING: Lanczos residual {res:.3e} vs lambda {lam:.3e} "
              f"(iters={info['iters']})")
    return lam


# --------------------------------------------------------------------- #
# Validation gate (A24): operator vs brute-force dense GGN
# --------------------------------------------------------------------- #
def validate(device: torch.device, init_mode: str = "default") -> float:
    """Cross-check Lanczos-on-operator lambda_max against a dense J^T H_tau J
    eigendecomposition on tiny configs. Returns the worst relative error.
    Asserts < VALIDATION_TOL."""
    print(f"[exp1_1v3] === VALIDATION GATE: operator vs brute-force dense GGN "
          f"(init_mode={init_mode}) ===")
    Hs = Ws = 4          # tiny spatial grid so the explicit J is tractable
    Bs = 4
    worst = 0.0
    for arch in ("mlp", "deep_mlp"):
        torch.manual_seed(0)
        spectral = SpectralReduction(S=S, K=K)
        spatial = build_spatial(arch, 16, init_mode=init_mode)
        model = CompositionModel(spectral, spatial).to(device).double()
        X, y = make_problem(n_samples=Bs, S=S, H=Hs, W=Ws,
                            n_classes=N_CLASSES, alpha=1.0, beta=1.0, seed=0)
        X = X.to(device).double()
        y = y.to(device)
        for block_name in ("spectral", "spatial"):
            block = getattr(model, block_name)
            params = [p for p in block.parameters() if p.requires_grad]
            lam_op = _top1_ggn(model, X, y, block, seed=0)
            G = _brute_force_ggn_block(model, X, y, params, class_dim=CLASS_DIM)
            G = 0.5 * (G + G.T)
            evals = torch.linalg.eigvalsh(G)
            lam_dense = float(evals[-1].item())
            assert float(evals[0].item()) > -1e-10, \
                f"{arch}/{block_name}: dense GGN not PSD (min eig {evals[0]:.3e})"
            rel = abs(lam_op - lam_dense) / max(abs(lam_dense), 1e-30)
            worst = max(worst, rel)
            print(f"[exp1_1v3]   {arch:8s} {block_name:8s} "
                  f"lanczos={lam_op:.12e}  dense={lam_dense:.12e}  rel.err={rel:.2e}")
            assert rel < VALIDATION_TOL, \
                f"{arch}/{block_name}: rel err {rel:.3e} >= {VALIDATION_TOL}"
        del model, spectral, spatial, X, y
    print(f"[exp1_1v3] VALIDATION PASSED  (worst rel.err = {worst:.2e} < {VALIDATION_TOL})\n")
    return worst


# --------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------- #
def measure_one(arch: str, M: int, seed: int, device: torch.device,
                init_mode: str = "default") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spectral = SpectralReduction(S=S, K=K)
    spatial = build_spatial(arch, M, init_mode=init_mode)
    model = CompositionModel(spectral, spatial).to(device).double()

    X, y = make_problem(n_samples=N_SAMPLES, S=S, H=H, W=W,
                        n_classes=N_CLASSES, alpha=1.0, beta=1.0, seed=seed)
    X = X.to(device).double()
    y = y.to(device)

    lambda_theta = _top1_ggn(model, X, y, model.spectral, seed=seed)
    lambda_phi = _top1_ggn(model, X, y, model.spatial, seed=seed)

    C_f = count_params(spectral)
    C_g = count_params(spatial)
    d_curv = lambda_phi / lambda_theta if lambda_theta > 0 else float("nan")

    del model, spectral, spatial, X, y
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "arch": arch,
        "M": M,
        "seed": seed,
        "init_mode": init_mode,
        "C_f": C_f,
        "C_g": C_g,
        "lambda_theta_ggn": float(lambda_theta),
        "lambda_phi_ggn": float(lambda_phi),
        "D_curv": float(d_curv),
    }


def _fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    slope, _ = np.polyfit(np.log(x), np.log(y), 1)
    return float(slope)


def run(smoke: bool = False,
        init_mode: str = "default") -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_1v3] device = {device}")
    print(f"[exp1_1v3] init_mode = {init_mode}")

    validate(device, init_mode=init_mode)

    arch_widths = {a: ws[:1] for a, ws in ARCH_WIDTHS.items()} if smoke else ARCH_WIDTHS
    seeds = SEEDS[:1] if smoke else SEEDS
    print(f"[exp1_1v3] arch_widths = {arch_widths}")
    print(f"[exp1_1v3] seeds = {seeds}\n")

    rows: list[dict] = []
    t0 = time.time()
    for arch, widths in arch_widths.items():
        for M in widths:
            for seed in seeds:
                t = time.time()
                row = measure_one(arch, M, seed, device, init_mode=init_mode)
                rows.append(row)
                print(f"[exp1_1v3] {arch:8s} M={M:5d} seed={seed}  "
                      f"C_g={row['C_g']:9d}  "
                      f"l_theta={row['lambda_theta_ggn']:.4e}  "
                      f"l_phi={row['lambda_phi_ggn']:.4e}  "
                      f"D_curv={row['D_curv']:.4e}  ({time.time()-t:.1f}s)",
                      flush=True)
    print(f"[exp1_1v3] sweep wall time: {time.time()-t0:.1f}s")

    raw_df = pd.DataFrame(rows)
    tag = "_theoreminit" if init_mode == "theorem" else ""
    suffix = "_smoke" if smoke else ""
    raw_path = RESULTS_DIR / f"exp1_1_v3_ggn{tag}{suffix}.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"[exp1_1v3] wrote {raw_path}")

    agg_df = raw_df.groupby(["arch", "M"], sort=True).agg(
        C_f=("C_f", "first"),
        C_g=("C_g", "first"),
        lambda_theta_ggn_mean=("lambda_theta_ggn", "mean"),
        lambda_theta_ggn_std=("lambda_theta_ggn", "std"),
        lambda_phi_ggn_mean=("lambda_phi_ggn", "mean"),
        lambda_phi_ggn_std=("lambda_phi_ggn", "std"),
        D_curv_mean=("D_curv", "mean"),
        D_curv_std=("D_curv", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    agg_df.insert(2, "init_mode", init_mode)
    agg_path = RESULTS_DIR / f"exp1_1_v3_ggn{tag}_aggregated{suffix}.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"[exp1_1v3] wrote {agg_path}")
    return raw_df, agg_df


def plot(agg_df: pd.DataFrame, raw_df: pd.DataFrame, tag: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"mlp": ("C0", "o"), "deep_mlp": ("C3", "s")}
    labels = {
        "mlp": r"SpatialMLP  ($C_g = \Theta(M)$)",
        "deep_mlp": r"SpatialDeepMLP  ($C_g = \Theta(M^2)$)",
    }

    # ---- Plot 1: lambda_phi_ggn vs width M (A6: plot vs M directly) ---- #
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch, (color, marker) in styles.items():
        sub = agg_df[agg_df["arch"] == arch]
        if sub.empty:
            continue
        Ms = sub["M"].to_numpy(float)
        mean = sub["lambda_phi_ggn_mean"].to_numpy(float)
        std = sub["lambda_phi_ggn_std"].to_numpy(float)
        slope = _fit_loglog_slope(Ms, mean)
        rsub = raw_df[raw_df["arch"] == arch]
        ax.scatter(rsub["M"], rsub["lambda_phi_ggn"], s=15, color=color, alpha=0.3)
        ax.plot(Ms, mean, marker + "-", color=color, lw=2,
                label=f"{labels[arch]}, slope={slope:.2f}")
        ax.fill_between(Ms, np.clip(mean - std, 1e-30, None), mean + std,
                        color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("width $M$")
    ax.set_ylabel(r"$\lambda_{\max}(G_{\phi\phi})$  (GGN, mean over seeds)")
    ax.set_title("Exp 1.1 v3: spatial GGN top eigenvalue vs width")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_1_v3_lambda_phi_vs_M{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_1v3] wrote {out}")

    # ---- Plot 2: D_curv vs width M ------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch, (color, marker) in styles.items():
        sub = agg_df[agg_df["arch"] == arch]
        if sub.empty:
            continue
        Ms = sub["M"].to_numpy(float)
        mean = sub["D_curv_mean"].to_numpy(float)
        std = sub["D_curv_std"].to_numpy(float)
        slope = _fit_loglog_slope(Ms, mean)
        rsub = raw_df[raw_df["arch"] == arch]
        ax.scatter(rsub["M"], rsub["D_curv"], s=15, color=color, alpha=0.3)
        ax.plot(Ms, mean, marker + "-", color=color, lw=2,
                label=f"{labels[arch]}, slope={slope:.2f}")
        ax.fill_between(Ms, np.clip(mean - std, 1e-30, None), mean + std,
                        color=color, alpha=0.15)
    # slope-1 reference through the first deep_mlp point
    ref = agg_df[agg_df["arch"] == "deep_mlp"]
    if not ref.empty:
        M0 = float(ref["M"].iloc[0])
        y0 = float(ref["D_curv_mean"].iloc[0])
        Ms_all = np.array([agg_df["M"].min(), agg_df["M"].max()], dtype=float)
        ax.plot(Ms_all, y0 * (Ms_all / M0), ":", color="grey", lw=2,
                label="slope = 1 reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("width $M$")
    ax.set_ylabel(r"$D_{\mathrm{curv}} = \lambda_{\max}(G_{\phi\phi})\,/\,\lambda_{\max}(G_{\theta\theta})$")
    ax.set_title("Exp 1.1 v3: curvature ratio $D_{\\mathrm{curv}}$ vs width")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_1_v3_Dcurv_vs_M{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_1v3] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="validation gate + 1 width/seed per arch")
    parser.add_argument("--init_mode", choices=["default", "theorem"],
                        default="default",
                        help="spatial-module init: 'default' (PyTorch nn.Linear, "
                             "reproduces original run) or 'theorem' (He weights, "
                             "fixed sigma_b=0.5 biases)")
    args = parser.parse_args()

    raw_df, agg_df = run(smoke=args.smoke, init_mode=args.init_mode)

    print(f"\n=== Exp 1.1 v3 (GGN blocks, init_mode={args.init_mode}) — "
          f"mean over seeds ===")
    cols = ["arch", "M", "init_mode", "C_f", "C_g",
            "lambda_theta_ggn_mean", "lambda_phi_ggn_mean",
            "D_curv_mean", "D_curv_std", "n_seeds"]
    print(agg_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    if not args.smoke:
        for arch in agg_df["arch"].unique():
            sub = agg_df[agg_df["arch"] == arch]
            s_lphi = _fit_loglog_slope(sub["M"].to_numpy(float),
                                       sub["lambda_phi_ggn_mean"].to_numpy(float))
            s_dcurv = _fit_loglog_slope(sub["M"].to_numpy(float),
                                        sub["D_curv_mean"].to_numpy(float))
            s_cg = _fit_loglog_slope(sub["M"].to_numpy(float),
                                     sub["C_g"].to_numpy(float))
            print(f"[exp1_1v3] {arch:8s} log-log slopes vs M:  "
                  f"C_g={s_cg:.3f}  lambda_phi_ggn={s_lphi:.3f}  D_curv={s_dcurv:.3f}")
        plot(agg_df, raw_df,
             tag="_theoreminit" if args.init_mode == "theorem" else "")


if __name__ == "__main__":
    main()
