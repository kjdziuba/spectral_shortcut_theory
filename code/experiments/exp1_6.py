"""
Experiment 1.6 — Spectral Decoupling comparison.

Tests Pezeshki et al. 2021 spectral decoupling (L2 on pre-softmax logits)
as an alternative to freezing the spectral module.

Conditions:
  - joint:  jointly train spectral + spatial (baseline)
  - frozen: random-init spectral, requires_grad=False (our prescription)
  - sd:     joint + (lambda/2) * ||logits||^2 (Pezeshki et al. 2021, Eq. 8)

Sweep (120 runs at WIDTH=256, EPOCHS=100):
  joint  : 4 noise levels x 5 seeds              = 20
  frozen : 4 noise levels x 5 seeds              = 20
  sd     : 4 lambdas x 4 noise levels x 5 seeds  = 80

Sources held fixed for cross-experiment comparability:
  * NOISE_LEVELS, EPOCHS, batch size, train/test sizes copy exp1_4_fw.py
    so EGR window summaries (egr_early / egr_mid / egr_late / egr_depth)
    align with the existing within-capacity diagnostic.
  * weight_decay=0 across all conditions; Pezeshki says SD replaces WD,
    not stacks on top of it.

Notes:
  * For the SD condition, EGRLogger captures gradients of (CE + SD), not
    CE alone — intentional: it is the gradient ratio the optimiser
    actually follows. final_train_loss is CE-only, for cross-condition
    comparability.
  * In the frozen condition, EGRLogger.theta_norm is identically zero by
    construction (requires_grad=False set before logger snapshot). The
    summary CSV writes NaN to the EGR fields for frozen rows so that a
    naive groupby cannot silently pool zeros. The per-step EGR DataFrame
    is still saved under results/exp1_6/egr/ for inspection.
  * Cross-experiment comparability with exp1_4_fw_summary.csv: overfit_gap
    matches (peak_test_acc - final_test_acc); final_train_loss matches
    (last epoch per-batch CE mean); EGR window summaries match.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Make sibling 'synthetic' / 'egr' packages importable when run directly.
CODE_DIR = Path(__file__).resolve().parent.parent
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

# ----- Config (locked to exp1_4_fw for comparability) -------------------
TAG = "exp1_6"
S = 64
K = 16
H = 16
W = 16
N_CLASSES = 2
N_TRAIN = 512
N_TEST = 128
BATCH_SIZE = 32
EPOCHS = 100               # match exp1_4_fw (NOT exp1_2v4's 150)
LEARNING_RATE = 1e-3
WIDTH = 256                # fixed CNN width
WEIGHT_DECAY = 0.0         # SD replaces WD per Pezeshki et al. 2021

NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]
LAMBDAS_SD = [1e-3, 1e-2, 1e-1, 1.0]
SEEDS = [42, 43, 44, 45, 46]

# EGR aggregation windows (verbatim from exp1_4_fw).
EARLY_WINDOW = (50, 200)
MID_WINDOW = (400, 800)
LATE_WINDOW = (1200, 1600)
FINAL_LAST_K = 10


# ----- Losses -----------------------------------------------------------
def per_pixel_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pre-softmax logits -> per-pixel CE (mean reduction)."""
    C = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, C), y.reshape(-1), reduction="mean")


def spectral_decoupling_penalty(logits: torch.Tensor) -> torch.Tensor:
    """
    Pezeshki et al. 2021 Eq. 8: (1/2) * sum_C(logit^2), averaged over examples.

    For our (B, H, W, C) logits, each pixel is one example: sum over class
    dim, then mean over the flattened (B, H, W) pixel axis. The lambda
    multiplier is applied at the call site.

    The penalty is on raw pre-softmax logits — do NOT softmax/log_softmax
    before this term.
    """
    return 0.5 * logits.pow(2).sum(dim=-1).mean()


# ----- Eval -------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    logits = model(X.to(device))
    y_dev = y.to(device)
    loss = per_pixel_ce(logits, y_dev).item()
    preds = logits.argmax(dim=-1)
    acc = (preds == y_dev).float().mean().item()
    model.train()
    return loss, acc


# ----- Single run -------------------------------------------------------
def train_one(condition: str, noise: float, seed: int,
              device: torch.device, lambda_sd: float = 0.0):
    """
    condition: 'joint' | 'frozen' | 'sd'
    lambda_sd: penalty weight; ignored unless condition == 'sd'.
    Returns (summary_row_dict, egr_df, metrics_df).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialCNN(K=K, n_classes=N_CLASSES, width=WIDTH)
    model = CompositionModel(spectral, spatial).to(device)

    if condition == "frozen":
        for p in spectral.parameters():
            p.requires_grad = False

    # Data: train uses seed, test uses seed+1000 (canonical convention).
    X_tr, y_tr = make_problem(N_TRAIN, S=S, H=H, W=W,
                              n_classes=N_CLASSES,
                              alpha=1.0, beta=1.0,
                              noise=noise, seed=seed)
    X_te, y_te = make_problem(N_TEST, S=S, H=H, W=W,
                              n_classes=N_CLASSES,
                              alpha=1.0, beta=1.0,
                              noise=noise, seed=seed + 1000)
    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_te = X_te.to(device); y_te = y_te.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # EGRLogger snapshot must come AFTER the freeze flag is set.
    logger = EGRLogger(model)

    steps_per_epoch = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE
    metrics_rows = []
    global_step = 0
    for epoch in range(EPOCHS):
        perm = torch.randperm(N_TRAIN, device=device)
        # Collect per-batch training losses for the epoch so train_loss
        # matches exp1_4_fw's estimator exactly (per-step mean over an epoch).
        epoch_losses = []
        for s_idx in range(steps_per_epoch):
            i0 = s_idx * BATCH_SIZE
            i1 = min(i0 + BATCH_SIZE, N_TRAIN)
            ix = perm[i0:i1]
            xb = X_tr[ix]
            yb = y_tr[ix]

            opt.zero_grad()
            logits = model(xb)
            loss_ce = per_pixel_ce(logits, yb)
            if condition == "sd":
                loss = loss_ce + lambda_sd * spectral_decoupling_penalty(logits)
            else:
                loss = loss_ce
            loss.backward()
            # log_step MUST be after backward, before opt.step().
            logger.log_step(global_step)
            opt.step()
            # Track per-step CE only (not the regularised loss) so train_loss
            # is comparable across conditions and matches exp1_4_fw's CE-only
            # statistic. For SD, the regularised loss is what the optimiser
            # follows but the CE component is the apples-to-apples number.
            epoch_losses.append(loss_ce.item())
            global_step += 1

        train_loss = float(np.mean(epoch_losses))
        te_loss, te_acc = evaluate(model, X_te, y_te, device)
        # Cheap per-epoch train_acc for diagnostic plots; not used in summary.
        _, tr_acc = evaluate(model, X_tr, y_tr, device)
        metrics_rows.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": tr_acc,
            "test_loss": te_loss, "test_acc": te_acc,
        })

    egr_df = logger.to_dataframe()
    metrics_df = pd.DataFrame(metrics_rows)

    # EGR window summaries (must match exp1_4_fw exactly).
    def winmean(lo, hi):
        w = egr_df[(egr_df["step"] >= lo) & (egr_df["step"] <= hi)]
        return float(w["egr"].mean()) if not w.empty else float("nan")

    egr_early = winmean(*EARLY_WINDOW)
    egr_mid = winmean(*MID_WINDOW)
    egr_late = winmean(*LATE_WINDOW)
    if not egr_df.empty:
        egr_min = float(egr_df["egr"].rolling(window=20, min_periods=1).mean().min())
    else:
        egr_min = float("nan")
    if math.isnan(egr_early) or math.isnan(egr_min):
        egr_depth = float("nan")
    else:
        egr_depth = egr_early - egr_min

    # Match exp1_4_fw exactly: final_train_loss = LAST epoch's per-batch CE
    # mean (NOT a 10-epoch rolling avg of eval-mode train passes).
    final_train_loss = float(metrics_df["train_loss"].iloc[-1])
    final_test_loss = float(metrics_df["test_loss"].iloc[-FINAL_LAST_K:].mean())
    final_test_acc = float(metrics_df["test_acc"].iloc[-FINAL_LAST_K:].mean())
    peak_test_acc = float(metrics_df["test_acc"].max())
    # Match exp1_4_fw L173 exactly: overfit_gap is the TEMPORAL collapse
    # (peak test acc -> final test acc), not a train-test gap.
    overfit_gap = peak_test_acc - final_test_acc

    # Frozen runs have grad_theta_norm == 0 by construction (EGRLogger
    # filters non-requires_grad params at snapshot time). Emit NaN for the
    # EGR fields so a naive groupby cannot silently pool zeros into
    # 'frozen has shallowest EGR'. Per-step egr_df is still saved
    # alongside for inspection.
    if condition == "frozen":
        egr_early = float("nan")
        egr_mid = float("nan")
        egr_late = float("nan")
        egr_min = float("nan")
        egr_depth = float("nan")

    row = {
        "condition": condition,
        "noise": noise,
        "seed": seed,
        "lambda_sd": lambda_sd if condition == "sd" else float("nan"),
        "egr_early": egr_early,
        "egr_mid": egr_mid,
        "egr_late": egr_late,
        "egr_min": egr_min,
        "egr_depth": egr_depth,
        "final_test_acc": final_test_acc,
        "peak_test_acc": peak_test_acc,
        "overfit_gap": overfit_gap,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
    }
    return row, egr_df, metrics_df


# ----- Main -------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{TAG}] device = {device}")
    print(f"[{TAG}] WIDTH = {WIDTH} (fixed)  EPOCHS = {EPOCHS}")
    print(f"[{TAG}] NOISE_LEVELS = {NOISE_LEVELS}")
    print(f"[{TAG}] LAMBDAS_SD = {LAMBDAS_SD}")
    print(f"[{TAG}] SEEDS = {SEEDS}")

    runs = []
    for noise in NOISE_LEVELS:
        for seed in SEEDS:
            runs.append(("joint", noise, seed, 0.0))
            runs.append(("frozen", noise, seed, 0.0))
            for lam in LAMBDAS_SD:
                runs.append(("sd", noise, seed, lam))

    total = len(runs)
    print(f"[{TAG}] total runs = {total}  (~{total * 29 / 60:.0f} min at 29s/run)")

    egr_dir = RESULTS_DIR / TAG / "egr"
    metrics_dir = RESULTS_DIR / TAG / "metrics"
    egr_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, (cond, noise, seed, lam) in enumerate(runs, start=1):
        if cond == "sd":
            head = f"cond={cond} noise={noise} seed={seed} lam={lam:.0e}"
        else:
            head = f"cond={cond} noise={noise} seed={seed}"
        print(f"[{TAG}] [{i}/{total}] {head}", flush=True)
        row, egr_df, metrics_df = train_one(cond, noise, seed, device, lambda_sd=lam)
        summary_rows.append(row)
        run_name = f"{cond}_n{noise:.2f}_s{seed}_lam{lam:.0e}"
        egr_df.to_csv(egr_dir / f"{run_name}.csv", index=False)
        metrics_df.to_csv(metrics_dir / f"{run_name}.csv", index=False)
        print(f"          EGR_min={row['egr_min']:.3f}  "
              f"acc={row['final_test_acc']:.3f}  "
              f"gap={row['overfit_gap']:.3f}  "
              f"trL={row['final_train_loss']:.3f}",
              flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    csv_out = RESULTS_DIR / f"{TAG}_summary.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"\n[{TAG}] wrote {csv_out}")

    # Quick headline tables.
    print(f"\n=== {TAG} mean test acc by condition x noise ===")
    pivot = (summary_df
             .groupby(["condition", "noise"])["final_test_acc"]
             .agg(["mean", "std", "count"]))
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    sd_only = summary_df[summary_df["condition"] == "sd"]
    if not sd_only.empty:
        print(f"\n=== {TAG} SD: best lambda per noise (by mean test acc) ===")
        best_lam = (sd_only.groupby(["noise", "lambda_sd"])["final_test_acc"]
                    .mean().reset_index()
                    .sort_values(["noise", "final_test_acc"],
                                 ascending=[True, False])
                    .groupby("noise").head(1))
        print(best_lam.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
