"""
Experiment 1.3: Equal-Information Killer Test.

Test the central conjecture of Theorem 2: when the spectral and spatial
pathways carry equal information about the label, end-to-end joint
training will preferentially use the spatial pathway (a shortcut),
under-utilizing the spectral signal.

For each of three calibration modes -- 'bayes', 'ntk', 'margin' -- we
find scalar (alpha, beta) such that the two modalities are "equal" in
the sense of that mode. We then train a CNN on the full data under four
conditions:

  joint:         end-to-end training with both modules trainable
  frozen:        spectral module frozen at init
  spectral_only: data with beta=0 (only spectral signal present)
  spatial_only:  data with alpha=0 (only spatial signal present)

Comparison: a clean spectral-shortcut effect manifests as
    spectral_only_acc ~= spatial_only_acc        (calibration succeeded)
AND joint_acc < frozen_acc                       (shortcut emerged in joint)

We then pick the calibration mode that maximizes both criteria.
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
    SpectralReduction, SpatialCNN, CompositionModel,
)
from synthetic.calibrate import (  # noqa: E402
    calibrate_bayes, calibrate_ntk, calibrate_margin, CalibResult,
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
NOISE = 0.10
SPATIAL_WIDTH = 256          # CNN width D — calibrated capacity from v3/v4
SEEDS = [42, 43, 44]
CONDITIONS = ["joint", "frozen", "spectral_only", "spatial_only"]

CALIBRATION_MODES = ["bayes", "ntk", "margin"]
TAG = "v1"


@dataclass
class RunResult:
    mode: str
    condition: str
    seed: int
    alpha: float
    beta: float
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


def get_alpha_beta(mode: str, condition: str, base_alpha: float, base_beta: float) -> tuple[float, float]:
    """Map (condition, base alpha, base beta) -> the (alpha, beta) to use for data gen."""
    if condition == "joint" or condition == "frozen":
        return base_alpha, base_beta
    if condition == "spectral_only":
        return base_alpha, 0.0
    if condition == "spatial_only":
        return 0.0, base_beta
    raise ValueError(condition)


def train_one(mode: str, condition: str, seed: int,
              base_alpha: float, base_beta: float,
              device: torch.device) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spectral = SpectralReduction(S=S, K=K)
    spatial = SpatialCNN(K=K, n_classes=N_CLASSES, width=SPATIAL_WIDTH)
    model = CompositionModel(spectral, spatial).to(device)

    if condition == "frozen":
        for p in spectral.parameters():
            p.requires_grad = False

    a, b = get_alpha_beta(mode, condition, base_alpha, base_beta)
    X_tr, y_tr = make_problem(N_TRAIN, S=S, H=H, W=W, n_classes=N_CLASSES,
                              alpha=a, beta=b, noise=NOISE, seed=seed)
    X_te, y_te = make_problem(N_TEST, S=S, H=H, W=W, n_classes=N_CLASSES,
                              alpha=a, beta=b, noise=NOISE, seed=seed + 1000)
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
        mode=mode, condition=condition, seed=seed,
        alpha=base_alpha, beta=base_beta,
        egr_df=egr_df, metrics_df=metrics_df,
        final_train_loss=final["train_loss"],
        final_test_loss=final["test_loss"],
        final_test_acc=final["test_acc"],
        peak_test_acc=max(r["test_acc"] for r in metrics_rows),
    )


def plot_mode(results: list[RunResult], mode: str, calib: CalibResult) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in results if r.mode == mode]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (A) test accuracy curves per condition
    ax = axes[0]
    colors = {"joint": "C3", "frozen": "C0", "spectral_only": "C2", "spatial_only": "C4"}
    for cond in CONDITIONS:
        cond_runs = [r for r in runs if r.condition == cond]
        if not cond_runs:
            continue
        accs = np.stack([r.metrics_df["test_acc"].values for r in cond_runs])
        mean = accs.mean(0); std = accs.std(0)
        epochs = np.arange(len(mean))
        ax.plot(epochs, mean, color=colors[cond], lw=2, label=cond)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=colors[cond], alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (mean ± 1σ)")
    ax.set_title(f"Test acc by condition  (CNN D={SPATIAL_WIDTH})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # (B) final-epoch bar chart of accuracies
    ax = axes[1]
    labels = []
    means = []
    stds = []
    cs = []
    for cond in CONDITIONS:
        cond_runs = [r for r in runs if r.condition == cond]
        if not cond_runs:
            continue
        finals = [r.final_test_acc for r in cond_runs]
        labels.append(cond)
        means.append(float(np.mean(finals)))
        stds.append(float(np.std(finals)))
        cs.append(colors[cond])
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, color=cs, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("final test accuracy")
    ax.set_title("final test accuracy per condition")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Exp 1.3 ({mode} calibration): alpha={calib.alpha:.3f}, beta={calib.beta:.3f}  "
        f"(spec-only Bayes={calib.spectral_only_acc:.3f}, "
        f"spat-only Bayes={calib.spatial_only_acc:.3f})"
    )
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_3{TAG}_{mode}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_3 {TAG}] wrote {out}")


def plot_summary(results: list[RunResult], calibs: dict) -> None:
    """Side-by-side bar charts: for each calibration mode, the 4 final accuracies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(calibs.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 5),
                              sharey=True)
    if len(modes) == 1:
        axes = [axes]

    colors = {"joint": "C3", "frozen": "C0", "spectral_only": "C2", "spatial_only": "C4"}
    for ax, mode in zip(axes, modes):
        runs = [r for r in results if r.mode == mode]
        labels = []
        means = []
        stds = []
        cs = []
        for cond in CONDITIONS:
            cond_runs = [r for r in runs if r.condition == cond]
            if not cond_runs:
                continue
            finals = [r.final_test_acc for r in cond_runs]
            labels.append(cond)
            means.append(float(np.mean(finals)))
            stds.append(float(np.std(finals)))
            cs.append(colors[cond])
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, color=cs, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(
            f"{mode}\nα={calibs[mode].alpha:.2f}, β={calibs[mode].beta:.2f}\n"
            f"spec-only={calibs[mode].spectral_only_acc:.2f}, "
            f"spat-only={calibs[mode].spatial_only_acc:.2f}"
        )
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("final test accuracy")
    fig.suptitle("Exp 1.3: equal-information killer test across calibration modes")
    fig.tight_layout()
    out = RESULTS_DIR / f"exp1_3{TAG}_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp1_3 {TAG}] wrote {out}")


def score_calibration(calib: CalibResult, results: list[RunResult]) -> dict:
    """Compute the 'is this calibration good' score.

    Two criteria:
      calibration_quality = 1 - |spectral_acc - spatial_acc|   (closer to 1 is better)
      shortcut_strength   = frozen_acc - joint_acc             (more positive is better)

    Combined score = calibration_quality + 2 * shortcut_strength.
    """
    cq = 1.0 - abs(calib.spectral_only_acc - calib.spatial_only_acc)
    joint_runs = [r.final_test_acc for r in results
                  if r.mode == calib.mode and r.condition == "joint"]
    frozen_runs = [r.final_test_acc for r in results
                   if r.mode == calib.mode and r.condition == "frozen"]
    j = float(np.mean(joint_runs)) if joint_runs else float("nan")
    fr = float(np.mean(frozen_runs)) if frozen_runs else float("nan")
    ss = fr - j
    return {
        "mode": calib.mode,
        "calibration_quality": cq,
        "shortcut_strength": ss,
        "combined_score": cq + 2 * ss,
        "joint_acc": j,
        "frozen_acc": fr,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1_3 {TAG}] device = {device}")
    print(f"[exp1_3 {TAG}] spatial CNN width = {SPATIAL_WIDTH}")
    print()

    # ---- step 1: calibrate ---- #
    calibs: dict[str, CalibResult] = {}
    print("[exp1_3] calibrating all three modes (CPU)...")
    calibs["bayes"]  = calibrate_bayes(target_acc=0.75)
    calibs["ntk"]    = calibrate_ntk(target_eig=1.0)
    calibs["margin"] = calibrate_margin(target_margin=0.10)
    for m, c in calibs.items():
        print(f"  {m}: alpha={c.alpha:.4f}  beta={c.beta:.4f}  "
              f"spec-only={c.spectral_only_acc:.3f}  "
              f"spat-only={c.spatial_only_acc:.3f}")
    print()

    # ---- step 2: run experiments ---- #
    all_results: list[RunResult] = []
    summary_rows = []
    total = len(CALIBRATION_MODES) * len(CONDITIONS) * len(SEEDS)
    i = 0
    for mode in CALIBRATION_MODES:
        c = calibs[mode]
        for cond in CONDITIONS:
            for seed in SEEDS:
                i += 1
                print(f"[exp1_3 {TAG}] [{i}/{total}] mode={mode} cond={cond} seed={seed} ...",
                      flush=True)
                r = train_one(mode, cond, seed, c.alpha, c.beta, device)
                all_results.append(r)
                summary_rows.append({
                    "mode": mode, "condition": cond, "seed": seed,
                    "alpha": r.alpha, "beta": r.beta,
                    "final_train_loss": r.final_train_loss,
                    "final_test_loss": r.final_test_loss,
                    "final_test_acc": r.final_test_acc,
                    "peak_test_acc": r.peak_test_acc,
                })
                print(f"           train={r.final_train_loss:.3f} "
                      f"test={r.final_test_loss:.3f} "
                      f"acc={r.final_test_acc:.3f} "
                      f"peak={r.peak_test_acc:.3f}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / f"exp1_3{TAG}_summary.csv", index=False)

    # ---- step 3: plot ---- #
    for mode in CALIBRATION_MODES:
        plot_mode(all_results, mode, calibs[mode])
    plot_summary(all_results, calibs)

    # ---- step 4: score and pick winner ---- #
    print()
    print("=== Calibration mode scores ===")
    scores = [score_calibration(calibs[m], all_results) for m in CALIBRATION_MODES]
    score_df = pd.DataFrame(scores).sort_values("combined_score", ascending=False)
    print(score_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    score_df.to_csv(RESULTS_DIR / f"exp1_3{TAG}_scores.csv", index=False)
    print(f"\nWINNER: {score_df.iloc[0]['mode']}  "
          f"(combined score = {score_df.iloc[0]['combined_score']:.4f})")


if __name__ == "__main__":
    main()
