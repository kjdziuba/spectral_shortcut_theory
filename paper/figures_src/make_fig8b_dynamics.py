"""Fig-8B: controlled retraining dynamics (E3c, breast fold 0).

Panel (a): held-out val macro-F1 per arm x width under AdamW (mean +/- sd
over 3 seeds; bars). Frozen arms beat joint arms at both widths.

Panel (b): the SGD pair at M=48 (theorems' regime): per-seed points +
mean bar — joint and frozen-random are statistically indistinguishable,
the theorem's functional-equivalence prediction.

Input:   results/e3c_analysis_runs.csv
Outputs: paper/figures/fig8b_dynamics.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

ARMS = ["joint_linear", "joint_mlp", "frozen_random", "frozen_pca"]
LABELS = {"joint_linear": "joint\n(linear)", "joint_mlp": "joint\n(MLP)",
          "frozen_random": "frozen\nrandom", "frozen_pca": "frozen\nPCA"}
COLORS = {"joint_linear": "#d95f02", "joint_mlp": "#e7298a",
          "frozen_random": "#7570b3", "frozen_pca": "#1b9e77"}

df = pd.read_csv(ROOT / "results" / "e3c_analysis_runs.csv")

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(5.5, 2.4), gridspec_kw={"width_ratios": [1.7, 1]})

# --------------------------------------------------------------- panel (a)
adamw = df[df.optimizer == "adamw"]
widths = [48, 192]
bar_w = 0.19
xg = np.arange(len(widths))
for i, arm in enumerate(ARMS):
    sub = adamw[adamw.arm == arm]
    means = [sub[sub.width == w].val_f1.mean() for w in widths]
    sds = [sub[sub.width == w].val_f1.std() for w in widths]
    x = xg + (i - 1.5) * bar_w
    ax1.bar(x, means, bar_w * 0.92, yerr=sds, capsize=2,
            color=COLORS[arm], label=LABELS[arm].replace("\n", " "),
            error_kw=dict(lw=0.8))
    for w, xx in zip(widths, x):
        pts = adamw[(adamw.arm == arm) & (adamw.width == w)].val_f1
        ax1.scatter(np.full(len(pts), xx), pts, s=6, color="k", alpha=0.55,
                    zorder=3)

ax1.set_xticks(xg)
ax1.set_xticklabels([f"$M={w}$" for w in widths])
ax1.set_ylabel("val macro-F1")
ax1.set_ylim(0.3, 0.82)
ax1.legend(frameon=False, ncol=2, loc="upper left",
           columnspacing=0.9, handlelength=1.2)
ax1.set_title("(a) AdamW: frozen beats joint", loc="left")

# --------------------------------------------------------------- panel (b)
sgd = df[df.optimizer == "sgd"]
for i, arm in enumerate(["joint_linear", "frozen_random"]):
    pts = sgd[sgd.arm == arm].val_f1
    ax2.bar(i, pts.mean(), 0.55, yerr=pts.std(), capsize=3,
            color=COLORS[arm], alpha=0.9, error_kw=dict(lw=0.8))
    ax2.scatter(np.full(len(pts), i), pts, s=14, color="k", alpha=0.6,
                zorder=3)

ax2.set_xticks([0, 1])
ax2.set_xticklabels(["joint\n(linear)", "frozen\nrandom"])
ax2.set_ylabel("val macro-F1")
ax2.set_ylim(0.3, 0.82)
ax2.set_title("(b) SGD, $M{=}48$:\nequivalence", loc="left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig8b_dynamics.{ext}", bbox_inches="tight")
print("saved fig8b")
