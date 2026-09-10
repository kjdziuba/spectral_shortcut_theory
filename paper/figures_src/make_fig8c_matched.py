"""Fig-8C: matched-protocol arms (E3c, breast fold 0, M=48, AdamW).

Seven arms run under the matched protocol -- identical BN-affine
treatment, phi-only clip scope, best-validation checkpoints saved.

Panel (a): best-validation macro-F1 (post-hoc maximum over 60
evaluations on the 28-core validation split; exploratory).
Panel (b): the pre-registered metric, mean val macro-F1 over the final
5 epochs.

Bars are seed means (whiskers: sd), dots are the 3 individual seeds.
Light brackets mark the two pre-registered pairs.

Input:   results/e3c_analysis_runs.csv
Outputs: paper/figures/fig8c_matched.{pdf,png}
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
    "legend.fontsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

WIDTH = 48

# plotting order requested for the figure (not the table order)
ARMS = [
    "frozen_random_m",
    "joint_linear_m",
    "joint_linear_speclr0.1_m",
    "joint_linear_speclr10_m",
    "joint_linear_m_cosine",
    "frozen_pretrained_m",
    "finetune_real_m",
]
LABELS = {
    "frozen_random_m": "frozen random",
    "joint_linear_m": "joint (linear)",
    "joint_linear_speclr0.1_m": "joint $\\times 0.1$",
    "joint_linear_speclr10_m": "joint $\\times 10$",
    "joint_linear_m_cosine": "joint cosine",
    "frozen_pretrained_m": "frozen pretrained",
    "finetune_real_m": "fine-tune pretr.",
}
# Dark2-based, colourblind-safe: purple = frozen-random (as in Fig 8B),
# orange family = joint arms, green family = pretrained arms.
COLORS = {
    "frozen_random_m": "#7570b3",
    "joint_linear_m": "#d95f02",
    "joint_linear_speclr0.1_m": "#fdb863",
    "joint_linear_speclr10_m": "#8c2d04",
    "joint_linear_m_cosine": "#e6ab02",
    "frozen_pretrained_m": "#1b9e77",
    "finetune_real_m": "#66a61e",
}

# the two pre-registered pairs: (arm_a, arm_b, short label)
PAIRS = [
    ("joint_linear_m", "frozen_random_m", "joint vs frozen"),
    ("finetune_real_m", "frozen_pretrained_m", "FT vs frozen pretr."),
]

YLIM = (0.45, 0.95)

df = pd.read_csv(ROOT / "results" / "e3c_analysis_runs.csv")
sub = df[(df.width == WIDTH) & (df.arm.isin(ARMS))]
assert len(sub) == 3 * len(ARMS), f"expected 21 runs, got {len(sub)}"

x = np.arange(len(ARMS))


def bracket(ax, i, j, text, y):
    """Light square bracket spanning bars i..j with a small caption."""
    h = 0.012
    ax.plot([i, i, j, j], [y, y + h, y + h, y],
            lw=0.6, color="0.45", clip_on=False, zorder=4)
    ax.text((i + j) / 2, y + h + 0.004, text, ha="center", va="bottom",
            fontsize=5.8, color="0.35")


def panel(ax, column, title):
    means, tops = {}, {}
    for i, arm in enumerate(ARMS):
        pts = sub[sub.arm == arm][column].to_numpy()
        mean, sd = pts.mean(), pts.std(ddof=1)
        means[arm] = mean
        tops[arm] = max(pts.max(), mean + sd)
        ax.bar(i, mean, 0.66, yerr=sd, capsize=2,
               color=COLORS[arm], alpha=0.9, error_kw=dict(lw=0.7))
        ax.scatter(np.full(len(pts), i), pts, s=10, color="k", alpha=0.75,
                   zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ARMS], rotation=32, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.7, len(ARMS) - 0.3)
    ax.set_ylim(*YLIM)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", length=0, pad=1.5)

    # brackets sit above every point/whisker they span, at a common height
    ybr = max(tops[a] for pair in PAIRS for a in pair[:2]) + 0.030
    for a, b, text in PAIRS:
        bracket(ax, ARMS.index(a), ARMS.index(b), text, ybr)
    return [means[a] for a in ARMS]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.65), sharey=True)

m_best = panel(ax1, "val_f1_best", "(a) best-validation (exploratory)")
m_final = panel(ax2, "val_f1", "(b) final-5 mean (pre-registered)")

ax1.set_ylabel("val macro-F1")

fig.tight_layout(w_pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig8c_matched.{ext}", bbox_inches="tight")

print(f"{'arm':<26}{'best-val':>10}{'final-5':>10}")
for arm, b, f in zip(ARMS, m_best, m_final):
    print(f"{arm:<26}{b:>10.4f}{f:>10.4f}")
print("saved fig8c")
