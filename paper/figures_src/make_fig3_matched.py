"""Fig 3: paired arm differences with 90% paired-t intervals (E3c, breast
fold 0, h = 48, AdamW, n = 3 seeds).

Four contrasts per panel, all paired by seed identifier:
  1. joint_linear_m   - frozen_random_m        (matched protocol)
  2. finetune_real_m  - frozen_pretrained_m    (matched protocol)
  3. joint_linear     - frozen_random          (original protocol; the
     freezing contrast is bundled with the clipping-scope and
     BN-affine-treatment differences)
  4. the paired change of contrast 1 relative to contrast 3, i.e. how the
     joint/frozen gap moves when the protocol differences are removed.

Panel (a): the pre-specified endpoint, mean validation macro-F1 over the
final 5 epochs (`val_f1`).
Panel (b): the exploratory best-validation macro-F1 (`val_f1_best`,
post-hoc maximum over 60 evaluations on the 28-core validation split).

Small points are the three per-seed paired differences, the large marker
is their mean, the bar is the 90% paired-t interval (t_{0.95,2} = 2.920).
These are conditional seed summaries on a single validation fold, not a
population analysis; both panels share an x-axis so interval widths are
directly comparable.

Input:   results/e3c_analysis_runs.csv
Outputs: paper/figures/fig3_matched.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 7.5,
    "legend.fontsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

WIDTH = 48
OPTIMIZER = "adamw"
N_SEEDS = 3
T_CRIT = 2.920  # t_{0.95, df = 2}: two-sided 90% interval

# rows, top to bottom.  kind "pair" = a single paired contrast,
# "diff_of_pairs" = contrast (a, b) minus contrast (c, d).
ROWS = [
    dict(kind="pair", arms=("joint_linear_m", "frozen_random_m"),
         label="joint $-$ frozen-random\n(matched)",
         color="#d95f02"),
    dict(kind="pair", arms=("finetune_real_m", "frozen_pretrained_m"),
         label="fine-tune $-$ frozen-pretrained\n(matched)",
         color="#1b9e77"),
    dict(kind="pair", arms=("joint_linear", "frozen_random"),
         label="joint $-$ frozen-random (original\nprotocol; bundled intervention)",
         color="#7570b3"),
    dict(kind="diff_of_pairs",
         arms=("joint_linear_m", "frozen_random_m",
               "joint_linear", "frozen_random"),
         label="change of joint$-$frozen gap,\nmatched $-$ original",
         color="#555555"),
]

PANELS = [
    ("val_f1", "(a) final-5 endpoint (pre-specified)"),
    ("val_f1_best", "(b) best-validation (exploratory)"),
]

# Astra's reproduction (review_packet/astra/reply_02.md, Q6); the
# original-protocol final-5 contrast is the -0.110 quoted in section 8.
TARGETS = {
    ("val_f1", 0): (+0.01057, -0.14192, +0.16305),
    ("val_f1", 1): (+0.02057, -0.02181, +0.06295),
    ("val_f1", 2): (-0.10949, None, None),
    ("val_f1", 3): (+0.12006, -0.23031, +0.47043),
    ("val_f1_best", 0): (+0.02321, -0.06588, +0.11230),
    ("val_f1_best", 1): (-0.00420, -0.04782, +0.03942),
    ("val_f1_best", 3): (-0.00995, -0.10503, +0.08513),
}
TOL = 1e-3

df = pd.read_csv(ROOT / "results" / "e3c_analysis_runs.csv")
sub = df[(df.width == WIDTH) & (df.optimizer == OPTIMIZER)]


def arm_values(arm, column):
    """Seed-sorted values for one arm; returns (seeds, values)."""
    d = sub[sub.arm == arm].sort_values("seed")
    assert len(d) == N_SEEDS, f"{arm}: expected {N_SEEDS} runs, got {len(d)}"
    return d.seed.to_numpy(), d[column].to_numpy()


def paired(arm_a, arm_b, column):
    """Per-seed differences arm_a - arm_b, paired on the seed identifier."""
    sa, va = arm_values(arm_a, column)
    sb, vb = arm_values(arm_b, column)
    assert np.array_equal(sa, sb), f"seed mismatch {arm_a}/{arm_b}: {sa} {sb}"
    return va - vb


def summarize(d):
    """Mean and 90% paired-t interval of a vector of paired differences."""
    mean = d.mean()
    half = T_CRIT * d.std(ddof=1) / np.sqrt(len(d))
    return mean, mean - half, mean + half


def row_diffs(row, column):
    if row["kind"] == "pair":
        a, b = row["arms"]
        return paired(a, b, column)
    a, b, c, e = row["arms"]
    return paired(a, b, column) - paired(c, e, column)


# ----------------------------------------------------------------- compute
stats = {}
for column, _ in PANELS:
    for i, row in enumerate(ROWS):
        d = row_diffs(row, column)
        stats[(column, i)] = (d,) + summarize(d)

# internal consistency: the gap change is exactly contrast 1 minus contrast 3
for column, _ in PANELS:
    got = stats[(column, 3)][1]
    want = stats[(column, 0)][1] - stats[(column, 2)][1]
    assert abs(got - want) < 1e-12, f"{column}: gap-change mean inconsistent"

# agreement with the independently reproduced targets
for (column, i), (mean, lo, hi) in TARGETS.items():
    _, g_mean, g_lo, g_hi = stats[(column, i)]
    assert abs(g_mean - mean) < TOL, f"{column} row {i}: mean {g_mean} vs {mean}"
    if lo is not None:
        assert abs(g_lo - lo) < TOL, f"{column} row {i}: lo {g_lo} vs {lo}"
        assert abs(g_hi - hi) < TOL, f"{column} row {i}: hi {g_hi} vs {hi}"

# ------------------------------------------------------------------- plot
y_pos = np.arange(len(ROWS))[::-1]  # row 0 on top
finite = np.concatenate([
    np.concatenate([stats[(c, i)][0], stats[(c, i)][2:4]])
    for c, _ in PANELS for i in range(len(ROWS))
])
pad = 0.06 * (finite.max() - finite.min())
XLIM = (finite.min() - pad, finite.max() + pad)

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), sharex=True, sharey=True,
                         layout="constrained")

for ax, (column, title) in zip(axes, PANELS):
    ax.axvline(0.0, lw=0.7, color="0.55", ls="--", zorder=0)
    # separates the two matched contrasts from the protocol-change rows
    ax.axhline(y_pos[1] - 0.5, lw=0.5, color="0.85", ls=":", zorder=0)
    for i, row in enumerate(ROWS):
        d, mean, lo, hi = stats[(column, i)]
        y = y_pos[i]
        ax.plot([lo, hi], [y, y], lw=1.4, color=row["color"],
                solid_capstyle="butt", zorder=2)
        for x_end in (lo, hi):  # interval caps
            ax.plot([x_end, x_end], [y - 0.13, y + 0.13], lw=1.0,
                    color=row["color"], zorder=2)
        ax.scatter(d, np.full(len(d), y + 0.24), s=6, color=row["color"],
                   alpha=0.85, edgecolors="white", linewidths=0.3, zorder=3)
        ax.scatter([mean], [y], s=26, marker="D", color=row["color"],
                   edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_title(title, loc="left")
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.62, len(ROWS) - 0.28)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)

axes[0].set_yticks(y_pos)
axes[0].set_yticklabels([r["label"] for r in ROWS], linespacing=1.25)
fig.supxlabel("paired difference in macro-F1 (n = 3 seeds)", fontsize=8)

for ext in ("pdf", "png"):
    # small pad so the cropped page stays inside the 5.5 in text width
    fig.savefig(OUT / f"fig3_matched.{ext}", bbox_inches="tight",
                pad_inches=0.02)

# ------------------------------------------------------------------ report
print(f"E3c, breast fold 0, h = {WIDTH}, {OPTIMIZER}, n = {N_SEEDS} seeds; "
      f"90% paired-t interval, t = {T_CRIT}")
for column, title in PANELS:
    print(f"\n{title}   [{column}]")
    for i, row in enumerate(ROWS):
        d, mean, lo, hi = stats[(column, i)]
        name = row["label"].replace("\n", " ").replace("$-$", "-")
        seeds = ", ".join(f"{v:+.5f}" for v in d)
        print(f"  {name:<62}{mean:+.5f} [{lo:+.5f}, {hi:+.5f}]   "
              f"seeds: {seeds}")
print("\nsaved fig3_matched.{pdf,png}")
