"""Fig-2 (ICLR): diagnostic limits, three-panel composite (skeleton Sec. 3).

Model and optimizer are named in every panel title.

Panel (a) -- production directional anisotropy at initialization
  (BlockViT-v2, breast fold 0, M=192).  Identical content and style to
  panel (b) of ``make_fig8a_geometry.py``: curvature along three
  directions for each of the 9 measurements (3 seeds x 3 batches) whose
  batch contains both classes,
    1. the mean-spectrum direction v1                (lam_v1)
    2. the full, un-orthogonalized CancerEpi-CAS contrast (lam_d_full)
    3. that contrast orthogonalized to v1            (lam_d_perp)
  Thin grey lines connect the three values of one measurement; the
  mean/full ratio is ~3.7x, the mean/orthogonalized ratio is ~20x -- the
  large anisotropy is a statement about the v1-orthogonal *part* of the
  contrast, not about the contrast as a whole.
  Data: results/exp1_8b_directions.csv.

Panel (b) -- production BN intervention (Exp 1.8d).  log-log lam_max of
  the GGN block of the first head convolution (``seg_head[0].weight``,
  column ``lam_WW``) against M in {48, 192, 384}, for the four BN modes
  train / eval_bn1 / eval_bn2 / eval_head (dark -> light).  Per-seed
  points plus the pooled line through the seed means; the legend carries
  the pooled log-log slope, which is the OLS slope of log10(mean over
  seeds of lam_WW) on log10(M) -- the "pooled-mean" definition of
  results/exp1_8d_REPORT.md Table "log-log slopes vs M".  The incoming
  3x3 patch-Gram top eigenvalue (``lamS9_valid``) is drawn per seed as a
  flat grey dashed line: it is constant in M to five figures.
  Data: results/exp1_8d_witness_jvp.csv, direction == 'ggn_top'.

Panel (c) -- shallow verified instance (Exp 1.2 v5, joint arm, Adam,
  probe N = 256, C = 2).  Against step (symlog, step 0 at the left):
  kappa_eff / lam_max(K_phi) solid (unweighted kernel; median thick,
  seeds thin) and the residual's top-20 mass f_20 dotted, in the same
  colour per width D in {128, 512, 2048}.  NO isotropic reference line:
  the reference would have to be recomputed inside the class-contrast
  subspace of dimension N(C-1) (Astra correction 1).
  Data: results/exp1_2v5_residual_export.csv (see
  results/exp1_2v5_residual_export_REPORT.md Sec. 2 for the columns).

Outputs: paper/figures/fig2_diagnostics.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

# --- panel (a): colours carried over verbatim from make_fig8a_geometry.py
C_MEAN = "#4d4d4d"
C_FULL = "#e6550d"
C_CONTRAST = "#d7301f"

# --- panel (b): single-hue ramp, train dark -> eval_head light
BN_MODES = ["train", "eval_bn1", "eval_bn2", "eval_head"]
BN_COLORS = {"train": "#08306b", "eval_bn1": "#2171b5",
             "eval_bn2": "#4292c6", "eval_head": "#9ecae1"}
BN_LABELS = {"train": "train", "eval_bn1": "eval bn1",
             "eval_bn2": "eval bn2", "eval_head": "eval head"}
WIDTHS = [48, 192, 384]

# --- panel (c): orange ramp (shares #e6550d with panel (a)), light -> dark in D
DS = [128, 512, 2048]
D_COLORS = {128: "#fd8d3c", 512: "#e6550d", 2048: "#7f2704"}


def loglog_slope(x, y):
    """OLS slope of log10(y) on log10(x)."""
    return float(np.polyfit(np.log10(np.asarray(x, float)),
                            np.log10(np.asarray(y, float)), 1)[0])


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 2.2))

# ------------------------------------------------------------- panel (a)
di = pd.read_csv(RES / "exp1_8b_directions.csv")
tri = di[(di.width == 192) & di.lam_d_full.notna()
         & di.lam_d_perp.notna()].sort_values(["seed", "batch"])
assert sorted(tri.seed.unique()) == [0, 1, 2], "expected all three seeds"
assert len(tri) == 9, f"expected 9 measurements, got {len(tri)}"

x_mean = tri["lam_v1"].to_numpy()        # == summary lam_along_vdata
x_full = tri["lam_d_full"].to_numpy()    # un-orthogonalized class contrast
x_perp = tri["lam_d_perp"].to_numpy()    # == summary lam_contrast_CancerEpi-CAS
r_full = x_mean / x_full
r_perp = x_mean / x_perp

for a, b, c in zip(x_mean, x_full, x_perp):
    ax1.plot([0, 1, 2], [a, b, c], color="0.75", lw=0.7, zorder=1)
for xpos, vals, color in [(0, x_mean, C_MEAN), (1, x_full, C_FULL),
                          (2, x_perp, C_CONTRAST)]:
    ax1.scatter(np.full_like(vals, xpos), vals, s=13, color=color, zorder=2)

# Ratio annotations, to the precision used in the body text: 3.65 -> 3.7x,
# 19.67 -> 20x.
assert 3.6 <= r_full.mean() <= 3.8 and 19.0 <= r_perp.mean() <= 21.0
ax1.text(0.5, 15.0, "mean / full\n$\\approx 3.7\\times$", ha="center",
         va="center", fontsize=6, color="0.25", linespacing=1.2)
ax1.text(1.5, 520.0, "mean / orthog.\n$\\approx 20\\times$", ha="center",
         va="center", fontsize=6, color="0.25", linespacing=1.2)

ax1.set_yscale("log")
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(6, 2600)
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(["mean\nspectrum", "full\ncontrast",
                     "$v_1$-orth.\ncontrast"], fontsize=6)
for tick, color in zip(ax1.get_xticklabels(), [C_MEAN, C_FULL, C_CONTRAST]):
    tick.set_color(color)
ax1.tick_params(axis="x", length=0, pad=2)
ax1.set_ylabel("curvature along dir.", fontsize=7)
ax1.set_title("(a) BlockViT-v2, init:\ndirectional anisotropy", loc="left",
              fontsize=7, linespacing=1.25)

# ------------------------------------------------------------- panel (b)
jv = pd.read_csv(RES / "exp1_8d_witness_jvp.csv")
gg = jv[jv.direction == "ggn_top"]
assert sorted(gg.width.unique()) == WIDTHS
assert len(gg) == 36, f"expected 4 modes x 3 widths x 3 seeds, got {len(gg)}"

slopes = {}
for mode in BN_MODES:
    s = gg[gg["mode"] == mode]
    col = BN_COLORS[mode]
    ax2.scatter(s.width, s.lam_WW, s=8, color=col, alpha=0.55, zorder=2,
                linewidths=0)
    means = s.groupby("width").lam_WW.mean().reindex(WIDTHS)
    slopes[mode] = loglog_slope(means.index, means.values)
    ax2.plot(means.index, means.values, color=col, lw=1.6, zorder=3,
             marker="o", ms=2.6,
             label=f"{BN_LABELS[mode]} ${slopes[mode]:+.2f}$"
                   if mode != "eval_head"
                   else f"{BN_LABELS[mode]} ${slopes[mode]:+.3f}$")

# incoming 3x3 patch Gram, one flat dashed line per seed
for i, (sd, s) in enumerate(gg.groupby("seed")):
    s = s[s["mode"] == "train"].sort_values("width")
    ax2.plot(s.width, s.lamS9_valid, color="0.6", ls="--", lw=0.8, zorder=1,
             label="patch Gram (flat)" if i == 0 else None)
gram_slope = loglog_slope(
    gg.groupby("width").lamS9_valid.mean().reindex(WIDTHS).index,
    gg.groupby("width").lamS9_valid.mean().reindex(WIDTHS).values)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(40, 470)
# headroom at the BOTTOM (no data below lam_WW = 1.33) carries the legend
ax2.set_ylim(0.018, 800)
ax2.xaxis.set_major_locator(FixedLocator(WIDTHS))
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xticklabels([str(w) for w in WIDTHS])
ax2.set_xlabel("$M$")
ax2.set_ylabel(r"$\lambda_{\max}$ (head-conv block)", fontsize=7)
assert gg.lam_WW.min() > 1.0, "legend corner assumed empty"
ax2.legend(frameon=False, loc="lower left", fontsize=5.4, handlelength=1.1,
           labelspacing=0.22, borderaxespad=0.1, handletextpad=0.5)
ax2.set_title("(b) BlockViT-v2, init:\nhead BN modes", loc="left",
              fontsize=7, linespacing=1.25)

# ------------------------------------------------------------- panel (c)
rr = pd.read_csv(RES / "exp1_2v5_residual_export.csv")
jo = rr[rr.arm == "joint"].copy()
# unweighted spatial kernel (lam_max_Kphi_H is the H-weighted variant)
jo["ratio"] = jo.kappa_eff / jo.lam_max_Kphi
assert sorted(jo.width.unique()) == DS

med_t0, med_tend = {}, {}
last_step = int(jo.step.max())
for D in DS:
    col = D_COLORS[D]
    s = jo[jo.width == D]
    for sd, g in s.groupby("seed"):
        g = g.sort_values("step")
        ax3.plot(g.step, g.ratio, color=col, lw=0.5, alpha=0.45, zorder=2)
        ax3.plot(g.step, g.frac_r_topk20_Kphi, color=col, lw=0.5, ls=":",
                 alpha=0.45, zorder=2)
    m = s.groupby("step")[["ratio", "frac_r_topk20_Kphi"]].median()
    ax3.plot(m.index, m.ratio, color=col, lw=1.5, zorder=3,
             label=f"$D={D}$")
    ax3.plot(m.index, m.frac_r_topk20_Kphi, color=col, lw=1.5, ls=":",
             zorder=3)
    med_t0[D] = (m.ratio.loc[0], m.frac_r_topk20_Kphi.loc[0])
    med_tend[D] = (m.ratio.loc[last_step], m.frac_r_topk20_Kphi.loc[last_step])

ax3.set_xscale("symlog", linthresh=1)
ax3.set_yscale("log")
ax3.set_xlim(0, 2400)
ax3.set_ylim(1.5e-4, 1.6)
ax3.xaxis.set_major_locator(FixedLocator([0, 1, 10, 100, 1000]))
ax3.set_xticklabels(["0", "1", "10", "$10^2$", "$10^3$"])
ax3.xaxis.set_minor_formatter(NullFormatter())
ax3.set_xlabel("step")
ax3.set_ylabel(r"$\kappa_{\mathrm{eff}}/\lambda_{\max}$ (solid),"
               "\n" r"top-20 mass (dotted)", fontsize=7, linespacing=1.3)
handles = [Line2D([], [], color=D_COLORS[D], lw=1.5, label=f"$D={D}$")
           for D in DS]
ax3.legend(handles=handles, frameon=False, loc="lower left", fontsize=5.8,
           handlelength=1.1, labelspacing=0.22, borderaxespad=0.1,
           handletextpad=0.5)
ax3.set_title("(c) shallow instance,\nAdam, probe $N = 256$", loc="left",
              fontsize=7, linespacing=1.25)

fig.tight_layout(w_pad=0.9)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig2_diagnostics.{ext}", bbox_inches="tight")

# ------------------------------------------------------------- printout
print(f"panel (a): n measurements = {len(tri)}  "
      f"mean/full {r_full.mean():.3f} ({r_full.min():.2f}-{r_full.max():.2f}), "
      f"mean/orthog. {r_perp.mean():.3f} "
      f"({r_perp.min():.2f}-{r_perp.max():.2f})")
print("panel (b) pooled log-log slopes of lam_WW vs M "
      "(OLS on log10 of the seed mean):")
for mode in BN_MODES:
    per_seed = [loglog_slope(g.sort_values("width").width,
                             g.sort_values("width").lam_WW)
                for _, g in gg[gg["mode"] == mode].groupby("seed")]
    print(f"  {mode:<10s} {slopes[mode]:+.4f}   per-seed "
          + " ".join(f"{p:+.3f}" for p in per_seed))
print(f"  {'patch Gram':<10s} {gram_slope:+.6f}")
print(f"panel (c) medians over 3 seeds (joint arm), "
      f"t = 0 and t = {last_step}:")
print(f"  {'D':>5}  {'kappa_eff/lam_max t=0':>22}  {'t=last':>10}"
      f"  {'f_20 t=0':>10}  {'t=last':>10}")
for D in DS:
    print(f"  {D:>5}  {med_t0[D][0]:>22.5f}  {med_tend[D][0]:>10.5f}"
          f"  {med_t0[D][1]:>10.5f}  {med_tend[D][1]:>10.5f}")
print(f"saved {OUT / 'fig2_diagnostics.pdf'}")
