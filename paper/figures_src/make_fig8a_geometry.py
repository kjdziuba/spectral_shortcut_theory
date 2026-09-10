"""Fig-8A: direction-wise starvation at initialization (E3b, breast fold 0,
M=192, 3 seeds x 4 batches).

Panel (a): GGN block eigenvalue spectra — spectral (theta) top-40 vs spatial
(phi) top-15. Thin lines = individual (seed, batch) measurements; thick =
mean over the 12 measurements. The spatial spectrum overtakes the spectral
one at rank 2-5: theta's curvature is concentrated in a handful of
directions.

Panel (b): curvature along THREE directions, for each of the 9 measurements
whose batch contains both classes (3 seeds x 3 batches at M=192):
  1. the mean-spectrum direction v1        (lam_v1 == summary lam_along_vdata)
  2. the full, un-orthogonalized
     CancerEpi-CAS class contrast          (lam_d_full)
  3. that contrast orthogonalized to v1    (lam_d_perp == summary
                                            lam_contrast_CancerEpi-CAS)
Thin grey lines connect the three values of a single measurement. The
mean/full ratio is ~3.7x; the mean/orthogonalized ratio is ~20x — i.e. the
large anisotropy is a statement about the v1-orthogonal *part* of the
contrast, not about the contrast as a whole.

Note on sources: exp1_8b_summary.csv carries lam_along_vdata /
lam_contrast_CancerEpi-CAS for seeds 0-1 only; exp1_8b_directions.csv carries
the identical quantities (verified equal where both are defined) plus
lam_d_full for all three seeds, so panel (b) reads the directions file.

Inputs:  results/exp1_8b_spectra.csv, results/exp1_8b_directions.csv
Outputs: paper/figures/fig8a_geometry.{pdf,png}
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

C_THETA = "#4d4d4d"
C_PHI = "#2c7fb8"
C_MEAN = "#4d4d4d"
C_FULL = "#e6550d"
C_CONTRAST = "#d7301f"

sp = pd.read_csv(ROOT / "results" / "exp1_8b_spectra.csv")
di = pd.read_csv(ROOT / "results" / "exp1_8b_directions.csv")

fig, (ax1, ax2) = plt.subplots(
    # Panel (a) keeps its previous width (5.5 x 1.35/2.35 = 3.16 in); the
    # figure is widened only to give panel (b) room for a third column.
    1, 2, figsize=(5.75, 2.5), gridspec_kw={"width_ratios": [1.22, 1]})

# --------------------------------------------------------------- panel (a)
for block, color in [("theta", C_THETA), ("phi", C_PHI)]:
    b = sp[(sp.dataset == "breast") & (sp.block == block)]
    for (s, bt), g in b.groupby(["seed", "batch"]):
        g = g.sort_values("rank")
        ax1.plot(g["rank"], g.eigenvalue, color=color, alpha=0.18, lw=0.6)
    m = b.groupby("rank").eigenvalue.mean()
    label = (r"spectral block $G_{\theta\theta}$" if block == "theta"
             else r"spatial block $G_{\phi\phi}$")
    ax1.plot(m.index, m.values, color=color, lw=1.8, label=label)

lam1_phi = (sp[(sp.dataset == "breast") & (sp.block == "phi")
               & (sp["rank"] == 1)].eigenvalue.mean())
ax1.axhline(lam1_phi, color=C_PHI, lw=0.9, ls="--", alpha=0.9)
ax1.set_yscale("log")
ax1.set_xlim(1, 40)
ax1.set_xlabel("eigenvalue rank")
ax1.set_ylabel(r"GGN block eigenvalue")
ax1.legend(frameon=False, loc="upper right")
ax1.annotate("overtake at rank 2–5",
             xy=(3.2, lam1_phi * 1.05), xytext=(16, 210), fontsize=7,
             color="0.25", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))
ax1.set_title("(a) block spectra at initialization", loc="left")

# --------------------------------------------------------------- panel (b)
# M=192, batches in which the CancerEpi-CAS contrast is defined (both classes
# present); all 3 seeds.
tri = di[(di.width == 192) & di.lam_d_full.notna()
         & di.lam_d_perp.notna()].sort_values(["seed", "batch"])
assert sorted(tri.seed.unique()) == [0, 1, 2], "expected all three seeds"

x_mean = tri["lam_v1"].to_numpy()        # == summary lam_along_vdata
x_full = tri["lam_d_full"].to_numpy()    # un-orthogonalized class contrast
x_perp = tri["lam_d_perp"].to_numpy()    # == summary lam_contrast_CancerEpi-CAS

r_full = (x_mean / x_full)
r_perp = (x_mean / x_perp)

for a, b, c in zip(x_mean, x_full, x_perp):
    ax2.plot([0, 1, 2], [a, b, c], color="0.75", lw=0.7, zorder=1)
for xpos, vals, color, lab in [
        (0, x_mean, C_MEAN, "mean-spectrum dir."),
        (1, x_full, C_FULL, "full class contrast"),
        (2, x_perp, C_CONTRAST, r"$v_1$-orthog. contrast")]:
    ax2.scatter(np.full_like(vals, xpos), vals, s=18, color=color, zorder=2,
                label=lab)

# Ratio annotations. Displayed to the precision used in the body text of
# Sec. 8.3: mean(r_full) = 3.65 -> 3.7x, mean(r_perp) = 19.67 -> 20x.
assert 3.6 <= r_full.mean() <= 3.8 and 19.0 <= r_perp.mean() <= 21.0
ax2.text(0.5, 16.0, "mean / full\n$\\approx 3.7\\times$",
         ha="center", va="center", fontsize=7, color="0.25",
         linespacing=1.25)
ax2.text(1.5, 460.0, "mean / orthog.\n$\\approx 20\\times$",
         ha="center", va="center", fontsize=7, color="0.25",
         linespacing=1.25)

ax2.set_yscale("log")
ax2.set_xlim(-0.5, 2.5)
ax2.set_ylim(6, 2600)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["mean-spectrum\ndirection", "full class\ncontrast",
                     "$v_1$-orthog.\ncontrast"])
for tick, color in zip(ax2.get_xticklabels(), [C_MEAN, C_FULL, C_CONTRAST]):
    tick.set_color(color)
ax2.set_ylabel(r"curvature along direction")
ax2.set_title("(b) directional curvature anisotropy", loc="left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig8a_geometry.{ext}", bbox_inches="tight")

print(f"n measurements (3 seeds x 3 batches): {len(tri)}")
print(f"mean/full        : mean {r_full.mean():.3f}  "
      f"range {r_full.min():.2f}-{r_full.max():.2f}")
print(f"mean/orthogonal. : mean {r_perp.mean():.3f}  "
      f"range {r_perp.min():.2f}-{r_perp.max():.2f}")
print("saved fig8a")
