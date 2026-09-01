"""Fig-8A: direction-wise starvation at initialization (E3b, breast fold 0,
M=192, 3 seeds x 4 batches).

Panel (a): GGN block eigenvalue spectra — spectral (theta) top-40 vs spatial
(phi) top-15. Thin lines = individual (seed, batch) measurements; thick =
mean over the 12 measurements. The spatial spectrum overtakes the spectral
one at rank 2-5: theta's curvature is concentrated in a handful of
directions.

Panel (b): curvature along the mean-spectrum direction vs along the
CancerEpi-CAS class-contrast direction, per measurement (6 batches contain
both classes). The contrast direction is 12-28x starved.

Inputs:  results/exp1_8b_spectra.csv, results/exp1_8b_summary.csv
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
C_CONTRAST = "#d7301f"

sp = pd.read_csv(ROOT / "results" / "exp1_8b_spectra.csv")
su = pd.read_csv(ROOT / "results" / "exp1_8b_summary.csv")

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(5.5, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})

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
pairs = su.dropna(subset=["lam_contrast_CancerEpi-CAS", "lam_along_vdata"])
x_mean = pairs["lam_along_vdata"].to_numpy()
x_con = pairs["lam_contrast_CancerEpi-CAS"].to_numpy()

for i, (a, b) in enumerate(zip(x_mean, x_con)):
    ax2.plot([0, 1], [a, b], color="0.75", lw=0.7, zorder=1)
ax2.scatter(np.zeros_like(x_mean), x_mean, s=18, color=C_MEAN, zorder=2,
            label="mean-spectrum dir.")
ax2.scatter(np.ones_like(x_con), x_con, s=18, color=C_CONTRAST, zorder=2,
            label="CancerEpi–CAS contrast")
ratios = x_mean / x_con
ax2.text(0.5, np.sqrt(x_mean.mean() * x_con.mean()),
         f"{ratios.min():.0f}–{ratios.max():.0f}$\\times$",
         ha="center", va="center", fontsize=8,
         bbox=dict(fc="white", ec="none", pad=1))

ax2.set_yscale("log")
ax2.set_xlim(-0.45, 1.45)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["mean-spectrum\ndirection", "class-contrast\ndirection"])
ax2.set_ylabel(r"curvature along direction")
ax2.set_title("(b) direction-wise starvation", loc="left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig8a_geometry.{ext}", bbox_inches="tight")
print("ratios:", np.sort(ratios).round(1))
print("saved fig8a")
