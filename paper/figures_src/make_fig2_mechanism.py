"""Fig 2 (v2 paper): the exact mechanism of Theorem 1 (thm:serial).

Adapted from make_fig1_serial.py (v1 Fig 1); adds panel (c), the expected
reversal error over isotropic Gaussian initialization against width, with
the limit Phi(m/2 sigma) (Theorem 1(b)).

Panel (a): spectral update fraction (a(T_m)-a0)/(m-a0) at the matched
margin m = 2.9 versus width M for four (a0, v0); dashed = bound
1/(1+M v0^2); grey = normalized readout (part (c)), flat at the M = 1 value.
Panel (b): joint-minus-frozen margin gap sup_{0<=t<=T_m}(q_J - q_F) with the
bound C0/(p0 M v0^2) dashed (finite horizon: the gap is only bounded on the
fitting interval).
Panel (c): E_init Err_rev versus M for (a0, v0) ~ N(0, sigma^2 I), m = 2,
sigma = 1, with the limit Phi(m/(2 sigma)) = Phi(1); spectral-only
comparator error is zero on every draw.

Inputs:  results/toy_serial_ce_isotropic.csv (m = 2.9 slice)
         results/toy_serial_ce_isotropic_gaussian.csv
Outputs: paper/figures/fig2_mechanism.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

PAIRS = [(0.0, 1.0), (-0.3, 1.0), (1.7, 0.2), (-1.5, -1.3)]
COLORS = {(0.0, 1.0): "#2c7fb8", (-0.3, 1.0): "#e6550d",
          (1.7, 0.2): "#31a354", (-1.5, -1.3): "#756bb1"}
LABELS = {(0.0, 1.0): r"$(0,\,1.0)$", (-0.3, 1.0): r"$(-0.3,\,1.0)$",
          (1.7, 0.2): r"$(1.7,\,0.2)$", (-1.5, -1.3): r"$(-1.5,\,-1.3)$"}
C_CTRL = "#8c8c8c"
M_GRID = [1, 4, 16, 64, 256, 1024, 4096]
M_TARGET = 2.9

iso = pd.read_csv(ROOT / "results" / "toy_serial_ce_isotropic.csv")
iso = iso[iso.m == M_TARGET]
series = {}
for a0, v0 in PAIRS:
    g = iso[(iso.a0 == a0) & (iso.v0 == v0)].sort_values("M")
    assert list(g.M) == M_GRID, f"missing widths for ({a0}, {v0})"
    assert (g.immediate_stop == 0).all()
    series[(a0, v0)] = dict(
        M=g.M.to_numpy(float), frac=g.suppression_I4.to_numpy(float),
        frac_bound=g.suppression_bound_I2.to_numpy(float),
        ctrl=g.ctrl_suppression.to_numpy(float), gap=g.gap_sup_cf.to_numpy(float),
        gap_bound=g.gap_bound_uniform_I5.to_numpy(float),
        C0=g.C0_I5.to_numpy(float), p0=g.p0_I5.to_numpy(float))
for (a0, v0), s in series.items():
    np.testing.assert_allclose(s["frac_bound"], 1.0 / (1.0 + s["M"] * v0 ** 2), rtol=1e-12)
    assert np.all(s["frac"] <= s["frac_bound"]) and np.all(s["frac"] > 0)
    np.testing.assert_allclose(s["gap_bound"], s["C0"] / (s["p0"] * s["M"] * v0 ** 2), rtol=1e-12)
    assert np.all(s["gap"] >= 0) and np.all(s["gap"] <= s["gap_bound"])
    np.testing.assert_allclose(s["ctrl"], s["ctrl"][0], rtol=1e-12)
    np.testing.assert_allclose(s["ctrl"][0], s["frac"][0], rtol=1e-12)

gau = pd.read_csv(ROOT / "results" / "toy_serial_ce_isotropic_gaussian.csv")
assert (gau.E_err_reversal_spectral_only == 0).all()
G_SIGMA, G_M = 1.0, 2.0
gs = gau[(gau.sigma == G_SIGMA) & (gau.m == G_M)].sort_values("M")
assert len(gs) >= 3, "gaussian sweep slice missing"
phi = float(gs.Phi_m_over_2sigma.iloc[0])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.6, 2.05))


def _style_x(ax, grid):
    ax.set_xscale("log")
    ax.set_xticks(grid)
    ax.set_xticklabels([str(int(M)) for M in grid])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r"width $M$")


for a0, v0 in PAIRS:
    s, c = series[(a0, v0)], COLORS[(a0, v0)]
    ax1.plot(s["M"], s["ctrl"], color=C_CTRL, lw=0.9, alpha=0.8, zorder=1)
    ax1.plot(s["M"], s["frac_bound"], color=c, lw=0.9, ls="--", alpha=0.85, zorder=2)
    ax1.plot(s["M"], s["frac"], color=c, lw=0.8, marker="o", ms=2.4, zorder=3, label=LABELS[(a0, v0)])
ax1.set_yscale("log"); ax1.set_ylim(6e-5, 4.0)
ax1.set_ylabel(r"$(a(T_m)-a_0)/(m-a_0)$")
ax1.set_title("(a) spectral update at matched fit", loc="left", fontsize=7)
h_pairs, l_pairs = ax1.get_legend_handles_labels()
style_h = [plt.Line2D([], [], color="0.35", lw=0.9, ls="--"), plt.Line2D([], [], color=C_CTRL, lw=0.9)]
style_l = [r"bound $1/(1{+}Mv_0^2)$", "normalized readout"]
leg = ax1.legend(h_pairs + style_h, l_pairs + style_l, title=r"$(a_0,\,v_0)$", loc="lower left",
                 ncol=2, frameon=False, fontsize=5.2, handlelength=1.4, handletextpad=0.4,
                 columnspacing=0.7, labelspacing=0.25, borderpad=0.0, borderaxespad=0.1)
leg.get_title().set_fontsize(5.5)
_style_x(ax1, M_GRID); ax1.set_xlim(0.6, 9000)

for a0, v0 in PAIRS:
    s, c = series[(a0, v0)], COLORS[(a0, v0)]
    ax2.plot(s["M"], s["gap_bound"], color=c, lw=0.9, ls="--", alpha=0.85, zorder=2)
    ax2.plot(s["M"], s["gap"], color=c, lw=0.8, marker="o", ms=2.4, zorder=3)
ax2.set_yscale("log"); ax2.set_ylim(1.5e-3, 5e3)
ax2.set_ylabel(r"$\sup_{0\le t\le T_m}(q_J-q_F)$")
ax2.set_title("(b) joint$-$frozen gap on $[0,T_m]$", loc="left", fontsize=7)
ax2.annotate(r"dashed: $C_0/(p_0Mv_0^2)$", xy=(0.03, 0.03), xycoords="axes fraction",
             ha="left", va="bottom", fontsize=6, color="0.25")
_style_x(ax2, M_GRID); ax2.set_xlim(0.6, 9000)

ax3.axhline(phi, color="0.35", lw=0.9, ls="--", zorder=1)
ax3.errorbar(gs.M, gs.E_err_reversal, yerr=3 * gs.mc_se, color="#d62728", lw=0.8, marker="o",
             ms=2.6, capsize=1.5, zorder=3, label=r"$\mathbb{E}_{W_0}\,\mathrm{Err}_{\rm rev}$")
ax3.axhline(0.0, color="#31a354", lw=0.9, zorder=2, label="spectral-only comparator")
ax3.set_ylim(-0.03, 1.0)
ax3.set_ylabel("reversal error")
ax3.set_title(r"(c) reversal, $(a_0,v_0)\sim\mathcal{N}(0,I)$, $m=2$", loc="left", fontsize=7)
ax3.annotate(r"limit $\Phi(m/2\sigma)=%.4f$" % phi, xy=(0.97, phi + 0.03), xycoords=("axes fraction", "data"),
             ha="right", va="bottom", fontsize=6, color="0.25")
ax3.legend(loc="center right", frameon=False, fontsize=5.5, bbox_to_anchor=(1.0, 0.45))
_style_x(ax3, [int(M) for M in gs.M])

fig.tight_layout(pad=0.35, w_pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig2_mechanism.{ext}", bbox_inches="tight")
print("panel (c) rows:")
print(gs[["M", "E_err_reversal", "mc_se", "Phi_m_over_2sigma"]].to_string(index=False))
print(f"saved {OUT / 'fig2_mechanism.pdf'}")
