"""Fig-1: the solvable serial CE instance (Sec. 6, thm:instance).

Model: q = a + b v with b = sum_j beta_j over M replicated readouts, logistic
CE, unit-rate gradient flow, zero-sum readout initialization. At
initialization D_curv(0) = M and the envelope rate is mu = (M+1)/4.

Panel (a): the spectral update fraction (a(T_m) - a0)/(m - a0) at the time
T_m when the margin first reaches m, versus width M, for four (a0, v0)
initializations. Solid: the closed-form solution (column suppression_I4).
Dashed, matching colour: the exact (I2) bound 1/(1 + M v0^2). Flat grey: the
normalized-readout control f = M^{-1/2} U h, whose update fraction is the
M = 1 value at every width (layer-balance invariant) -- the M-dependence is a
property of the parameterization, not of the task.

Panel (b): the joint-vs-frozen margin gap sup_t (q_J - q_F) versus M
(column gap_sup_cf), with the uniform (I5) bound C0/(p0 M v0^2) dashed in the
matching colour.

Inputs:  results/toy_serial_ce_isotropic.csv  (579 rows; m = 2.9 slice)
         results/toy_serial_ce.csv            (a0 = 0, v0 = 1 cross-check,
                                               incl. the 1/sqrt(M) control)
Outputs: paper/figures/fig1_serial.{pdf,png}
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

# (a0, v0) initializations, in the order they appear in the legend.
PAIRS = [(0.0, 1.0), (-0.3, 1.0), (1.7, 0.2), (-1.5, -1.3)]
COLORS = {(0.0, 1.0): "#2c7fb8", (-0.3, 1.0): "#e6550d",
          (1.7, 0.2): "#31a354", (-1.5, -1.3): "#756bb1"}
LABELS = {(0.0, 1.0): r"$(0,\,1.0)$", (-0.3, 1.0): r"$(-0.3,\,1.0)$",
          (1.7, 0.2): r"$(1.7,\,0.2)$", (-1.5, -1.3): r"$(-1.5,\,-1.3)$"}
C_CTRL = "#8c8c8c"
M_GRID = [1, 4, 16, 64, 256, 1024, 4096]
M_TARGET = 2.9

# --------------------------------------------------------------------- data
iso = pd.read_csv(ROOT / "results" / "toy_serial_ce_isotropic.csv")
iso = iso[iso.m == M_TARGET]

series = {}
for a0, v0 in PAIRS:
    g = iso[(iso.a0 == a0) & (iso.v0 == v0)].sort_values("M")
    assert list(g.M) == M_GRID, f"missing widths for ({a0}, {v0}): {list(g.M)}"
    assert (g.immediate_stop == 0).all(), f"immediate stop at ({a0}, {v0})"
    series[(a0, v0)] = dict(
        M=g.M.to_numpy(float),
        frac=g.suppression_I4.to_numpy(float),        # (a(T_m)-a0)/(m-a0)
        frac_bound=g.suppression_bound_I2.to_numpy(float),   # 1/(1+M v0^2)
        ctrl=g.ctrl_suppression.to_numpy(float),      # normalized readout
        gap=g.gap_sup_cf.to_numpy(float),             # sup_t (q_J - q_F)
        gap_bound=g.gap_bound_uniform_I5.to_numpy(float),    # C0/(p0 M v0^2)
        C0=g.C0_I5.to_numpy(float), p0=g.p0_I5.to_numpy(float),
    )

# --- checks: stored bounds are the stated closed forms, and they hold -------
for (a0, v0), s in series.items():
    tag = f"({a0}, {v0})"
    # u_M/Delta is a fraction of the way from a0 to m.
    assert np.all(s["frac"] > 0) and np.all(s["frac"] < 1), f"frac range {tag}"
    # (I2): 1/(1 + M v0^2).
    np.testing.assert_allclose(s["frac_bound"], 1.0 / (1.0 + s["M"] * v0**2),
                               rtol=1e-12, err_msg=f"(I2) formula {tag}")
    assert np.all(s["frac"] <= s["frac_bound"]), f"(I2) violated at {tag}"
    # (I5): C0/(p0 M v0^2).
    np.testing.assert_allclose(s["gap_bound"], s["C0"] / (s["p0"] * s["M"] * v0**2),
                               rtol=1e-12, err_msg=f"(I5) formula {tag}")
    assert np.all(s["gap"] >= 0), f"negative margin gap at {tag}"
    assert np.all(s["gap"] <= s["gap_bound"]), f"(I5) violated at {tag}"
    # The normalized-readout control is width-independent.
    np.testing.assert_allclose(s["ctrl"], s["ctrl"][0], rtol=1e-12,
                               err_msg=f"control not flat at {tag}")
    np.testing.assert_allclose(s["ctrl"][0], s["frac"][0], rtol=1e-12,
                               err_msg=f"control != M=1 value at {tag}")

# --- cross-check the (0, 1) column against the Thm-5.1 toy (m = log 19) -----
toy = pd.read_csv(ROOT / "results" / "toy_serial_ce.csv")
m_toy = float(toy.loc[toy.quantity.str.startswith("m = log"), "formula_value"].iloc[0])
a_toy = toy[toy.quantity == "a(T_m)"].set_index("M").formula_value
c_toy = toy[toy.quantity.str.startswith("control (1/sqrt(M) readout)")
            ].set_index("M").formula_value
assert np.allclose(c_toy.to_numpy(), a_toy.loc[1]), "toy control is not flat"
# a0 = 0 there, so the update fraction is a(T_m)/m; m = log 19 = 2.944 vs 2.9
# here, so agreement is close but not exact.
s01 = series[(0.0, 1.0)]
shared = [M for M in M_GRID if M in a_toy.index]
np.testing.assert_allclose(
    (a_toy.loc[shared] / m_toy).to_numpy(),
    s01["frac"][[M_GRID.index(M) for M in shared]], rtol=0.03,
    err_msg="toy_serial_ce vs isotropic (0, 1) disagree beyond the m offset")

# --- support for the "reversal error -> Phi(m/2 sigma)" annotation ---------
# Isotropic Gaussian (a0, v0); the spectral-only comparator errs on no draw.
gau = pd.read_csv(ROOT / "results" / "toy_serial_ce_isotropic_gaussian.csv")
assert (gau.E_err_reversal_spectral_only == 0).all(), "spectral-only comparator errs"
wide = gau[gau.M == gau.M.max()]
# 1e-3 absolute slack: where the MC error saturates at 1.0 the s.e. is exactly
# zero, so the "3 s.e." tolerance would otherwise be degenerate.
assert np.all(np.abs(wide["diff_vs_Phi"]) <= 3 * wide["mc_se"] + 1e-3), \
    "reversal error does not match Phi(m/2 sigma) at the largest width"
rev = (gau.sort_values("M").groupby(["sigma", "m"])
       .agg(M_min=("M", "first"), err_min=("E_err_reversal", "first"),
            M_max=("M", "last"), err_max=("E_err_reversal", "last"),
            Phi=("Phi_m_over_2sigma", "first")).reset_index())

# --- the two headline numbers quoted in the caption ------------------------
f03 = series[(-0.3, 1.0)]["frac"]
assert abs(f03[0] - 0.34) < 5e-3, f"(-0.3, 1.0) at M=1 is {f03[0]}"
assert abs(f03[-1] - 2.4e-4) < 5e-6, f"(-0.3, 1.0) at M=4096 is {f03[-1]}"

# -------------------------------------------------------------------- plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.3))


def _style_x(ax):
    ax.set_xscale("log")
    ax.set_xticks(M_GRID)
    ax.set_xticklabels([str(M) for M in M_GRID])
    ax.set_xticks([], minor=True)
    # Wide enough that the "1" and "4096" tick labels sit inside the axes, so
    # the tight bbox is set by the axes and the figure keeps its 5.5 in width.
    ax.set_xlim(0.6, 9000)
    ax.set_xlabel(r"width $M$")


# --------------------------------------------------------------- panel (a)
for a0, v0 in PAIRS:
    s, c = series[(a0, v0)], COLORS[(a0, v0)]
    ax1.plot(s["M"], s["ctrl"], color=C_CTRL, lw=0.9, ls="-", alpha=0.8,
             zorder=1)
    ax1.plot(s["M"], s["frac_bound"], color=c, lw=0.9, ls="--", alpha=0.85,
             zorder=2)
    ax1.plot(s["M"], s["frac"], color=c, lw=0.8, marker="o", ms=2.6,
             zorder=3, label=LABELS[(a0, v0)])

ax1.set_yscale("log")
ax1.set_ylim(6e-5, 4.0)
ax1.set_ylabel(r"$(a(T_m)-a_0)/(m-a_0)$")
ax1.set_title("(a) spectral update fraction vs width (serial CE)", loc="left",
              fontsize=7.5)
ax1.annotate(r"$\mathcal{D}_{\mathrm{curv}}(0)=M$, $\mu=(M+1)/4$",
             xy=(0.985, 0.985), xycoords="axes fraction", ha="right",
             va="top", fontsize=6.5, color="0.25")

# One legend, two columns: the four (a0, v0) pairs on the left, the two line
# styles on the right (blank slots pad the column-major fill).
h_pairs, l_pairs = ax1.get_legend_handles_labels()
blank = plt.Line2D([], [], ls="none")
style_h = [plt.Line2D([], [], color="0.35", lw=0.9, ls="--"),
           plt.Line2D([], [], color=C_CTRL, lw=0.9, ls="-"), blank, blank]
style_l = [r"bound $1/(1{+}Mv_0^2)$", "normalized readout", "", ""]
leg = ax1.legend(h_pairs + style_h, l_pairs + style_l,
                 title=r"$(a_0,\,v_0)$", loc="lower left", ncol=2,
                 frameon=False, fontsize=6, handlelength=1.6,
                 handletextpad=0.5, columnspacing=0.9, labelspacing=0.3,
                 borderpad=0.0, borderaxespad=0.15)
leg.get_title().set_fontsize(6)
leg._legend_box.align = "left"

# --------------------------------------------------------------- panel (b)
for a0, v0 in PAIRS:
    s, c = series[(a0, v0)], COLORS[(a0, v0)]
    ax2.plot(s["M"], s["gap_bound"], color=c, lw=0.9, ls="--", alpha=0.85,
             zorder=2)
    ax2.plot(s["M"], s["gap"], color=c, lw=0.8, marker="o", ms=2.6, zorder=3)

ax2.set_yscale("log")
ax2.set_ylim(1.5e-3, 5e3)
ax2.set_ylabel(r"$\sup_t\,(q_J-q_F)$")
ax2.set_title("(b) joint$-$frozen margin gap", loc="left", fontsize=7.5)
ax2.annotate(r"dashed: bound $C_0/(p_0Mv_0^2)$", xy=(0.03, 0.145),
             xycoords="axes fraction", ha="left", va="bottom", fontsize=6.5,
             color="0.25")
ax2.annotate(r"reversal error $\to\Phi(m/2\sigma)$", xy=(0.03, 0.03),
             xycoords="axes fraction", ha="left", va="bottom", fontsize=6.5,
             color="0.25")

for ax in (ax1, ax2):
    _style_x(ax)

fig.tight_layout(pad=0.4, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1_serial.{ext}", bbox_inches="tight")

# ------------------------------------------------------------------ report
print(f"Fig 1 serial CE instance   m = {M_TARGET}   widths {M_GRID}")
print(f"source: results/toy_serial_ce_isotropic.csv (m = {M_TARGET} slice, "
      f"{sum(len(s['M']) for s in series.values())} plotted widths)")
print()
hdr = "  M".ljust(6) + "".join(x.rjust(13) for x in
                               ["update frac", "(I2) bound", "control",
                                "gap sup", "(I5) bound"])
for a0, v0 in PAIRS:
    s = series[(a0, v0)]
    print(f"(a0, v0) = ({a0:g}, {v0:g})    "
          f"C0 = {s['C0'][0]:.4f}   p0 = {s['p0'][0]:.6f}")
    print(hdr)
    for i, M in enumerate(M_GRID):
        print(f"{M:>5} " + "".join(f"{v:>13.4e}" for v in
                                   (s["frac"][i], s["frac_bound"][i],
                                    s["ctrl"][i], s["gap"][i],
                                    s["gap_bound"][i])))
    print()
print("reversal annotation (results/toy_serial_ce_isotropic_gaussian.csv, "
      "isotropic Gaussian (a0, v0), 4000 draws):")
print("  sigma    m " + "".join(x.rjust(14) for x in
                                ["err(M=64)", "err(M=16384)", "Phi(m/2sigma)"]))
for r in rev.itertuples():
    print(f"  {r.sigma:>5g} {r.m:>4g} " + "".join(
        f"{v:>14.5f}" for v in (r.err_min, r.err_max, r.Phi)))
print("  spectral-only comparator error: 0 on every draw")
print()
print("checks: every update fraction <= 1/(1+M v0^2); every margin gap <= "
      "C0/(p0 M v0^2); control flat and equal to the M=1 value; reversal "
      "error within 3 MC s.e. of Phi(m/2 sigma) at M = 16384")
print(f"headline: (-0.3, 1.0) update fraction {f03[0]:.4f} at M=1 -> "
      f"{f03[-1]:.4e} at M=4096  ({f03[0] / f03[-1]:.0f}x)")
print(f"saved {OUT / 'fig1_serial.pdf'}")
print(f"saved {OUT / 'fig1_serial.png'}")
