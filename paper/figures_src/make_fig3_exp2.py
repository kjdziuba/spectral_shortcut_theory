"""Fig 3 (v2 paper): Experiment 2 at matched fit L* = 0.30.

(a) spectral probe gain over initialization vs head width M for sp (standard
    parameterization), mup (normalized readout), ctxfree (uninformative
    training context) and frozen; seeds as points, means as lines.
(b) accuracy under context reversal vs M, same arms.
(c) whole-head rate multiplier kappa at M = 32: probe gain, h_u and reversal
    accuracy (seed points + means).
(d) trajectories of h_u (row-space retention of u) for sp at three widths,
    seed 0, with the loss thresholds crossed marked.

Inputs:  results/exp2_summary.csv (via `exp2_intervention.py --collect`),
         results/exp2/traj_sp_M*_x1_s0.csv, results/exp2/snap_sp_M*_x1_s0.csv
Outputs: paper/figures/fig3_exp2.{pdf,png}; prints the plotted numbers.
Tolerates missing arms (partial grids) so it can be run while the grid runs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
RES = ROOT / "results"
OUT.mkdir(exist_ok=True)
L_STAR = 0.30

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})
ARMS = {"sp": ("standard param.", "#d62728"), "mup": ("normalized readout", "#1f77b4"),
        "ctxfree": ("context uninformative", "#2ca02c"), "frozen": ("frozen encoder", "#7f7f7f")}

df = pd.read_csv(RES / "exp2_summary.csv")
if "head_mult" not in df.columns:
    df["head_mult"] = 1.0
df["head_mult"] = df["head_mult"].fillna(1.0)
star = df[df["threshold"].astype(str) == str(L_STAR)].copy()
base = star[(star["mult"] == 1) & (star["head_mult"] == 1)]

fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.0))
(ax_a, ax_b), (ax_c, ax_d) = axes


def width_panel(ax, metric, ylabel, title):
    for arm, (label, col) in ARMS.items():
        a = base[base["arm"] == arm]
        if a.empty:
            continue
        for _, g in a.groupby("seed"):
            g = g.sort_values("width")
            ax.plot(g["width"], g[metric], "o", color=col, ms=2.2, alpha=0.45)
        m = a.groupby("width")[metric].mean()
        ax.plot(m.index, m.values, "-", color=col, lw=1.6, label=label)
        print(f"{title} {arm}: " + ", ".join(f"M={int(w)}: {v:.3f}" for w, v in m.items()))
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"head width $M$"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left"); ax.grid(True, alpha=0.25)


width_panel(ax_a, "probe_acc_gain", r"probe gain at $L^\ast$", "(a) spectral learning at matched fit")
ax_a.axhline(0, color="k", lw=0.6)
ax_a.legend(frameon=False, loc="upper right")
width_panel(ax_b, "acc_reversed", r"accuracy, context reversed, at $L^\ast$", "(b) reliance: reversal accuracy")
ax_b.axhline(0.5, color="k", ls=":", lw=0.8)
ax_b.set_ylim(-0.02, 1.02)

# (c) kappa sweep at M = 32 (headlr arm; includes downward extension on seed 0)
k = star[star["arm"] == "headlr"].sort_values("head_mult")
if not k.empty:
    for _, g in k.groupby("seed"):
        g = g.sort_values("head_mult")
        ax_c.plot(g["head_mult"], g["probe_acc_gain"], "o", color="#2ca02c", ms=2.2, alpha=0.45)
        ax_c.plot(g["head_mult"], g["h_u"], "s", color="#9467bd", ms=2.2, alpha=0.45)
        ax_c.plot(g["head_mult"], g["acc_reversed"], "^", color="#d62728", ms=2.2, alpha=0.45)
    m = k.groupby("head_mult")[["probe_acc_gain", "h_u", "acc_reversed"]].mean()
    ax_c.plot(m.index, m["probe_acc_gain"], "-", color="#2ca02c", lw=1.6, label="probe gain")
    ax_c.plot(m.index, m["h_u"], "-", color="#9467bd", lw=1.6, label=r"$h_u$")
    ax_c.plot(m.index, m["acc_reversed"], "-", color="#d62728", lw=1.6, label="reversal acc.")
    print("(c) kappa: " + "; ".join(f"k={i:g}: gain {r.probe_acc_gain:.3f}, h_u {r.h_u:.3f}, rev {r.acc_reversed:.3f}"
                                   for i, r in m.iterrows()))
    # runs of the extension that never reached L*: mark at their final state
    fin = df[(df["arm"] == "headlr") & (df["threshold"] == "final") & (~df["reached_star"])]
    for _, r in fin.iterrows():
        ax_c.plot(r["head_mult"], r["acc_reversed"], "^", mfc="none", mec="#d62728", ms=4)
        ax_c.plot(r["head_mult"], r["probe_acc_gain"], "o", mfc="none", mec="#2ca02c", ms=4)
        print(f"(c) kappa={r['head_mult']:g} did not reach L* (final loss {r['loss']:.3f}); open markers")
ax_c.set_xscale("log", base=16); ax_c.invert_xaxis()
ax_c.set_xlabel(r"whole-head rate multiplier $\kappa$ ($M=32$; slower $\to$)")
ax_c.set_ylabel(r"value at $L^\ast$"); ax_c.set_title("(c) relative speed of the head", loc="left")
ax_c.axhline(0.5, color="k", ls=":", lw=0.8); ax_c.grid(True, alpha=0.25); ax_c.legend(frameon=False, loc="center left")

# (d) h_u trajectories for sp, seed 0, three widths
cmap = plt.get_cmap("viridis")
widths_avail = sorted({int(p.stem.split("_M")[1].split("_")[0]) for p in (RES / "exp2").glob("traj_sp_M*_x1_s0.csv")
                       if "h_u" in pd.read_csv(p, nrows=1).columns})   # skip stale pilot files without h_u
pick = [w for w in (2, 32, 2048) if w in widths_avail] or widths_avail[:3]
for i, w in enumerate(pick):
    t = pd.read_csv(RES / "exp2" / f"traj_sp_M{w}_x1_s0.csv")
    col = cmap(i / max(1, len(pick) - 1))
    ax_d.plot(t["step"] + 1, t["h_u"], color=col, lw=1.4, label=f"$M={w}$")
    s = pd.read_csv(RES / "exp2" / f"snap_sp_M{w}_x1_s0.csv")
    s = s[s["threshold"].astype(str) == str(L_STAR)]
    if not s.empty:
        ax_d.plot(s["step"] + 1, s["h_u"], "o", color=col, ms=4, mec="k", mew=0.5)
ax_d.set_xscale("log"); ax_d.set_yscale("log")
ax_d.set_xlabel("gradient-descent step"); ax_d.set_ylabel(r"$h_u(W)=\|P_{\mathrm{row}(W)}u\|^2$")
ax_d.set_title(r"(d) trajectories (standard param., seed 0); $\bullet$ = $L^\ast$", loc="left")
ax_d.grid(True, alpha=0.25); ax_d.legend(frameon=False, loc="upper left")

fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3_exp2.{ext}", bbox_inches="tight")
print(f"saved {OUT / 'fig3_exp2.pdf'}  (runs in summary: {df.groupby(['arm','width','mult','head_mult','seed']).ngroups})")
