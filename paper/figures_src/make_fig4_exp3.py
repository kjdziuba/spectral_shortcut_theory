"""Fig 4 (v2 paper): Experiment 3 (real centre spectra, constructed context) at
matched fit L* = 0.30, two readiness regimes.

(a) spectral probe gain (validation centres) vs head width M, sp arm, both
    regimes, with the context-uninformative comparator (ctxfree) dashed.
(b) accuracy under context reversal (validation) vs M, both regimes, with
    ctxfree dashed: the failure criterion is excess shifted risk over ctxfree.
(c) whole-head rate kappa at M = 32: probe gain and reversal accuracy, both regimes.
(d) context-random accuracy (validation) vs M: what the model does from the
    spectrum alone in the presence of context.

Inputs:  results/exp3_summary.csv (via `exp3_train.py --collect`)
Outputs: paper/figures/fig4_exp3.{pdf,png}; prints the plotted numbers.
Tolerates partial grids.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
RES = ROOT / "results"
L_STAR = 0.30
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})
REG = {"unready": ("unready (whitened, $\\gamma=30$)", "#d62728"), "unready10": ("unready (whitened, $\\gamma=10$)", "#ff7f0e"),
       "ready": ("ready (standardized)", "#1f77b4")}

df = pd.read_csv(RES / "exp3_summary.csv")
star = df[df["threshold"].astype(str) == str(L_STAR)].copy()
fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.0))
(ax_a, ax_b), (ax_c, ax_d) = axes


def width_panel(ax, metric, ylabel, title, comparator=True):
    for regime, (label, col) in REG.items():
        a = star[(star["regime"] == regime) & (star["arm"] == "sp")]
        if not a.empty:
            for _, g in a.groupby("seed"):
                ax.plot(g.sort_values("width")["width"], g.sort_values("width")[metric], "o", color=col, ms=2.2, alpha=0.45)
            m = a.groupby("width")[metric].mean()
            ax.plot(m.index, m.values, "-", color=col, lw=1.6, label=label)
            print(f"{title} {regime} sp: " + ", ".join(f"M={int(w)}: {v:.3f}" for w, v in m.items()))
        if comparator:
            c = star[(star["regime"] == regime) & (star["arm"] == "ctxfree")]
            if not c.empty:
                m = c.groupby("width")[metric].mean()
                ax.plot(m.index, m.values, "--", color=col, lw=1.2, label=f"{label}, context uninformative")
                print(f"{title} {regime} ctxfree: " + ", ".join(f"M={int(w)}: {v:.3f}" for w, v in m.items()))
    ax.set_xscale("log", base=2); ax.set_xlabel(r"head width $M$"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left"); ax.grid(True, alpha=0.25)


width_panel(ax_a, "probe_val_gain", r"probe gain at $L^\ast$ (val.)", "(a) spectral learning at matched fit")
ax_a.axhline(0, color="k", lw=0.6); ax_a.legend(frameon=False, loc="best")
width_panel(ax_b, "acc_reversed_val", r"accuracy, context reversed (val.)", "(b) reliance: reversal accuracy")
ax_b.axhline(0.5, color="k", ls=":", lw=0.8); ax_b.set_ylim(-0.02, 1.02)
width_panel(ax_d, "acc_ctx_random_val", r"accuracy, context random (val.)", "(d) spectrum in the presence of context")
ax_d.axhline(0.5, color="k", ls=":", lw=0.8); ax_d.set_ylim(0.4, 1.02)

for regime, (label, col) in REG.items():
    k = star[(star["regime"] == regime) & (star["arm"] == "headlr")].sort_values("head_mult")
    if k.empty:
        continue
    for _, g in k.groupby("seed"):
        g = g.sort_values("head_mult")
        ax_c.plot(g["head_mult"], g["probe_val_gain"], "o", color=col, ms=2.2, alpha=0.45)
        ax_c.plot(g["head_mult"], g["acc_reversed_val"], "^", color=col, ms=2.2, alpha=0.45)
    m = k.groupby("head_mult")[["probe_val_gain", "acc_reversed_val"]].mean()
    ax_c.plot(m.index, m["probe_val_gain"], "-", color=col, lw=1.6, label=f"{label}: probe gain")
    ax_c.plot(m.index, m["acc_reversed_val"], "--", color=col, lw=1.6, label=f"{label}: reversal acc.")
    print(f"(c) {regime} kappa: " + "; ".join(f"k={i:g}: gain {r.probe_val_gain:.3f}, rev {r.acc_reversed_val:.3f}"
                                             for i, r in m.iterrows()))
ax_c.set_xscale("log", base=16); ax_c.invert_xaxis()
kt = sorted(set(star[star["arm"] == "headlr"]["head_mult"]), reverse=True) or [1, 1 / 16, 1 / 256]
ax_c.set_xticks(kt); ax_c.set_xticklabels(["1" if v == 1 else f"1/{round(1 / v)}" for v in kt]); ax_c.set_xticks([], minor=True)
ax_c.set_xlabel(r"whole-head rate multiplier $\kappa$ ($M=32$; slower $\to$)"); ax_c.set_ylabel(r"value at $L^\ast$ (val.)")
ax_c.set_title("(c) relative speed of the head", loc="left"); ax_c.axhline(0.5, color="k", ls=":", lw=0.8)
ax_c.grid(True, alpha=0.25); ax_c.legend(frameon=False, loc="best")

fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig4_exp3.{ext}", bbox_inches="tight")
miss = df[(df["threshold"] == "final") & (~df["reached_star"])]
for _, r in miss.iterrows():
    print(f"did not reach L*: {r['regime']} {r['arm']} M={r['width']} k={r['head_mult']:g} s{r['seed']} (final loss {r['loss']:.3f})")
print(f"saved {OUT / 'fig4_exp3.pdf'}  (runs: {df.groupby(['regime','arm','width','head_mult','seed']).ngroups})")
