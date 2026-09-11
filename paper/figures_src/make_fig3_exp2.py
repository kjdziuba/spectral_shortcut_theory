"""Fig 3 (v2 paper): Experiment 2 at matched fit L* = 0.30.

(a) spectral probe gain over initialization vs head width M for sp (standard
    parameterization), mup (normalized readout), ctxfree (uninformative
    training context) and frozen; seeds as points, means as lines.
(b) accuracy under context reversal vs M, same arms.
(c) whole-head rate multiplier kappa at M = 32: probe gain, h_u and reversal
    accuracy (seed points + means).
(d) recovery (Astra review 03, W1): reversal accuracy of a fresh whole head
    retrained for 20,000 steps on context-random images from frozen encoders
    taken at L* (random initialization, kappa = 1, kappa = 1/256; M = 32),
    against the original classifiers' reversal accuracy; two facets, linear
    encoder (registered seeds 0-2 filled, added seeds 3-4 open) and two-layer
    ReLU encoder (seeds 0-2). Mean paired slow-minus-fast gain annotated.

The h_u trajectory panel that was (d) until review 03 is written separately as
paper/figures/figC_exp2_traj.{pdf,png} for Appendix C.

Inputs:  results/exp2_summary.csv (via `exp2_intervention.py --collect`),
         results/exp2/recovery_summary.csv, results/exp2/recovery_summary_seeds3_4.csv,
         results/exp2/recovery_summary_nl_init_nl_kappa1_nl_kappa256.csv,
         results/exp2/traj_sp_M*_x1_s0.csv, results/exp2/snap_sp_M*_x1_s0.csv
Outputs: paper/figures/fig3_exp2.{pdf,png}, paper/figures/figC_exp2_traj.{pdf,png};
         prints the plotted numbers.
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
df = df[~df["arm"].astype(str).str.startswith("nlenc")]          # linear-encoder grid only in (a)-(c)
star = df[df["threshold"].astype(str) == str(L_STAR)].copy()
base = star[(star["mult"] == 1) & (star["head_mult"] == 1)]

fig = plt.figure(figsize=(5.6, 3.3))
gs = fig.add_gridspec(2, 2)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
sub = gs[1, 1].subgridspec(1, 2, wspace=0.10)
ax_d1 = fig.add_subplot(sub[0])
ax_d2 = fig.add_subplot(sub[1], sharey=ax_d1)


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
fin = df[(df["arm"] == "headlr") & (df["threshold"] == "final") & (~df["reached_star"])]
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
    for _, r in fin.iterrows():
        ax_c.plot(r["head_mult"], r["acc_reversed"], "^", mfc="none", mec="#d62728", ms=4)
        ax_c.plot(r["head_mult"], r["probe_acc_gain"], "o", mfc="none", mec="#2ca02c", ms=4)
        ax_c.annotate(f"loss {r['loss']:.2f}\n(n=1)", (r["head_mult"], r["probe_acc_gain"]), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=5, color="0.35")
        print(f"(c) kappa={r['head_mult']:g} did not reach L* (final loss {r['loss']:.3f}); open markers")
    if not fin.empty:
        ax_c.plot([], [], "o", mfc="none", mec="0.35", ms=4, label="budget endpoint, not matched")
ax_c.set_xscale("log", base=16); ax_c.invert_xaxis()
kt = sorted(set(k["head_mult"]) | set(fin["head_mult"]) if not k.empty else [1, 1 / 16, 1 / 256], reverse=True)
ax_c.set_xticks(kt); ax_c.set_xticklabels(["1" if v == 1 else f"1/{round(1 / v)}" for v in kt]); ax_c.set_xticks([], minor=True)
ax_c.set_xlabel(r"whole-head rate multiplier $\kappa$ ($M=32$; slower $\to$)")
ax_c.set_ylabel(r"value at $L^\ast$"); ax_c.set_title("(c) relative speed of the head", loc="left")
ax_c.axhline(0.5, color="k", ls=":", lw=0.8); ax_c.grid(True, alpha=0.25); ax_c.legend(frameon=False, loc="center left")

# (d) recovery: fresh whole head retrained from frozen L* encoders, reversal accuracy
ENC_ORDER = ["init", "kappa1", "kappa256"]
ENC_LABEL = ["random\ninit.", r"$\kappa=1$", r"$\kappa=1/256$"]
ENC_COL = {"init": "#7f7f7f", "kappa1": "#d62728", "kappa256": "#2ca02c"}


def load_recovery(path, prefix=""):
    if not path.exists():
        return pd.DataFrame()
    r = pd.read_csv(path)
    r["encoder"] = r["encoder"].astype(str).str.replace(prefix, "", regex=False)
    return r[r["encoder"].isin(ENC_ORDER)]


rec_lin = load_recovery(RES / "exp2" / "recovery_summary.csv")
rec_lin_added = load_recovery(RES / "exp2" / "recovery_summary_seeds3_4.csv")
rec_nl = load_recovery(RES / "exp2" / "recovery_summary_nl_init_nl_kappa1_nl_kappa256.csv", prefix="nl_")


def recovery_facet(ax, groups, title, show_ylabel):
    """groups: list of (df, filled: bool, label). Points per seed, mean line over the registered group."""
    xs = np.arange(3)
    any_orig = False
    for r, filled, label in groups:
        if r.empty:
            continue
        for s, g in r.groupby("seed"):
            g = g.set_index("encoder").reindex(ENC_ORDER)
            ax.plot(xs, g["retrained_acc_reversed"], "o", ms=2.6, alpha=0.75,
                    mfc=("k" if filled else "none"), mec="k", mew=0.6, zorder=3)
            o = g["original_acc_reversed"]
            ax.plot(xs[1:], o.values[1:], "x", ms=3.2, color="#d62728", mew=0.8, zorder=3)
            any_orig = True
    reg = groups[0][0]
    if not reg.empty:
        m = reg.groupby("encoder")["retrained_acc_reversed"].mean().reindex(ENC_ORDER)
        ax.plot(xs, m.values, "-", color="k", lw=1.4, zorder=2)
        for x, v, e in zip(xs, m.values, ENC_ORDER):
            ax.plot([x], [v], "s", ms=3.5, color=ENC_COL[e], zorder=4)
        # paired slow-minus-fast gain
        piv = reg.pivot(index="seed", columns="encoder", values="retrained_acc_reversed")
        gain = (piv["kappa256"] - piv["kappa1"])
        allrows = pd.concat([g for g, _, _ in groups if not g.empty])
        pivall = allrows.pivot(index="seed", columns="encoder", values="retrained_acc_reversed")
        gainall = pivall["kappa256"] - pivall["kappa1"]
        ax.annotate(f"slow $-$ fast: $+{gain.mean():.2f}$",
                    (2.0, m["kappa256"] + 0.09), ha="right", fontsize=5.5, color="0.2")
        print(f"(d) {title}: retrained reversal means " + ", ".join(f"{e} {v:.3f}" for e, v in m.items())
              + f"; paired slow-fast per seed (registered) {[round(x, 3) for x in gain.tolist()]} mean {gain.mean():.3f}"
              + (f"; all seeds {[round(x, 3) for x in gainall.tolist()]} mean {gainall.mean():.3f}" if len(gainall) > len(gain) else "")
              + f"; fast-random per seed {[round(x, 3) for x in (piv['kappa1'] - piv['init']).tolist()]}"
              + f"; original reversal fast {piv.index.map(lambda s: round(float(reg[(reg.seed == s) & (reg.encoder == 'kappa1')]['original_acc_reversed'].iloc[0]), 3)).tolist()}")
    ax.set_xticks(xs); ax.set_xticklabels(ENC_LABEL)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.grid(True, alpha=0.25)
    ax.set_title(title, loc="left")
    if show_ylabel:
        ax.set_ylabel("accuracy, context reversed")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    return any_orig


recovery_facet(ax_d1, [(rec_lin, True, "seeds 0-2"), (rec_lin_added, False, "seeds 3-4")],
               "(d) recovery: linear", True)
recovery_facet(ax_d2, [(rec_nl, True, "seeds 0-2")], "two-layer ReLU", False)
ax_d1.plot([], [], "o", ms=2.6, mfc="k", mec="k", label="retrained, seeds 0--2")
ax_d1.plot([], [], "o", ms=2.6, mfc="none", mec="k", label="retrained, seeds 3--4")
ax_d1.plot([], [], "x", ms=3.2, color="#d62728", label="original classifier")
ax_d1.legend(frameon=False, loc="center", bbox_to_anchor=(0.55, 0.36), handletextpad=0.3, fontsize=5)

fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3_exp2.{ext}", bbox_inches="tight")
print(f"saved {OUT / 'fig3_exp2.pdf'}  (runs in summary: {df.groupby(['arm','width','mult','head_mult','seed']).ngroups})")

# Appendix C figure: h_u trajectories for sp, seed 0, three widths (formerly Fig 3(d))
figt, ax_t = plt.subplots(figsize=(3.0, 2.1))
cmap = plt.get_cmap("viridis")
widths_avail = sorted({int(p.stem.split("_M")[1].split("_")[0]) for p in (RES / "exp2").glob("traj_sp_M*_x1_s0.csv")
                       if "h_u" in pd.read_csv(p, nrows=1).columns})   # skip stale pilot files without h_u
pick = [w for w in (2, 32, 2048) if w in widths_avail] or widths_avail[:3]
for i, w in enumerate(pick):
    t = pd.read_csv(RES / "exp2" / f"traj_sp_M{w}_x1_s0.csv")
    col = cmap(i / max(1, len(pick) - 1))
    ax_t.plot(t["step"] + 1, t["h_u"], color=col, lw=1.4, label=f"$M={w}$")
    s = pd.read_csv(RES / "exp2" / f"snap_sp_M{w}_x1_s0.csv")
    s = s[s["threshold"].astype(str) == str(L_STAR)]
    if not s.empty:
        ax_t.plot(s["step"] + 1, s["h_u"], "o", color=col, ms=4, mec="k", mew=0.5)
ax_t.set_xscale("log"); ax_t.set_yscale("log")
ax_t.set_xlabel("gradient-descent step"); ax_t.set_ylabel(r"$h_u(W)=\|P_{\mathrm{row}(W)}u\|^2$")
ax_t.set_title(r"standard param., seed 0; $\bullet$ = $L^\ast$", loc="left")
ax_t.grid(True, alpha=0.25); ax_t.legend(frameon=False, loc="upper left")
figt.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    figt.savefig(OUT / f"figC_exp2_traj.{ext}", bbox_inches="tight")
print(f"saved {OUT / 'figC_exp2_traj.pdf'}")
