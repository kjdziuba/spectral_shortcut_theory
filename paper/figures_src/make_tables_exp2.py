"""Appendix C result tables for Experiment 2, generated from results/exp2_summary.csv.

Writes paper/sections_iclr_v2/appendix_exp2_tables.tex with:
  C1  values at L* = 0.30 by arm and width: probe gain, h_u, reversal acc.,
      context-random acc. (mean +- sd over seeds; n shown)
  C2  the same at the secondary thresholds 0.15 and 0.10 (probe gain, h_u, reversal)
  C3  whole-head rate kappa at M = 32 (per seed, with the seed-0 downward extension)
  C4  readout multiplier at M = 32 (mean +- sd)
  C5  runs that never reached L* (status, final loss, step)
  C6  registered verdicts (Spearman per seed) as computed by exp2_intervention.py
Also prints the means used in the main text. Re-run after the grid completes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results" / "exp2_summary.csv")
df["head_mult"] = df["head_mult"].fillna(1.0)
OUT = ROOT / "paper" / "sections_iclr_v2" / "appendix_exp2_tables.tex"
L = []


def ms(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x[0]:.3f}"
    return f"{x.mean():.3f}$\\pm${x.std(ddof=1):.3f}"


def spearman(x, y):
    xr = pd.Series(x).rank().values; yr = pd.Series(y).rank().values
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def at(thr):
    return df[df["threshold"].astype(str) == str(thr)]


def width_table(thr, label, caption, metrics):
    d = at(thr)
    d = d[(d["mult"] == 1) & (d["head_mult"] == 1)]
    L.append("\\begin{table}[htbp]\\centering\\scriptsize")
    L.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    cols = "ll" + "c" * len(metrics)
    L.append(f"\\begin{{tabular}}{{{cols}}}\\toprule")
    L.append("arm & $M$ & " + " & ".join(m[1] for m in metrics) + " \\\\ \\midrule")
    for arm in ("sp", "mup", "ctxfree", "frozen"):
        a = d[d["arm"] == arm]
        for w, g in a.groupby("width"):
            L.append(f"\\texttt{{{arm}}} & {int(w)} & " + " & ".join(ms(g[m[0]]) + f" ({len(g)})" for m in metrics) + " \\\\")
        if not a.empty:
            L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}\\end{table}")
    print(f"--- {label} (thr {thr}) means ---")
    for arm in ("sp", "mup", "ctxfree", "frozen"):
        a = d[d["arm"] == arm]
        if a.empty:
            continue
        print(arm, {int(w): tuple(round(float(g[m[0]].mean()), 3) for m in metrics) for w, g in a.groupby("width")})


metrics_star = [("probe_acc_gain", "probe gain"), ("h_u", "$h_u$"), ("acc_reversed", "reversed acc."),
                ("acc_ctx_random", "context-random acc."), ("acc_spec_only", "spectral-only acc.")]
width_table(0.3, "tab:exp2_star", "Experiment 2 at the pre-registered threshold $L^\\ast=0.30$: mean$\\pm$sd over seeds (number of seeds). "
            "Probe gain is the discriminant accuracy on the encoder output minus its initial value (0.647, 0.653, 0.608 for seeds 0--2); "
            "$h_u=\\|P_{\\mathrm{row}(W)}u\\|^2$ (0.055 at initialization).", metrics_star)
for thr, lab in ((0.15, "tab:exp2_015"), (0.10, "tab:exp2_010")):
    width_table(thr, lab, f"Experiment 2 at the secondary threshold {thr} (labelled; not pre-registered): mean$\\pm$sd over seeds that reached it.",
                [("probe_acc_gain", "probe gain"), ("h_u", "$h_u$"), ("acc_reversed", "reversed acc."), ("acc_ctx_random", "context-random acc.")])

# C3 kappa table (headlr), per seed, incl. runs that never reached L* (final state, marked)
L.append("\\begin{table}[htbp]\\centering\\scriptsize")
L.append("\\caption{Whole-head rate multiplier $\\kappa$ at $M=32$ (arm \\texttt{headlr}) at $L^\\ast$; rows marked $\\dagger$ never reached $L^\\ast$ within 40,000 steps and show the final state (loss in parentheses).}\\label{tab:exp2_kappa}")
L.append("\\begin{tabular}{llrccccc}\\toprule seed & $\\kappa$ & step & probe gain & $h_u$ & reversed & context-random & spectral-only \\\\ \\midrule")
h = df[df["arm"] == "headlr"]
for s, gs in h.groupby("seed"):
    for k, g in gs.groupby("head_mult"):
        r = g[g["threshold"].astype(str) == "0.3"]
        if r.empty:
            r = g[g["threshold"] == "final"]; mark = "$\\dagger$"; extra = f" ({float(r['loss'].iloc[0]):.3f})"
        else:
            mark = ""; extra = ""
        r = r.iloc[0]
        L.append(f"{s} & {k:g}{mark} & {int(r['step'])}{extra} & {r['probe_acc_gain']:.3f} & {r['h_u']:.3f} & {r['acc_reversed']:.3f} & {r['acc_ctx_random']:.3f} & {r['acc_spec_only']:.3f} \\\\")
L.append("\\bottomrule\\end{tabular}\\end{table}")
print("--- kappa (headlr) at L* or final ---")
print(h[h["threshold"].astype(str).isin(["0.3"])].groupby("head_mult")[["probe_acc_gain", "h_u", "acc_reversed"]].agg(["mean", "count"]).round(3))

# C4 readout multiplier
m = at(0.3); m = m[m["arm"] == "lrmult"]
L.append("\\begin{table}[htbp]\\centering\\scriptsize")
L.append("\\caption{Readout learning-rate multiplier at $M=32$ (arm \\texttt{lrmult}) at $L^\\ast$: mean$\\pm$sd over three seeds.}\\label{tab:exp2_lrmult}")
L.append("\\begin{tabular}{lcccc}\\toprule multiplier & probe gain & $h_u$ & reversed acc. & context-random acc. \\\\ \\midrule")
for k, g in m.groupby("mult"):
    L.append(f"{k:g} & {ms(g['probe_acc_gain'])} & {ms(g['h_u'])} & {ms(g['acc_reversed'])} & {ms(g['acc_ctx_random'])} \\\\")
L.append("\\bottomrule\\end{tabular}\\end{table}")
print("--- lrmult means ---"); print(m.groupby("mult")[["probe_acc_gain", "h_u", "acc_reversed"]].mean().round(3))

# C5 runs that never reached L*
miss = df[(df["threshold"] == "final") & (~df["reached_star"])]
L.append("\\paragraph{Runs that did not reach $L^\\ast$.}")
if miss.empty:
    L.append("Every run reached $L^\\ast$.")
else:
    L.append("The following runs ended at the step budget above $L^\\ast$: " + "; ".join(
        f"\\texttt{{{r['arm']}}} $M={int(r['width'])}$" + (f", $\\kappa={r['head_mult']:g}$" if r['head_mult'] != 1 else "")
        + f", seed {int(r['seed'])} (final loss {r['loss']:.3f} at step {int(r['step'])})" for _, r in miss.iterrows()) + ".")

# C6 registered verdicts: Spearman per seed at L*
L.append("\\paragraph{Registered verdicts.}")
d = at(0.3); d = d[(d["mult"] == 1) & (d["head_mult"] == 1)]
rows = []
for arm in ("sp", "mup", "ctxfree", "frozen"):
    for s, g in d[d["arm"] == arm].groupby("seed"):
        g = g.sort_values("width")
        rows.append(f"\\texttt{{{arm}}} seed {int(s)}: $\\rho(a_u,M)={spearman(g['width'], g['align_u']):+.2f}$, "
                    f"$\\rho(\\text{{probe}},M)={spearman(g['width'], g['probe_acc']):+.2f}$, $\\rho(\\text{{rev}},M)={spearman(g['width'], g['acc_reversed']):+.2f}$ "
                    f"over $M\\in\\{{{','.join(str(int(w)) for w in g['width'])}\\}}$")
L.append("Spearman correlations across widths at $L^\\ast$, per seed (P1--P3 as registered use $a_u$ and reversal accuracy): " + "; ".join(rows) + ".")
sp2048 = d[(d["arm"] == "sp") & (d["width"] == 2048)].set_index("seed")["align_u"]
mu2048 = d[(d["arm"] == "mup") & (d["width"] == 2048)].set_index("seed")["align_u"]
comp = [f"seed {int(s)}: {mu2048[s]:.4f} vs {sp2048[s]:.4f}" for s in mu2048.index if s in sp2048.index]
L.append("P2's second clause ($a_u$ at $M=2048$, normalized readout versus standard): " + "; ".join(comp) + ".")
for arm in ("sp", "mup", "ctxfree", "frozen"):
    a = d[d["arm"] == arm]
    print(arm, "per-seed rho(a_u,M):", [round(spearman(g.sort_values('width')['width'], g.sort_values('width')['align_u']), 2) for _, g in a.groupby('seed')],
          "rho(rev,M):", [round(spearman(g.sort_values('width')['width'], g.sort_values('width')['acc_reversed']), 2) for _, g in a.groupby('seed')])
print("P2 clause 2:", comp)

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}")
