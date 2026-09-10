"""Appendix D result tables for Experiment 3, generated from results/exp3_summary.csv.

Writes paper/sections_iclr_v2/appendix_exp3_tables.tex with, per regime:
  D1  values at L* = 0.30 by arm / width / kappa: step, probe gain (val), reversal,
      context-random and spectral-only accuracy on validation centres
      (mean +- sd over seeds; n), and the same accuracies on test centres.
  D2  runs that never reached L*.
Prints the means used in the main text. Re-run after the grid completes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results" / "exp3_summary.csv")
OUT = ROOT / "paper" / "sections_iclr_v2" / "appendix_exp3_tables.tex"
L = []


def ms(x, nd=3):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x[0]:.{nd}f}"
    return f"{x.mean():.{nd}f}$\\pm${x.std(ddof=1):.{nd}f}"


star = df[df["threshold"].astype(str) == "0.3"]
for regime, name in (("unready", "unready (whitened)"), ("ready", "ready (standardized)")):
    d = star[star["regime"] == regime]
    if d.empty:
        continue
    L.append("\\begin{table}[htbp]\\centering\\scriptsize")
    L.append(f"\\caption{{Experiment 3, {name} regime, at $L^\\ast=0.30$: mean$\\pm$sd over seeds (number of seeds). "
             "Probe gain = discriminant accuracy on the encoder output of validation centres minus its initial value; "
             "accuracies on validation (val) and test (test) patients under the paired conditions.}"
             f"\\label{{tab:exp3_{regime}}}")
    L.append("\\begin{tabular}{llrcccccc}\\toprule")
    L.append("arm & $M$ / $\\kappa$ & step & probe gain & rev.\\ (val) & ctx-rand.\\ (val) & spec.-only (val) & rev.\\ (test) & ctx-rand.\\ (test) \\\\ \\midrule")
    print(f"--- {regime} at L* ---")
    for arm in ("sp", "headlr", "ctxfree", "frozen"):
        a = d[d["arm"] == arm]
        for (w, h), g in a.groupby(["width", "head_mult"]):
            lab = f"{int(w)}" + ("" if h == 1 else f" / {h:g}")
            L.append(f"\\texttt{{{arm}}} & {lab} & {ms(g['step'], 0)} & {ms(g['probe_val_gain'])} & {ms(g['acc_reversed_val'])} & "
                     f"{ms(g['acc_ctx_random_val'])} & {ms(g['acc_spec_only_val'])} & {ms(g['acc_reversed_test'])} & {ms(g['acc_ctx_random_test'])} ({len(g)}) \\\\")
            print(f"{arm:8s} M={int(w):4d} k={h:g}: n={len(g)} step {g['step'].mean():.0f} probe_gain {g['probe_val_gain'].mean():+.3f} "
                  f"(probe {g['probe_val'].mean():.3f}) rev_val {g['acc_reversed_val'].mean():.3f} ctxrnd_val {g['acc_ctx_random_val'].mean():.3f} "
                  f"spec_val {g['acc_spec_only_val'].mean():.3f} iid_val {g['acc_iid_val'].mean():.3f} rev_test {g['acc_reversed_test'].mean():.3f}")
        if not a.empty:
            L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}\\end{table}")
    init = df[(df["regime"] == regime) & (df["threshold"].astype(str) == "init")]
    print(f"init probe_val {regime}: {init.groupby('seed')['probe_val'].first().round(3).to_dict()}")

miss = df[(df["threshold"] == "final") & (~df["reached_star"])]
L.append("\\paragraph{Runs that did not reach $L^\\ast$.}")
if miss.empty:
    L.append("Every run reached $L^\\ast$.")
else:
    L.append("; ".join(f"{r['regime']} \\texttt{{{r['arm']}}} $M={int(r['width'])}$" + (f", $\\kappa={r['head_mult']:g}$" if r['head_mult'] != 1 else "")
                       + f", seed {int(r['seed'])} (final loss {r['loss']:.3f} at step {int(r['step'])})" for _, r in miss.iterrows()) + ".")
    print("did not reach L*:", len(miss))
OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}")
