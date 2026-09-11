"""Five-fold extension tables (REFOCUS_PLAN section 12): the key contrasts of Experiment 3 and
Experiment 4 on tissue, per patient fold (fold 0 = the registered fold), with mean and range.

Inputs (fold 0 / fold f): results/exp3_summary.csv | results/exp3_fold{f}_summary.csv,
results/exp3[/_fold{f}]/recovery_summary.csv, results/exp4_summary.csv | results/exp4_fold{f}_summary.csv,
results/exp4[_fold{f}]/{pixel_sweep.csv,recovery_summary.csv}. Seeds 0-2 only (the fold protocol).
Writes paper/sections_iclr_v2/appendix_folds_tables.tex (tab:exp3_folds, tab:exp4_folds) and prints
the per-fold values. Folds without results are skipped.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "sections_iclr_v2" / "appendix_folds_tables.tex"
SEEDS = [0, 1, 2]
L = []


def paths(f):
    sfx = "" if f == 0 else f"_fold{f}"
    return dict(e3=ROOT / "results" / f"exp3{sfx}_summary.csv", e3r=ROOT / "results" / f"exp3{sfx}" / "recovery_summary.csv",
                e4=ROOT / "results" / f"exp4{sfx}_summary.csv", e4r=ROOT / "results" / f"exp4{sfx}" / "recovery_summary.csv",
                e4p=ROOT / "results" / f"exp4{sfx}" / "pixel_sweep.csv")


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def exp3_row(f):
    p = paths(f)
    if not p["e3"].exists():
        return None
    d = pd.read_csv(p["e3"]); d = d[d["seed"].isin(SEEDS)]
    st = d[d["threshold"].astype(str) == "0.3"]
    def m(reg, arm, col, hm=None):
        g = st[(st["regime"] == reg) & (st["arm"] == arm)]
        if hm is not None:
            g = g[np.isclose(g["head_mult"], hm)]
        return float(g[col].mean()) if len(g) else float("nan")
    def kappa_change(reg):
        g = st[(st["regime"] == reg) & (st["arm"] == "headlr")]
        a = g[np.isclose(g["head_mult"], 1.0)].set_index("seed")["probe_val_gain"]
        b = g[np.isclose(g["head_mult"], 1 / 256)].set_index("seed")["probe_val_gain"]
        common = a.index.intersection(b.index)
        return float((b[common] - a[common]).mean()) if len(common) else float("nan")
    row = dict(fold=f,
               u_gain_sp=m("unready", "sp", "probe_val_gain"), u_gain_cf=m("unready", "ctxfree", "probe_val_gain"),
               u_rev_sp=m("unready", "sp", "acc_reversed_val"), u_rev_cf=m("unready", "ctxfree", "acc_reversed_val"),
               u_kappa=kappa_change("unready"), u10_kappa=kappa_change("unready10"),
               r_rev_sp=m("ready", "sp", "acc_reversed_val"), r_rev_cf=m("ready", "ctxfree", "acc_reversed_val"))
    if p["e3r"].exists():
        r = pd.read_csv(p["e3r"]); r = r[r["seed"].isin(SEEDS)]
        def rec(reg, enc):
            g = r[(r["regime"] == reg) & (r["encoder"] == enc)]
            return float(g["retrained_acc_reversed_val"].mean()) if len(g) else float("nan")
        row.update(r_rec_init=rec("ready", "init"), r_rec_k1=rec("ready", "kappa1"), r_rec_k256=rec("ready", "kappa256"),
                   u_rec_k1=rec("unready", "kappa1"), u_rec_k256=rec("unready", "kappa256"))
    return row


def exp4_row(f):
    p = paths(f)
    if not p["e4"].exists():
        return None
    d = pd.read_csv(p["e4"]); d = d[d["seed"].isin(SEEDS)]
    fin = d[d["threshold"].astype(str) == "final"]
    def m(arm, col):
        g = fin[fin["arm"] == arm]
        return float(g[col].mean()) if len(g) else float("nan")
    row = dict(fold=f, iid_nat=m("nat", "f1_iid_val"), iid_nat16=m("nat16", "f1_iid_val"), iid_natfull=m("natfull", "f1_iid_val"),
               probe_nat=m("nat", "probe_val"), probe_shuf=m("shuf", "probe_val"),
               ctx_nat=m("nat", "f1_ctx_random_val"), ctx_shuf=m("shuf", "f1_ctx_random_val"))
    if p["e4p"].exists():
        sw = pd.read_csv(p["e4p"]); sw = sw[(sw["model"] == "mlp") & (sw["split"] == "val")]
        kmax = sw["k"].max()
        row["stageA_gain"] = float(sw[sw["k"] == kmax]["macro_f1"].mean() - sw[sw["k"] == 16]["macro_f1"].mean())
    if p["e4r"].exists():
        r = pd.read_csv(p["e4r"]); r = r[(r["status"] == "done") & (r["seed"].isin(SEEDS))]
        def rec(arm, enc):
            g = r[(r["arm"] == arm) & (r["encoder"].astype(str) == enc)]
            return float(g["f1_ctx_random_val"].mean()) if len(g) else float("nan")
        row.update(rec_nat=rec("nat", "final"), rec_shuf=rec("shuf", "final"), rec_init=rec("nat", "init"))
    return row


def table(rows, cols, label, caption):
    df = pd.DataFrame(rows).set_index("fold")
    L.append("\\begin{table}[htbp]\\centering\\scriptsize")
    L.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    L.append("\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * len(cols) + "}\\toprule")
    L.append("fold & " + " & ".join(c[1] for c in cols) + " \\\\ \\midrule")
    for f, r in df.iterrows():
        L.append(f"{int(f)}" + ("$^\\ast$" if f == 0 else "") + " & " + " & ".join(f3(r.get(c[0], float("nan"))) for c in cols) + " \\\\")
    L.append("\\midrule mean [min, max] & " + " & ".join(
        f"{df[c[0]].mean():.3f} [{df[c[0]].min():.2f}, {df[c[0]].max():.2f}]" if c[0] in df and df[c[0]].notna().any() else "--" for c in cols) + " \\\\")
    L.append("\\bottomrule\\end{tabular}}\\end{table}")
    print(df.round(3).to_string())


rows3 = [r for r in (exp3_row(f) for f in range(5)) if r]
rows4 = [r for r in (exp4_row(f) for f in range(5)) if r]
if rows3:
    table(rows3, [("u_gain_sp", "probe gain, inf.\\ ctx"), ("u_gain_cf", "probe gain, uninf.\\ ctx"), ("u_rev_sp", "reversal, inf."), ("u_rev_cf", "reversal, uninf."),
                  ("u_kappa", "$\\Delta$probe $\\kappa\\,1\\to1/256$ ($\\gamma=30$)"), ("u10_kappa", "same, $\\gamma=10$"),
                  ("r_rev_sp", "reversal, high acc."), ("r_rec_init", "retrained, random enc."), ("r_rec_k256", "retrained, $\\kappa=1/256$")],
          "tab:exp3_folds",
          "Experiment 3 across the five patient folds (fold $0^\\ast$ is the registered fold; validation patients; seeds 0--2; values at $L^\\ast$, "
          "means over widths and seeds). Low-accessibility regime unless stated; retrained = reversal accuracy of the retrained head in the high-accessibility regime.")
if rows4:
    table(rows4, [("stageA_gain", "per-pixel gain beyond 16 PCs (MLP)"), ("iid_nat", "iid, natural"), ("iid_nat16", "iid, natural 16 PCs"), ("iid_natfull", "iid, no bottleneck"),
                  ("probe_nat", "probe, natural"), ("probe_shuf", "probe, shuffled"), ("ctx_nat", "ctx-random, natural"), ("ctx_shuf", "ctx-random, shuffled"),
                  ("rec_nat", "retrained, natural enc."), ("rec_shuf", "retrained, shuffled enc."), ("rec_init", "retrained, random enc.")],
          "tab:exp4_folds",
          "Experiment 4 (non-denoised copy) across the five patient folds (fold $0^\\ast$ registered; validation macro-F1 at budget end; seeds 0--2).")
OUT.write_text("% generated by paper/figures_src/make_tables_folds.py\n" + "\n".join(L) + "\n")
print(f"wrote {OUT} ({len(rows3)} Exp 3 folds, {len(rows4)} Exp 4 folds)")
