"""Registered verdicts P12-P15 for the nonlinear-encoder arm of Experiment 2 (REFOCUS_PLAN section 10),
from results/exp2_summary.csv (arms nlenc, nlenc_ctxfree, nlenc_headlr, nlenc_frozen) and
results/exp2/recovery_summary_nl_init_nl_kappa1_nl_kappa256.csv.

Writes paper/sections_iclr_v2/appendix_exp2_nlenc_tables.tex (values at L* by arm and width; the
retraining table) and prints the per-seed verdicts. Re-run after the grid and the recovery runs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "sections_iclr_v2" / "appendix_exp2_nlenc_tables.tex"
df = pd.read_csv(ROOT / "results" / "exp2_summary.csv")
df = df[df["arm"].str.startswith("nlenc")]
star = df[df["threshold"].astype(str) == "0.3"]
init = df[df["threshold"].astype(str) == "init"]
L = []


def ms(x, nd=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return "--"
    return f"{x[0]:.{nd}f}" if len(x) == 1 else f"{x.mean():.{nd}f}$\\pm${x.std(ddof=1):.{nd}f}"


if star.empty:
    print("no nlenc runs at L* yet"); raise SystemExit
print(f"nlenc runs: {len(df[df['threshold'].astype(str) == 'final'])} finished, {len(star)} reached L*")
print("readiness (probe at init, spectral-only data):", init.groupby("arm")["probe_acc"].mean().round(3).to_dict())

# ---- table: values at L*
L.append("\\begin{table}[htbp]\\centering\\scriptsize")
L.append("\\caption{Experiment 2 with a two-layer ReLU encoder ($256\\to64\\to12$) at $L^\\ast=0.30$: mean$\\pm$sd over "
         "seeds (number of seeds). Probe gain = spectral-only probe accuracy minus its value at initialization; "
         "paired shifted accuracies as in Table~\\ref{tab:exp2_star}.}\\label{tab:exp2_nlenc}")
L.append("\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrcccccc}\\toprule")
L.append("arm & $M$ / $\\kappa$ & step & probe & probe gain & iid & reversed & ctx-random & spec.-only \\\\ \\midrule")
for arm in ("nlenc", "nlenc_headlr", "nlenc_ctxfree", "nlenc_frozen"):
    a = star[star["arm"] == arm]
    for (w, h), g in a.groupby(["width", "head_mult"]):
        lab = f"{int(w)}" + ("" if h == 1 else f" / 1/{round(1 / h)}")
        L.append(f"\\texttt{{{arm.replace(chr(95), chr(92) + chr(95))}}} & {lab} & {ms(g['step'], 0)} & {ms(g['probe_acc'])} & {ms(g['probe_acc_gain'])} & "
                 f"{ms(g['acc_iid'])} & {ms(g['acc_reversed'])} & {ms(g['acc_ctx_random'])} & {ms(g['acc_spec_only'])} ({len(g)}) \\\\")
    if not a.empty:
        L.append("\\midrule")
L[-1] = "\\bottomrule"
L.append("\\end{tabular}}\\end{table}")

# ---- verdicts
def by_seed(arm, width, hm=1.0, col="probe_acc_gain"):
    g = star[(star["arm"] == arm) & (star["width"] == width) & (np.isclose(star["head_mult"], hm))]
    return g.set_index("seed")[col].to_dict()

print("\n--- P12 (suppression persists): probe gain nlenc < nlenc_ctxfree at M=8 and M=128, every seed")
for w in (8, 128):
    a, c = by_seed("nlenc", w), by_seed("nlenc_ctxfree", w)
    common = sorted(set(a) & set(c))
    print(f"  M={w}: " + ", ".join(f"s{s}: {a[s]:+.3f} vs {c[s]:+.3f} {'OK' if a[s] < c[s] else 'FAIL'}" for s in common))
print("--- P13 (reliance persists): reversal nlenc <= 0.5 all widths/seeds; nlenc_ctxfree >= 0.75")
r = star[star["arm"] == "nlenc"]["acc_reversed"]; rc = star[star["arm"] == "nlenc_ctxfree"]["acc_reversed"]
print(f"  nlenc reversal max {r.max():.3f} ({'OK' if r.max() <= 0.5 else 'FAIL'}); ctxfree min {rc.min():.3f} ({'OK' if rc.min() >= 0.75 else 'FAIL'})")
print("--- P14 (rate response): probe gain kappa=1/256 > kappa=1 at M=32 every seed; mean reversal change < 0.05")
k1, k256 = by_seed("nlenc", 32), by_seed("nlenc_headlr", 32, 1 / 256)
rv1, rv256 = by_seed("nlenc", 32, col="acc_reversed"), by_seed("nlenc_headlr", 32, 1 / 256, col="acc_reversed")
common = sorted(set(k1) & set(k256))
print("  " + ", ".join(f"s{s}: {k1[s]:+.3f} -> {k256[s]:+.3f} {'OK' if k256[s] > k1[s] else 'FAIL'}" for s in common))
if common:
    d = np.mean([rv256[s] - rv1[s] for s in common]); print(f"  mean reversal change {d:+.3f} ({'OK' if abs(d) < 0.05 else 'FAIL'})")

rec_path = ROOT / "results" / "exp2" / "recovery_summary_nl_init_nl_kappa1_nl_kappa256.csv"
if rec_path.exists():
    rec = pd.read_csv(rec_path)
    print("--- P15 (recovery ordering): retrained reversal kappa256 > kappa1 every seed; kappa1 <= init + 0.05")
    piv = rec.pivot(index="seed", columns="encoder", values="retrained_acc_reversed")
    for s, row in piv.iterrows():
        ok1 = row["nl_kappa256"] > row["nl_kappa1"]; ok2 = row["nl_kappa1"] <= row["nl_init"] + 0.05
        print(f"  s{s}: init {row['nl_init']:.3f} k1 {row['nl_kappa1']:.3f} k256 {row['nl_kappa256']:.3f} "
              f"{'OK' if ok1 else 'FAIL'} / {'OK' if ok2 else 'FAIL'}")
    L.append("\\begin{table}[htbp]\\centering\\scriptsize")
    L.append("\\caption{Retraining a fresh paired-initialization head (20{,}000 GD steps on fresh context-random images) on "
             "frozen two-layer ReLU encoders saved at $L^\\ast$: shifted accuracies of the retrained classifier "
             "(mean$\\pm$sd over seeds).}\\label{tab:exp2_nlenc_recovery}")
    L.append("\\begin{tabular}{lccccc}\\toprule")
    L.append("encoder & probe & iid & reversed & ctx-random & spec.-only \\\\ \\midrule")
    for enc, name in (("nl_init", "random init"), ("nl_kappa1", "$\\kappa=1$ at $L^\\ast$"), ("nl_kappa256", "$\\kappa=1/256$ at $L^\\ast$")):
        g = rec[rec["encoder"] == enc]
        L.append(f"{name} & {ms(g['probe_acc'])} & {ms(g['retrained_acc_iid'])} & {ms(g['retrained_acc_reversed'])} & "
                 f"{ms(g['retrained_acc_ctx_random'])} & {ms(g['retrained_acc_spec_only'])} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")
OUT.write_text("% generated by paper/figures_src/evaluate_nlenc.py\n" + "\n".join(L) + "\n")
print(f"wrote {OUT}")
