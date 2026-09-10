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

# C4b recovery table (retraining the head on frozen L* encoders)
rec_path = ROOT / "results" / "exp2" / "recovery_summary.csv"
if rec_path.exists():
    r = pd.read_csv(rec_path)
    L.append("\\begin{table}[htbp]\\centering\\scriptsize")
    L.append("\\caption{Recovery at $M=32$: a fresh head with paired initialization trained for 20,000 steps on newly sampled "
             "context-random images with the encoder frozen at its $L^\\ast$ checkpoint (\\texttt{init} = the random initial encoder). "
             "Original = the run's own classifier at $L^\\ast$; retrained = the new head; accuracies on the same paired test family.}"
             "\\label{tab:exp2_recovery}")
    L.append("\\begin{tabular}{llccccccc}\\toprule seed & encoder & probe & retrain loss & orig.\\ reversed & orig.\\ ctx-random & retr.\\ iid & retr.\\ reversed & retr.\\ ctx-random \\\\ \\midrule")
    for _, x in r.sort_values(["seed", "encoder"]).iterrows():
        o_rev = f"{x['original_acc_reversed']:.3f}" if pd.notna(x.get("original_acc_reversed", float('nan'))) else "--"
        o_ctx = f"{x['original_acc_ctx_random']:.3f}" if pd.notna(x.get("original_acc_ctx_random", float('nan'))) else "--"
        L.append(f"{int(x['seed'])} & \\texttt{{{x['encoder']}}} & {x['probe_acc']:.3f} & {x['retrain_train_loss']:.3f} & {o_rev} & {o_ctx} & "
                 f"{x['retrained_acc_iid']:.3f} & {x['retrained_acc_reversed']:.3f} & {x['retrained_acc_ctx_random']:.3f} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")
    print("--- recovery means ---"); print(r.groupby("encoder")[["probe_acc", "retrained_acc_iid", "retrained_acc_reversed", "retrained_acc_ctx_random"]].mean().round(3))

# C5 runs that never reached L*
miss = df[(df["threshold"] == "final") & (~df["reached_star"])]
L.append("\\paragraph{Runs that did not reach $L^\\ast$.}")
if miss.empty:
    L.append("Every run reached $L^\\ast$.")
else:
    L.append("The following runs ended at the step budget above $L^\\ast$: " + "; ".join(
        f"\\texttt{{{r['arm']}}} $M={int(r['width'])}$" + (f", $\\kappa={r['head_mult']:g}$" if r['head_mult'] != 1 else "")
        + f", seed {int(r['seed'])} (final loss {r['loss']:.3f} at step {int(r['step'])})" for _, r in miss.iterrows()) + ".")

# C6 registered verdicts: criterion table on the original five widths and the amended grid
d = at(0.3); d = d[(d["mult"] == 1) & (d["head_mult"] == 1)]
ORIG = [8, 32, 128, 512, 2048]


def rhos(arm, widths, col):
    out = []
    for s, g in d[(d["arm"] == arm) & (d["width"].isin(widths))].groupby("seed"):
        g = g.sort_values("width"); out.append(spearman(g["width"], g[col]))
    return out


def fmt(v):
    return ", ".join(f"{x:+.2f}" for x in v)


L.append("\\begin{table}[htbp]\\centering\\scriptsize")
L.append("\\caption{Registered predictions (REFOCUS\\_PLAN \\S4.4) evaluated at $L^\\ast=0.30$ on the original five widths and on the amended grid; "
         "$\\rho$ = Spearman rank correlation across widths (or multipliers), one value per seed. P2's second clause compares $a_u$ at $M=2048$.}\\label{tab:exp2_verdicts}")
L.append("\\begin{tabular}{p{5.2cm}p{3.1cm}p{4.2cm}p{2.6cm}}\\toprule criterion & grid & per-seed values & verdict \\\\ \\midrule")
sp_au_o, sp_rev_o = rhos("sp", ORIG, "align_u"), rhos("sp", ORIG, "acc_reversed")
sp_au_a, sp_rev_a = rhos("sp", sorted(d["width"].unique()), "align_u"), rhos("sp", sorted(d["width"].unique()), "acc_reversed")
both_o = sum(1 for a, b in zip(sp_au_o, sp_rev_o) if a <= -0.8 and b <= -0.8)
L.append(f"P1: standard param., $\\rho(a_u,M)\\le-0.8$ and $\\rho(\\text{{rev}},M)\\le-0.8$ in every seed & original & $a_u$: {fmt(sp_au_o)}; rev: {fmt(sp_rev_o)} & not met ({both_o}/3 seeds) \\\\")
L.append(f" & amended & $a_u$: {fmt(sp_au_a)}; rev: {fmt(sp_rev_a)} & not met \\\\")
mu_au_o, mu_pr_o, mu_rev_o = rhos("mup", ORIG, "align_u"), rhos("mup", ORIG, "probe_acc"), rhos("mup", ORIG, "acc_reversed")
mu_au_a, mu_rev_a = rhos("mup", sorted(d["width"].unique()), "align_u"), rhos("mup", sorted(d["width"].unique()), "acc_reversed")
flat = lambda v: sum(1 for x in v if abs(x) < 0.5) >= 2
L.append(f"P2a: normalized readout, $|\\rho(\\cdot,M)|<0.5$ in at least two seeds & original & $a_u$: {fmt(mu_au_o)}; probe: {fmt(mu_pr_o)}; rev: {fmt(mu_rev_o)} & alignment {'met' if flat(mu_au_o) else 'not met'}; probe {'met' if flat(mu_pr_o) else 'not met'}; reversal {'met' if flat(mu_rev_o) else 'not met'} \\\\")
L.append(f" & amended & $a_u$: {fmt(mu_au_a)}; rev: {fmt(mu_rev_a)} & alignment {'met' if flat(mu_au_a) else 'not met'}; reversal {'met' if flat(mu_rev_a) else 'not met'} \\\\")
sp2048 = d[(d["arm"] == "sp") & (d["width"] == 2048)].set_index("seed")["align_u"]
mu2048 = d[(d["arm"] == "mup") & (d["width"] == 2048)].set_index("seed")["align_u"]
comp = [f"{mu2048[s]:.4f} vs {sp2048[s]:.4f}" for s in sorted(mu2048.index) if s in sp2048.index]
L.append(f"P2b: $a_u(2048)$ normalized $>$ standard in every seed & paired seeds & {'; '.join(comp)} & {'met' if all(mu2048[s] > sp2048[s] for s in mu2048.index if s in sp2048.index) else 'not met'} \\\\")
cf = d[d["arm"] == "ctxfree"]
cf_au = rhos("ctxfree", sorted(cf["width"].unique()), "align_u"); cf_pr = rhos("ctxfree", sorted(cf["width"].unique()), "probe_acc")
L.append(f"P3: uninformative context, $a_u>0.25$ at every width and $|\\rho(a_u,M)|<0.5$ & amended control widths $\\{{{','.join(str(int(w)) for w in sorted(cf['width'].unique()))}\\}}$ & "
         f"$a_u$ range {cf['align_u'].min():.3f}--{cf['align_u'].max():.3f}; $\\rho(a_u,M)$: {fmt(cf_au)}; $\\rho(\\text{{probe}},M)$: {fmt(cf_pr)} & cutoff not met; flatness not met (probe {cf.groupby('width')['probe_acc'].mean().iloc[0]:.3f}$\\to${cf.groupby('width')['probe_acc'].mean().iloc[-1]:.3f}) \\\\")
lm = at(0.3); lm = lm[lm["arm"] == "lrmult"]
lm_au = [spearman(g.sort_values("mult")["mult"], g.sort_values("mult")["align_u"]) for _, g in lm.groupby("seed")]
lm_rev = [spearman(g.sort_values("mult")["mult"], g.sort_values("mult")["acc_reversed"]) for _, g in lm.groupby("seed")]
L.append(f"P4: readout multiplier, $a_u$ and reversal accuracy decrease with the multiplier in every seed & run multipliers $\\{{1/16,1/4,1,4,16\\}}$ (64 dropped) & $a_u$: {fmt(lm_au)}; rev: {fmt(lm_rev)} & $a_u$ met; reversal not met (opposite sign) \\\\")
fr = d[d["arm"] == "frozen"]
L.append(f"P5: frozen encoder, reversal accuracy $<0.5$ at every width & amended widths & max {fr['acc_reversed'].max():.3f} & met \\\\")
hk = at(0.3); hk = hk[hk["arm"] == "headlr"]
hk_pr = [spearman(g.sort_values("head_mult")["head_mult"], g.sort_values("head_mult")["probe_acc"]) for _, g in hk.groupby("seed")]
hk_hu = [spearman(g.sort_values("head_mult")["head_mult"], g.sort_values("head_mult")["h_u"]) for _, g in hk.groupby("seed")]
hk_rev = [spearman(g.sort_values("head_mult")["head_mult"], g.sort_values("head_mult")["acc_reversed"]) for _, g in hk.groupby("seed")]
L.append(f"P6 (amendment 4.4b): whole-head $\\kappa$, probe, $h_u$ and reversal accuracy increase as $\\kappa$ decreases, $\\rho\\le-0.8$ in every seed & $\\kappa\\in\\{{1,1/16,1/256\\}}$, three seeds & probe: {fmt(hk_pr)}; $h_u$: {fmt(hk_hu)}; rev: {fmt(hk_rev)} & probe and $h_u$ met; reversal not met ({sum(1 for x in hk_rev if x <= -0.8)}/3 seeds) \\\\")
L.append("\\bottomrule\\end{tabular}\\end{table}")
print("verdict table: original-width sp a_u", fmt(sp_au_o), "rev", fmt(sp_rev_o), "| mup a_u", fmt(mu_au_o), "probe", fmt(mu_pr_o), "rev", fmt(mu_rev_o))
for arm in ("sp", "mup", "ctxfree", "frozen"):
    a = d[d["arm"] == arm]
    print(arm, "per-seed rho(a_u,M):", [round(spearman(g.sort_values('width')['width'], g.sort_values('width')['align_u']), 2) for _, g in a.groupby('seed')],
          "rho(rev,M):", [round(spearman(g.sort_values('width')['width'], g.sort_values('width')['acc_reversed']), 2) for _, g in a.groupby('seed')])
print("P2 clause 2:", comp)

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}")
