"""Appendix G result tables for Experiment 4 (natural context, usage test), generated from
results/exp4/pixel_sweep.csv, results/exp4_summary.csv, results/exp4/recovery_summary.csv and
results/exp4/neighbourhood_stats.json.

Writes paper/sections_iclr_v2/appendix_exp4_tables.tex with:
  G1  Stage A: per-pixel macro-F1 (validation / test patients) against the number of principal
      components for the shrinkage discriminant, balanced logistic regression and the MLP
      (mean +- sd over three seeds for the MLP).
  G2  Stage B at the matched threshold and at budget end, validation patients: step, probe and
      probe gain, macro-F1 under iid, context-random and homogeneous evaluation (mean +- sd; n).
  G3  the same on test patients.
  G4  head retraining on context-random training patches from the saved encoders.
Prints the numbers used in the text. Re-run after the grid and the recovery runs complete.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "sections_iclr_v2" / "appendix_exp4_tables.tex"
OUT_PAVIA = ROOT / "paper" / "sections_iclr_v2" / "appendix_exp4_paviau_tables.tex"
L = []
L_PAVIA = []
COPIES = (("exp4", "nodenoise", "non-denoised copy"), ("exp4_pca23", "pca23", "PCA-23-denoised copy"),
          ("exp4_paviau", "paviau", "Pavia University"))


def ms(x, nd=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x[0]:.{nd}f}"
    return f"{x.mean():.{nd}f}$\\pm${x.std(ddof=1):.{nd}f}"


# ------------------------------------------------------------------ per data copy
for RES_NAME, TAG, COPY_LABEL in COPIES:
    RES = ROOT / "results" / RES_NAME
    if not (RES / "pixel_sweep.csv").exists():
        continue
    if TAG == "paviau":            # Pavia tables go to their own file (Appendix H)
        L_BREAST, L = L, L_PAVIA
    print(f"===== {COPY_LABEL} =====")
    sw = pd.read_csv(RES / "pixel_sweep.csv")
    ks = sorted(sw["k"].unique())
    L.append("\\begin{table}[htbp]\\centering\\scriptsize")
    L.append(f"\\caption{{Experiment 4 ({COPY_LABEL}), Stage A: four-class macro-F1 of per-pixel classifiers on the centre spectra "
             "against the number of principal components $k$ (PCA on standardized training-split centres). Shrinkage Fisher discriminant (LDA), balanced logistic regression (LR) and a "
             "256-unit ReLU MLP with a fixed schedule (mean$\\pm$sd over three seeds), on validation and test patients.}"
             f"\\label{{tab:exp4_pixel_{TAG}}}")
    L.append("\\begin{tabular}{r" + "c" * 6 + "}\\toprule")
    L.append("$k$ & LDA (val) & LR (val) & MLP (val) & LDA (test) & LR (test) & MLP (test) \\\\ \\midrule")
    print("--- Stage A: macro-F1 by k ---")
    for k in ks:
        cells = []
        for split in ("val", "test"):
            for model in ("lda", "logreg", "mlp"):
                g = sw[(sw["k"] == k) & (sw["model"] == model) & (sw["split"] == split)]["macro_f1"]
                cells.append(ms(g))
        L.append(f"{int(k)} & " + " & ".join(cells) + " \\\\")
        print(f"k={int(k):4d}: " + "  ".join(f"{s}/{m}={sw[(sw['k']==k)&(sw['model']==m)&(sw['split']==s)]['macro_f1'].mean():.3f}"
                                             for s in ("val", "test") for m in ("lda", "logreg", "mlp")))
    L.append("\\bottomrule\\end{tabular}\\end{table}")
    stats = json.loads((RES / "neighbourhood_stats.json").read_text())
    print("neighbourhood stats:", {s: {k: round(v, 3) for k, v in d.items()} for s, d in stats.items()})

    # ------------------------------------------------------------------ G2/G3: Stage B
    summ_path = ROOT / "results" / f"{RES_NAME}_summary.csv"
    if summ_path.exists():
        df = pd.read_csv(summ_path)
        ARM_ORDER = ("natfull", "nat", "headlr", "nat16", "shuffull", "shuf", "shuf16", "frozen")
        ARM_NAME = {"nat": "natural", "headlr": "natural, $\\kappa=1/256$", "nat16": "natural, 16 PCs",
                    "shuf": "shuffled (comparator)", "shuf16": "shuffled, 16 PCs", "frozen": "natural, frozen encoder", "natfull": "natural, no bottleneck", "shuffull": "shuffled, no bottleneck"}
        for split, lab in (("val", "validation"), ("test", "test")):
            L.append("\\begin{table}[htbp]\\centering\\scriptsize")
            L.append(f"\\caption{{Experiment 4 ({COPY_LABEL}), Stage B, {lab} patients: mean$\\pm$sd over seeds (number of seeds) at the "
                     "matched threshold $L^\\ast$ and at budget end (40{,}000 steps or training loss $0.10$). Probe = "
                     "macro-F1 of a discriminant on the encoder output of centre spectra; gain relative to initialization. "
                     "Macro-F1 of the trained model under iid (natural), context-random (neighbours replaced by pixels "
                     "from other cores with independent labels) and homogeneous (all nine pixels equal to the centre) "
                     f"evaluation.}}\\label{{tab:exp4_{split}_{TAG}}}")
            L.append("\\begin{tabular}{llrccccc}\\toprule")
            L.append("arm & at & step & probe & probe gain & iid & ctx-random & homogeneous \\\\ \\midrule")
            for thr, thr_lab in (("0.3", "$L^\\ast$"), ("final", "end")):
                d = df[df["threshold"].astype(str) == thr]
                for arm in ARM_ORDER:
                    g = d[d["arm"] == arm]
                    if g.empty:
                        continue
                    L.append(f"{ARM_NAME[arm]} & {thr_lab} & {ms(g['step'], 0)} & {ms(g[f'probe_{split}'])} & "
                             f"{ms(g[f'probe_{split}_gain'])} & {ms(g[f'f1_iid_{split}'])} & {ms(g[f'f1_ctx_random_{split}'])} & "
                             f"{ms(g[f'f1_homogeneous_{split}'])} ({len(g)}) \\\\")
                    if split == "val":
                        print(f"[{thr:5s}] {arm:7s}: n={len(g)} step {g['step'].mean():.0f} probe {g['probe_val'].mean():.3f} "
                              f"(gain {g['probe_val_gain'].mean():+.3f}) iid {g['f1_iid_val'].mean():.3f} "
                              f"ctxrnd {g['f1_ctx_random_val'].mean():.3f} homog {g['f1_homogeneous_val'].mean():.3f} "
                              f"| test iid {g['f1_iid_test'].mean():.3f} ctxrnd {g['f1_ctx_random_test'].mean():.3f}")
                L.append("\\midrule")
            L[-1] = "\\bottomrule"
            L.append("\\end{tabular}\\end{table}")
        nr = df[(df["threshold"].astype(str) == "final") & (~df["reached_star"])]
        if not nr.empty:
            print("runs that did not reach L*:", nr[["arm", "seed", "step", "loss", "status"]].to_string(index=False))

    # ------------------------------------------------------------------ G4: recovery
    rec_path = RES / "recovery_summary.csv"
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        rec = rec[rec["status"] == "done"]
        ENC_NAME = {"init": "random init", "0.3": "at $L^\\ast$", "final": "at end"}
        L.append("\\begin{table}[htbp]\\centering\\scriptsize")
        L.append(f"\\caption{{Experiment 4 ({COPY_LABEL}), head retraining: a fresh paired-initialization head trained for 20{{,}}000 GD steps "
                 "on context-random training patches from a frozen saved encoder; macro-F1 under context-random and iid "
                 f"evaluation on validation and test patients (mean$\\pm$sd over seeds; n).}}\\label{{tab:exp4_recovery_{TAG}}}")
        L.append("\\begin{tabular}{llcccc}\\toprule")
        L.append("encoder from & saved & ctx-random (val) & iid (val) & ctx-random (test) & iid (test) \\\\ \\midrule")
        print("--- recovery ---")
        for arm in ("nat", "headlr", "nat16", "shuf"):
            for enc in ("init", "0.3", "final"):
                g = rec[(rec["arm"] == arm) & (rec["encoder"].astype(str) == enc)]
                if g.empty:
                    continue
                name = {"nat": "natural", "headlr": "natural, $\\kappa=1/256$", "nat16": "natural, 16 PCs", "shuf": "shuffled"}[arm]
                L.append(f"{name} & {ENC_NAME[enc]} & {ms(g['f1_ctx_random_val'])} & {ms(g['f1_iid_val'])} & "
                         f"{ms(g['f1_ctx_random_test'])} & {ms(g['f1_iid_test'])} ({len(g)}) \\\\")
                print(f"{arm:7s} enc={enc:5s}: n={len(g)} ctxrnd_val {g['f1_ctx_random_val'].mean():.3f} iid_val {g['f1_iid_val'].mean():.3f} "
                      f"ctxrnd_test {g['f1_ctx_random_test'].mean():.3f}")
        L.append("\\bottomrule\\end{tabular}\\end{table}")

if L_PAVIA:
    L = L_BREAST
OUT.write_text("% generated by paper/figures_src/make_tables_exp4.py\n" + "\n".join(L) + "\n")
print(f"wrote {OUT}")
if L_PAVIA:
    OUT_PAVIA.write_text("% generated by paper/figures_src/make_tables_exp4.py\n" + "\n".join(L_PAVIA) + "\n")
    print(f"wrote {OUT_PAVIA}")
