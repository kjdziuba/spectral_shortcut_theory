"""Print the arm x width BN-recalibration table from results/e3c_bn_recal.csv.

Separate from the measurement script so the table can be re-rendered without
re-running the GPU passes. Adds the across-seed SD, which is the quantity the
per-checkpoint log lines cannot show: the joint arm's final-epoch val F1 is a
draw from a wildly oscillating sequence, so a drop in SD after recalibration is
itself evidence for the BatchNorm-mismatch hypothesis (M1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THEORY_ROOT = Path(__file__).resolve().parents[2]
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else THEORY_ROOT / "results" / "e3c_bn_recal.csv"
ARMS = ["joint_linear", "frozen_random", "frozen_pca"]
CLASSES = ["cancer_epi", "cas", "normal_stroma", "normal_epi"]

df = pd.read_csv(CSV)
print(f"[read] {CSV}  ({len(df)} checkpoints)")

bad = df[df.asis_csv_absdiff > 1e-6]
print(f"\nSANITY  max |f1_asis - epochs.csv epoch-60| = {df.asis_csv_absdiff.max():.3e}"
      f"   ({len(bad)} of {len(df)} checkpoints off by >1e-6)")

print("\n" + "=" * 92)
print("VAL MACRO-F1: as-is (stored BN stats) vs BN-recalibrated on the train set")
print("=" * 92)
hdr = (f"{'arm':<15}{'w':>5}{'n':>3}{'as-is':>20}{'recal':>20}"
       f"{'delta':>9}{'peak(csv)':>11}{'recov':>8}")
print(hdr)
rows = []
for arm in ARMS:
    for w in sorted(df.width.unique()):
        s = df[(df.arm == arm) & (df.width == w)]
        if not len(s):
            continue
        a, asd = s.f1_asis.mean(), s.f1_asis.std(ddof=1)
        b, bsd = s.f1_recal.mean(), s.f1_recal.std(ddof=1)
        rows.append(dict(arm=arm, width=w, n=len(s), asis=a, asis_sd=asd,
                         recal=b, recal_sd=bsd, peak=s.f1_peak_csv.mean(),
                         recov=s.recal_recovery_frac.mean()))
        print(f"{arm:<15}{w:>5}{len(s):>3}"
              f"{f'{a:.4f} +/- {asd:.4f}':>20}{f'{b:.4f} +/- {bsd:.4f}':>20}"
              f"{b - a:>+9.4f}{s.f1_peak_csv.mean():>11.4f}{s.recal_recovery_frac.mean():>8.2f}")
R = pd.DataFrame(rows)

print("\nJOINT minus FROZEN gap in mean val macro-F1")
print(f"{'comparison':<38}{'as-is':>10}{'recal':>10}{'change':>10}")
for w in sorted(df.width.unique()):
    j = R[(R.arm == "joint_linear") & (R.width == w)]
    if not len(j):
        continue
    for fr in ("frozen_random", "frozen_pca"):
        f = R[(R.arm == fr) & (R.width == w)]
        if not len(f):
            continue
        ga = float(j.asis.iloc[0] - f.asis.iloc[0])
        gb = float(j.recal.iloc[0] - f.recal.iloc[0])
        print(f"{f'h{w}: joint_linear - {fr}':<38}{ga:>+10.4f}{gb:>+10.4f}{gb - ga:>+10.4f}")

print("\nPER-CLASS mean F1 (as-is -> recal)")
print(f"{'arm':<15}{'w':>5}" + "".join(f"{c:>26}" for c in CLASSES))
for arm in ARMS:
    for w in sorted(df.width.unique()):
        s = df[(df.arm == arm) & (df.width == w)]
        if not len(s):
            continue
        cells = "".join(
            f"{f'{s[f'asis_f1_{c}'].mean():.3f}->{s[f'recal_f1_{c}'].mean():.3f}':>26}"
            for c in CLASSES)
        print(f"{arm:<15}{w:>5}{cells}")

print("\nPER-CHECKPOINT")
show = ["arm", "width", "seed", "f1_asis", "f1_epoch60_csv", "f1_recal",
        "delta", "f1_peak_csv", "peak_epoch", "recal_recovery_frac"]
print(df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("=" * 92)
