"""Render the full-contrast directional-curvature table (reviewer B2).

Reads results/exp1_8b_directions.csv (written incrementally by
exp1_8b_spectrum.py --full_contrast_directions) so the table can be produced
without re-running, and regardless of where the run was stopped.

Columns are the exact lambda_max of the K x K GGN block restricted to
{u (x) w : u in R^K} for four input directions w:
  v1     leading eigenvector of the post-BN input Gram
  d      un-orthogonalized unit class contrast mu_a - mu_b
  d_perp d with its v1 component removed, renormalized
  mean   post-BN empirical mean spectrum over valid pixels
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

THEORY_ROOT = Path(__file__).resolve().parents[2]
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    THEORY_ROOT / "results" / "exp1_8b_directions.csv"

dd = pd.read_csv(CSV)
print(f"[read] {CSV}  ({len(dd)} rows)")

print("\n" + "=" * 108)
print("RESTRICTED CURVATURE  max_u R(u (x) w)  BY INPUT DIRECTION w   "
      "(breast fold0, M=192, GGN theta block)")
print("=" * 108)
cols = ["seed", "batch", "pair", "n_valid_px", "lam_v1", "lam_d_full",
        "lam_d_perp", "lam_mean", "ratio_v1_over_dfull",
        "ratio_v1_over_dperp", "ratio_v1_over_mean"]
print(dd[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

print("\n" + "-" * 108)
print("|cos| OVERLAP MATRIX among {d, d_perp, mean, v1}")
print("-" * 108)
ocols = ["seed", "batch", "pair", "cos_d_v1", "cos_d_mean", "cos_d_dperp",
         "cos_dperp_v1", "cos_dperp_mean", "cos_mean_v1"]
print(dd[ocols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

fin = dd.dropna(subset=["lam_d_full"])
print("\n" + "-" * 108)
print(f"MEANS over the {len(fin)} rows that have a class contrast "
      f"(of {len(dd)} rows total)")
print("-" * 108)
if len(fin):
    for c in ["ratio_v1_over_dfull", "ratio_v1_over_dperp", "ratio_v1_over_mean"]:
        print(f"  {c:<22} mean={fin[c].mean():>10.4g}  "
              f"min={fin[c].min():>10.4g}  max={fin[c].max():>10.4g}")
    print(f"  {'cos_d_v1':<22} mean={fin.cos_d_v1.mean():>10.4g}")
    print(f"  {'cos_mean_v1':<22} mean={fin.cos_mean_v1.mean():>10.4g}")
    print(f"  {'cos_dperp_v1 (should be ~0)':<22} "
          f"max={fin.cos_dperp_v1.max():.3e}")
print(f"  {'ratio_v1_over_mean (all rows)':<22} "
      f"mean={dd.ratio_v1_over_mean.mean():.4g}")

print("\nCLASS COUNTS PER BATCH (why some rows have no contrast)")
print(dd[["seed", "batch", "pair", "n_a", "n_b", "class_counts"]]
      .drop_duplicates(subset=["seed", "batch", "pair"])
      .to_string(index=False))
print("=" * 108)
