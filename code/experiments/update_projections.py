#!/usr/bin/env python3
"""
Encoder-update projections (Astra review_iclr_02 §V): for every saved encoder
checkpoint pair (W0, W at L*), the fraction of the squared net update that lies
in the constructed-cue span and in the spectral direction.

Exp 3: R_C = ||(W* - W0) C^T||_F^2 / ||W* - W0||_F^2, C = eight orthonormal cue rows.
Exp 2: R_V = ||(W* - W0) V^T||_F^2 / ||W* - W0||_F^2 (eight context directions) and
       R_u = ||(W* - W0) u||^2 / ||W* - W0||_F^2 (spectral direction).
These describe where the NET adaptation went; they are not a pathwise gradient
decomposition. Outputs results/exp3/update_projections.csv, results/exp2/update_projections.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def parse(tag: str):
    # e.g. unready_headlr_M32_h0.00390625_s1 or sp_M2048_x1_s0
    parts = tag.split("_")
    d = dict(tag=tag, seed=int(parts[-1][1:]), head_mult=1.0, mult=1.0)
    for p in parts:
        if p.startswith("M") and p[1:].isdigit():
            d["width"] = int(p[1:])
        elif p.startswith("h") and p[1:].replace(".", "").replace("e-", "").isdigit():
            d["head_mult"] = float(p[1:])
        elif p.startswith("x"):
            try:
                d["mult"] = float(p[1:])
            except ValueError:
                pass
    return d


rows3 = []
for f in sorted((ROOT / "results" / "exp3").glob("enc_*.npz")):
    z = np.load(f)
    if "W_0.3" not in z.files:
        continue
    d = parse(f.stem[4:]); d["regime"] = f.stem.split("_")[1]; d["arm"] = f.stem.split("_")[2]
    dW = z["W_0.3"] - z["W0"]; C = z["C"]
    den = float((dW ** 2).sum())
    d["R_C"] = float(((dW @ C.T) ** 2).sum() / den) if den > 0 else float("nan")
    d["update_norm"] = float(np.sqrt(den))
    rows3.append(d)
df3 = pd.DataFrame(rows3)
df3.to_csv(ROOT / "results" / "exp3" / "update_projections.csv", index=False)
print("Exp 3 R_C (fraction of squared net update in the eight cue directions) at L*:")
print(df3[df3["arm"] != "frozen"].groupby(["regime", "arm", "head_mult"])["R_C"].agg(["min", "max", "count"]).round(6))

rows2 = []
for f in sorted((ROOT / "results" / "exp2").glob("enc_*.npz")):
    z = np.load(f)
    if "W_0.3" not in z.files:
        continue
    d = parse(f.stem[4:]); d["arm"] = f.stem.split("_")[1]
    dW = z["W_0.3"] - z["W0"]; u = z["u"]; V = z["V"]
    den = float((dW ** 2).sum())
    d["R_u"] = float(((dW @ u) ** 2).sum() / den) if den > 0 else float("nan")
    d["R_V"] = float(((dW @ V.T) ** 2).sum() / den) if den > 0 else float("nan")
    d["update_norm"] = float(np.sqrt(den))
    rows2.append(d)
df2 = pd.DataFrame(rows2)
df2.to_csv(ROOT / "results" / "exp2" / "update_projections.csv", index=False)
print("\nExp 2 at L*: R_u (spectral direction) and R_V (eight context directions), means over seeds:")
print(df2[df2["arm"] != "frozen"].groupby(["arm", "width", "head_mult", "mult"])[["R_u", "R_V"]].mean().round(4).to_string())
