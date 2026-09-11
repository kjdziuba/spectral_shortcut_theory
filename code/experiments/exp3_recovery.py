#!/usr/bin/env python3
"""
Experiment 3 recovery (real-spectrum version of exp2_recovery.py; Astra review 02):
retrain the patch head on FROZEN matched-fit encoders and measure shifted accuracy
on validation and test patients with context present.

Protocol (fixed before inspecting shifted results), per regime in {unready (gamma=30),
unready10 (gamma=10), ready} and seed s in {0,1,2}, M = 32:
  encoders: init = W0; kappa1 = W at L* of the kappa=1 run (enc_{regime}_headlr_M32_s{s}.npz["W_0.3"]);
            kappa256 = W at L* of the kappa=1/256 run (enc_{regime}_headlr_M32_h0.00390625_s{s}.npz["W_0.3"])
  head: PatchHead(K, 32), paired initialization torch.manual_seed(100+s) for the three encoders of a seed
  data: freshly sampled context-random training patches (seed 8500+s, same centres/donor pool rules as
        training; condition "ctx_random"), full-batch GD eta=1e-3 on the head only, budget 20,000 steps
        or training loss < 0.10; evaluated state = final
  evaluation: the same paired validation/test families as the original runs (seed 9500+s):
        iid, reversed, ctx_random, spec_only; plus the probe on the frozen encoder.
Output: results/exp3/recovery_summary.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from experiments.exp3_train import (  # noqa: E402
    OUT_DIR, PAIR, S_FEAT, K, N_TRAIN_PATCH, N_PROBE_FIT, N_EVAL_MAX, CONDITIONS, Preproc, Encoder, PatchHead,
    make_patches, margin_loss, load_pair, lda_fit, lda_eval,
)

import os
LR = 1e-3; BUDGET = 20_000; STOP_LOSS = 0.10; HEAD_WIDTH = 32
SEEDS = [int(x) for x in os.environ.get("EXP3_SEEDS", "0 1 2").split()]   # §11: EXP3_SEEDS="3 4" for the added seeds
OUT_NAME = "recovery_summary.csv" if "EXP3_SEEDS" not in os.environ else "recovery_summary_seeds" + "_".join(str(x) for x in SEEDS) + ".csv"
REGIMES = ["unready", "unready10", "ready"]
ENC = {"init": ("enc_{r}_headlr_M32_s{s}.npz", "W0"), "kappa1": ("enc_{r}_headlr_M32_s{s}.npz", "W_0.3"),
       "kappa256": ("enc_{r}_headlr_M32_h0.00390625_s{s}.npz", "W_0.3")}
ORIG = {"init": None, "kappa1": "{r}_headlr_M32_s{s}", "kappa256": "{r}_headlr_M32_h0.00390625_s{s}"}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cal = json.loads((OUT_DIR / "calibration3.json").read_text())
    Xn, yn, core, side = load_pair(PAIR)
    X = torch.from_numpy(Xn); y = torch.from_numpy(yn); tr = side == "train"; yd = y.to(device)
    rows = []
    for regime in REGIMES:
        tau = float(cal[regime]["tau"])
        pre = Preproc(regime, X[tr]); Fp = pre.transform(X).to(device)
        for s in SEEDS:
            g = torch.Generator().manual_seed(7000 + s)
            tr_idx = torch.from_numpy(np.where(tr)[0]); perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
            pos = perm[y[perm] > 0]; neg = perm[y[perm] < 0]
            nh = min(N_TRAIN_PATCH // 2, len(pos) - N_PROBE_FIT // 2, len(neg) - N_PROBE_FIT // 2)
            centres = torch.cat([pos[:nh], neg[:nh]])
            probe_fit = torch.cat([pos[nh:nh + N_PROBE_FIT // 2], neg[nh:nh + N_PROBE_FIT // 2]])
            Xrt, yrt = make_patches(Fp, y, core, centres, tr_idx, pre, tau, 8500 + s, "ctx_random", device)
            evals = {}
            for sd in ("val", "test"):
                idx = torch.from_numpy(np.where(side == sd)[0]); gs = torch.Generator().manual_seed(9000 + s)
                idx = idx[torch.randperm(len(idx), generator=gs)[:N_EVAL_MAX]]
                for cond in CONDITIONS:
                    evals[(sd, cond)] = make_patches(Fp, y, core, idx, torch.from_numpy(np.where(side == sd)[0]), pre, tau, 9500 + s, cond, device)
                evals[(sd, "centres")] = idx
            for name, (fname, key) in ENC.items():
                z = np.load(OUT_DIR / fname.format(r=regime, s=s)); Wenc = torch.from_numpy(z[key]).float().to(device)
                enc = Encoder(S_FEAT, K).to(device)
                with torch.no_grad():
                    enc.proj.weight.copy_(Wenc)
                enc.proj.weight.requires_grad_(False)
                torch.manual_seed(100 + s); head = PatchHead(K, HEAD_WIDTH).to(device)
                opt = torch.optim.SGD(head.parameters(), lr=LR, momentum=0.0)
                t0 = time.time(); step = 0; status = "budget"
                while True:
                    opt.zero_grad(set_to_none=True); logits = head(enc(Xrt)); loss = margin_loss(logits, yrt); lv = float(loss)
                    if lv < STOP_LOSS:
                        status = "stopped"; break
                    if step >= BUDGET:
                        break
                    loss.backward(); opt.step(); step += 1
                with torch.no_grad():
                    zf = enc.proj(Fp[probe_fit.to(device)]); w, thr = lda_fit(zf, yd[probe_fit.to(device)], 1e-4)
                    probe_val = lda_eval(w, thr, enc.proj(Fp[evals[("val", "centres")].to(device)]), yd[evals[("val", "centres")].to(device)])
                row = dict(regime=regime, seed=s, encoder=name, retrain_steps=step, retrain_status=status, retrain_train_loss=lv,
                           retrain_train_acc=float((torch.sign(logits) == yrt).float().mean()), probe_val=probe_val, wall_s=time.time() - t0)
                with torch.no_grad():
                    for (sd, cond), v in evals.items():
                        if cond != "centres":
                            row[f"retrained_acc_{cond}_{sd}"] = float((torch.sign(head(enc(v[0]))) == v[1]).float().mean())
                if ORIG[name] is not None:
                    snap = pd.read_csv(OUT_DIR / f"snap_{ORIG[name].format(r=regime, s=s)}.csv")
                    o = snap[snap["threshold"].astype(str) == "0.3"].iloc[0]
                    for cond in ("iid", "reversed", "ctx_random", "spec_only"):
                        row[f"original_acc_{cond}_val"] = float(o[f"acc_{cond}_val"])
                    row["original_probe_val"] = float(o["probe_val"])
                rows.append(row)
                print(f"[recovery3] {regime:9s} seed {s} {name:9s}: {status} at {step} (loss {lv:.3f}); probe {probe_val:.3f}; "
                      f"retrained val iid {row['retrained_acc_iid_val']:.3f} rev {row['retrained_acc_reversed_val']:.3f} "
                      f"ctxrnd {row['retrained_acc_ctx_random_val']:.3f} | test rev {row['retrained_acc_reversed_test']:.3f}"
                      + (f" | original rev {row['original_acc_reversed_val']:.3f}" if 'original_acc_reversed_val' in row else ""), flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / OUT_NAME, index=False)
    print(df.groupby(["regime", "encoder"])[["probe_val", "retrained_acc_reversed_val", "retrained_acc_ctx_random_val", "retrained_acc_reversed_test"]].mean().round(3))
    print(f"[recovery3] wrote {OUT_DIR / OUT_NAME}")


if __name__ == "__main__":
    main()
