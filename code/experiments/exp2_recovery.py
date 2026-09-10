#!/usr/bin/env python3
"""
Experiment 2 recovery (Astra review_iclr_02, "single change most likely to move
the score"): retrain the spatial head on FROZEN matched-fit encoders and measure
shifted accuracy with context present.

Protocol (fixed before inspecting any shifted evaluation):
  encoders, per seed s in {0,1,2}, all at M = 32:
    init      W0 (random initialization)                    enc_headlr_M32_x1_s{s}.npz["W0"]
    kappa1    W at L* = 0.30 of the kappa = 1 run            enc_headlr_M32_x1_s{s}.npz["W_0.3"]
    kappa256  W at L* = 0.30 of the kappa = 1/256 run        enc_headlr_M32_x1_h0.00390625_s{s}.npz["W_0.3"]
  head: CNNHead(K, 32, "sp"), PAIRED initialization (torch.manual_seed(100 + s)
        before constructing the head, identical for the three encoders of a seed)
  data: freshly sampled context-random training images (condition "ctx_random",
        seed 2500 + s, N_TRAIN = 256 images): context present but decorrelated
  optimizer: full-batch gradient descent, eta = 1e-3 on all head parameters,
        encoder frozen; budget 20,000 steps or training loss < 0.10, whichever
        first; the evaluated state is the final one (no selection on shifted results)
  evaluation: the SAME paired test family as the original runs (seed 3000 + s):
        iid, reversed, ctx_random, spec_only, ctx_only; plus the spectral probe on
        the frozen encoder (same probe sets) for reference.
Outputs: results/exp2/recovery_summary.csv (one row per encoder x seed with the
original run's L* shifted accuracies alongside), results/exp2/recovery_traj_*.csv.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from experiments.exp2_intervention import (  # noqa: E402
    ALPHA, BETA, SIGMA, S, K, H, W, N_TRAIN, CALIB, OUT_DIR, CNNHead, Encoder, ProblemSpec,
    build_data, margin_loss, accuracy, spectral_probe, make_problem_v2,
)

LR = 1e-3
BUDGET = 20_000
STOP_LOSS = 0.10
HEAD_WIDTH = 32
SEEDS = [0, 1, 2]
ENCODERS = {
    "init":     ("enc_headlr_M32_x1_s{s}.npz", "W0"),
    "kappa1":   ("enc_headlr_M32_x1_s{s}.npz", "W_0.3"),
    "kappa256": ("enc_headlr_M32_x1_h0.00390625_s{s}.npz", "W_0.3"),
}
ORIG_TAG = {"init": None, "kappa1": "headlr_M32_x1_s{s}", "kappa256": "headlr_M32_x1_h0.00390625_s{s}"}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = float(json.loads(CALIB.read_text())["tau"])
    spec = ProblemSpec(S=S, H=H, W=W, alpha=ALPHA, beta=BETA, tau=tau, sigma=SIGMA)
    rows = []
    for s in SEEDS:
        data = build_data(s, spec, "iid", device)          # paired test family 3000+s, probe sets, u, V
        u, V = data["u"].cpu(), data["V"].cpu()
        Xrt, yrt = make_problem_v2(N_TRAIN, spec, u, V, seed=2500 + s, condition="ctx_random")
        Xrt, yrt = Xrt.to(device), yrt.to(device)
        for name, (fname, key) in ENCODERS.items():
            z = np.load(OUT_DIR / fname.format(s=s))
            Wenc = torch.from_numpy(z[key]).float().to(device)
            enc = Encoder(S, K).to(device)
            with torch.no_grad():
                enc.proj.weight.copy_(Wenc)
            enc.proj.weight.requires_grad_(False)
            torch.manual_seed(100 + s)                       # paired head initialization across encoders
            head = CNNHead(K, HEAD_WIDTH, "sp").to(device)
            opt = torch.optim.SGD(head.parameters(), lr=LR, momentum=0.0)
            t0 = time.time(); traj = []
            step = 0; status = "budget"
            while True:
                opt.zero_grad(set_to_none=True)
                logits = head(enc(Xrt)); loss = margin_loss(logits, yrt); lv = float(loss)
                if step % 100 == 0:
                    traj.append(dict(step=step, loss=lv, acc=float((torch.sign(logits) == yrt).float().mean())))
                if lv < STOP_LOSS:
                    status = "stopped"; break
                if step >= BUDGET:
                    break
                loss.backward(); opt.step(); step += 1
            row = dict(seed=s, encoder=name, retrain_steps=step, retrain_status=status,
                       retrain_train_loss=lv, retrain_train_acc=float((torch.sign(logits) == yrt).float().mean()),
                       probe_acc=spectral_probe(enc, data), wall_s=time.time() - t0)
            for c, (Xc, yc) in data["tests"].items():
                row[f"retrained_acc_{c}"] = accuracy(enc, head, Xc, yc)
            # the original run's L* shifted accuracies, for the side-by-side table
            if ORIG_TAG[name] is not None:
                snap = pd.read_csv(OUT_DIR / f"snap_{ORIG_TAG[name].format(s=s)}.csv")
                o = snap[snap["threshold"].astype(str) == "0.3"].iloc[0]
                for c in ("iid", "reversed", "ctx_random", "spec_only"):
                    row[f"original_acc_{c}"] = float(o[f"acc_{c}"])
                row["original_probe_acc"] = float(o["probe_acc"]); row["original_h_u"] = float(o["h_u"])
            rows.append(row)
            pd.DataFrame(traj).to_csv(OUT_DIR / f"recovery_traj_{name}_s{s}.csv", index=False)
            print(f"[recovery] seed {s} {name:9s}: {status} at step {step} (loss {lv:.3f}); probe {row['probe_acc']:.3f}; "
                  f"retrained iid {row['retrained_acc_iid']:.3f} rev {row['retrained_acc_reversed']:.3f} "
                  f"ctxrnd {row['retrained_acc_ctx_random']:.3f} spec {row['retrained_acc_spec_only']:.3f}"
                  + (f" | original rev {row['original_acc_reversed']:.3f}" if 'original_acc_reversed' in row else ""), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "recovery_summary.csv", index=False)
    print(df.groupby("encoder")[["probe_acc", "retrained_acc_iid", "retrained_acc_reversed", "retrained_acc_ctx_random",
                                 "retrained_acc_spec_only"]].agg(["mean", "std"]).round(3))
    print(f"[recovery] wrote {OUT_DIR / 'recovery_summary.csv'}")


if __name__ == "__main__":
    main()
