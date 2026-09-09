"""E3c BN-recalibration test (reviewer hypothesis M1).

M1: the late-training validation collapse of the joint arms may be partly a
BatchNorm running-statistic mismatch under drifting features, not lost spectral
information. If so, resetting the BN running statistics and re-estimating them
from the training set (weights frozen) should recover most of the peak-to-final
validation drop.

For each saved E3c final.pt checkpoint we report
  f1_asis  : eval-mode val macro-F1 with the checkpoint's stored running stats
             (must match epochs.csv epoch-60 val_macro_f1 -- sanity check)
  f1_recal : same weights, but every BatchNorm's running_mean/var re-estimated
             as a cumulative average over ALL fold-0 train cores (momentum=None,
             BN modules in train() mode, everything else in eval(), no_grad)
plus per-class F1 for both, and the epochs.csv peak val F1 for context.

Nothing is trained; no checkpoint is modified on disk.

Usage:
  python code/experiments/e3c_bn_recal.py
  python code/experiments/e3c_bn_recal.py --max_train_cores 8 --limit 2   # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

THEORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THEORY_ROOT / "code" / "experiments"))
sys.path.insert(0, str(THEORY_ROOT / "code"))
sys.path.insert(0, "/home/u37314kd/Projects/spectral_tokenization/side_project")

from exp1_7_train import (  # noqa: E402
    NUM_CLASSES, SPATIAL, build_model, evaluate,
)
from data.core_dataset import CoreDataset  # noqa: E402

CLASS_NAMES = ["cancer_epi", "cas", "normal_stroma", "normal_epi"]
ARMS = ["joint_linear", "frozen_random", "frozen_pca"]
WIDTHS = [48, 192]
SEEDS = [0, 1, 2]


def bn_layers(model: nn.Module):
    return [m for m in model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def evaluate_ext(model, loader, device):
    """Val macro-F1 + per-class F1. Metric settings identical to exp1_7.evaluate."""
    model.eval()
    preds, labs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        valid = y != 255
        if int(valid.sum().item()) == 0:
            continue
        logits = model(x)
        preds.append(logits.argmax(1)[valid].cpu().numpy())
        labs.append(y[valid].cpu().numpy())
    preds, labs = np.concatenate(preds), np.concatenate(labs)
    labels = list(range(NUM_CLASSES))
    return {
        "macro_f1": float(f1_score(labs, preds, average="macro",
                                   labels=labels, zero_division=0)),
        "per_class": [float(v) for v in f1_score(
            labs, preds, average=None, labels=labels, zero_division=0)],
    }


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    """Reset every BN's running stats, re-estimate them over `loader`.

    Weights are untouched. model.eval() is applied first so any non-BN
    stochastic module (dropout is 0.0 here, but be explicit) stays in eval;
    only the BatchNorm modules are put in train() so their running buffers
    update. momentum=None -> cumulative moving average over the whole pass,
    i.e. the exact train-set mean/var rather than an EMA of the last batches.
    """
    model.eval()
    bns = bn_layers(model)
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()      # mean=0, var=1, num_batches_tracked=0
        m.momentum = None            # cumulative average
        m.train()
    n_batches = 0
    for x, _ in loader:
        model(x.to(device))
        n_batches += 1
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()
    return len(bns), n_batches


def csv_val_f1(run_dir: Path):
    """(epoch-60 val_macro_f1, peak val_macro_f1, peak epoch, n_epochs) from epochs.csv."""
    rows = list(csv.DictReader(open(run_dir / "epochs.csv")))
    f1s = [(int(r["epoch"]), float(r["val_macro_f1"])) for r in rows]
    last_ep, last_f1 = f1s[-1]
    peak_ep, peak_f1 = max(f1s, key=lambda t: t[1])
    return last_f1, peak_f1, peak_ep, last_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/hdd2/u37314kd/data_breast_v2_pca23")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--e3c_root", default=str(
        THEORY_ROOT / "experiments_shortcut" / "e3c" / "breast_f0"))
    ap.add_argument("--out_csv", default=str(
        THEORY_ROOT / "results" / "e3c_bn_recal.csv"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--max_train_cores", type=int, default=0, help="smoke only")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N checkpoints")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    e3c_root = Path(args.e3c_root)

    # ---- which checkpoints exist ------------------------------------------
    todo, missing = [], []
    for arm in ARMS:
        for w in WIDTHS:
            for s in SEEDS:
                d = e3c_root / f"{arm}_h{w}_adamw_s{s}"
                (todo if (d / "final.pt").exists() else missing).append((arm, w, s, d))
    if missing:
        for arm, w, s, d in missing:
            print(f"[skip] no final.pt: {d.name}", flush=True)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[plan] {len(todo)} checkpoints, {len(missing)} missing", flush=True)

    # ---- data: loaded ONCE, shared by every checkpoint ---------------------
    split_file = Path(args.data_dir) / f"splits_fold{args.fold}.json"
    if args.max_train_cores > 0:
        full = json.loads(split_file.read_text())
        trimmed = {k: [] for k in full}
        trimmed["train"] = full["train"][: args.max_train_cores]
        trimmed["val"] = full["val"][:4]
        tmp = Path("/tmp/claude-1008/spectral_shortcut_scratch") / "bnrecal_split.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(trimmed))
        split_file = tmp
        print(f"[data] TRIMMED split -> {split_file}", flush=True)

    train_ds = CoreDataset(args.data_dir, str(split_file), "train",
                           spatial_size=SPATIAL, augment=False)
    val_ds = CoreDataset(args.data_dir, str(split_file), "val",
                         spatial_size=SPATIAL, augment=False)
    # No shuffle, no drop_last: the recalibration pass must see EVERY train core.
    # augment=False -> the BN statistics are those of the un-augmented train set.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    print(f"[data] {len(train_ds)} train cores / {len(val_ds)} val cores", flush=True)

    fields = (["arm", "width", "seed", "n_bn", "n_recal_batches",
               "f1_asis", "f1_epoch60_csv", "f1_recal", "delta",
               "f1_peak_csv", "peak_epoch", "peak_minus_final",
               "recal_recovery_frac", "asis_csv_absdiff"]
              + [f"asis_f1_{c}" for c in CLASS_NAMES]
              + [f"recal_f1_{c}" for c in CLASS_NAMES])
    rows = []
    checked_evaluate_reuse = False

    for i, (arm, w, s, d) in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {d.name} ===", flush=True)
        margs = SimpleNamespace(arm=arm, width=w, seed=s,
                                num_layers=args.num_layers, pretrained_path=None)
        model = build_model(margs)
        # frozen_pca: PCAReduction.fit() is NOT re-run -- the fitted projection
        # is in the checkpoint. strict=True proves every buffer/param matched.
        ck = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model = model.to(device)
        assert int(ck["epoch"]) == 60, f"{d.name} epoch={ck['epoch']}"

        asis = evaluate_ext(model, val_loader, device)
        if not checked_evaluate_reuse:
            ref = evaluate(model, val_loader, device)["macro_f1"]
            print(f"[sanity] exp1_7.evaluate()={ref:.6f} vs "
                  f"evaluate_ext={asis['macro_f1']:.6f} "
                  f"(absdiff {abs(ref - asis['macro_f1']):.2e})", flush=True)
            assert abs(ref - asis["macro_f1"]) < 1e-9
            checked_evaluate_reuse = True

        f60, fpeak, peak_ep, n_ep = csv_val_f1(d)
        print(f"[asis ] f1={asis['macro_f1']:.4f}  csv_ep{n_ep}={f60:.4f}  "
              f"absdiff={abs(asis['macro_f1'] - f60):.2e}  "
              f"peak={fpeak:.4f}@ep{peak_ep}", flush=True)

        n_bn, n_bat = recalibrate_bn(model, train_loader, device)
        recal = evaluate_ext(model, val_loader, device)
        drop = fpeak - f60
        rec_frac = (recal["macro_f1"] - asis["macro_f1"]) / drop if drop > 1e-9 else float("nan")
        print(f"[recal] {n_bn} BN layers over {n_bat} batches -> "
              f"f1={recal['macro_f1']:.4f}  delta={recal['macro_f1'] - asis['macro_f1']:+.4f}  "
              f"recovery_frac={rec_frac:.3f}", flush=True)

        row = dict(arm=arm, width=w, seed=s, n_bn=n_bn, n_recal_batches=n_bat,
                   f1_asis=asis["macro_f1"], f1_epoch60_csv=f60,
                   f1_recal=recal["macro_f1"],
                   delta=recal["macro_f1"] - asis["macro_f1"],
                   f1_peak_csv=fpeak, peak_epoch=peak_ep,
                   peak_minus_final=drop, recal_recovery_frac=rec_frac,
                   asis_csv_absdiff=abs(asis["macro_f1"] - f60))
        for j, c in enumerate(CLASS_NAMES):
            row[f"asis_f1_{c}"] = asis["per_class"][j]
            row[f"recal_f1_{c}"] = recal["per_class"][j]
        rows.append(row)

        del model, ck
        torch.cuda.empty_cache()

        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)

    print(f"\n[done] wrote {args.out_csv} ({len(rows)} rows)", flush=True)

    # ---- arm x width summary ----------------------------------------------
    print("\n" + "=" * 78)
    print("MEAN val macro-F1 over seeds  (as-is -> BN-recalibrated)")
    print("=" * 78)
    print(f"{'arm':<16}{'width':>7}{'n':>4}{'as-is':>10}{'recal':>10}"
          f"{'delta':>10}{'peak_csv':>10}{'recov':>8}")
    means = {}
    for arm in ARMS:
        for w in WIDTHS:
            sub = [r for r in rows if r["arm"] == arm and r["width"] == w]
            if not sub:
                continue
            a = float(np.mean([r["f1_asis"] for r in sub]))
            b = float(np.mean([r["f1_recal"] for r in sub]))
            pk = float(np.mean([r["f1_peak_csv"] for r in sub]))
            rc = float(np.nanmean([r["recal_recovery_frac"] for r in sub]))
            means[(arm, w)] = (a, b, pk)
            print(f"{arm:<16}{w:>7}{len(sub):>4}{a:>10.4f}{b:>10.4f}"
                  f"{b - a:>+10.4f}{pk:>10.4f}{rc:>8.2f}")
    print("\nJOINT minus FROZEN gap (joint_linear - frozen_*), mean over seeds")
    print(f"{'comparison':<34}{'as-is':>10}{'recal':>10}")
    for w in WIDTHS:
        if ("joint_linear", w) not in means:
            continue
        ja, jb, _ = means[("joint_linear", w)]
        for fr in ("frozen_random", "frozen_pca"):
            if (fr, w) not in means:
                continue
            fa, fb, _ = means[(fr, w)]
            print(f"{f'h{w}: joint_linear - {fr}':<34}{ja - fa:>+10.4f}{jb - fb:>+10.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
