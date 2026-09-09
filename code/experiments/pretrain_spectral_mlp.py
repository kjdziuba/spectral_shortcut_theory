"""E3c-local — pretrain the per-pixel spectral MLP used by the joint_mlp arm.

Trains MLPSpectralReduction (the EXACT class / dims exp1_7_train.py builds for
--arm joint_mlp: 942 -> 512 -> GELU -> 64) plus a linear head 64 -> 4 with
per-pixel 4-class cross-entropy over the labeled pixels of the fold-0 TRAIN
cores, selecting the checkpoint by macro-F1 over the fold-0 VAL cores' labeled
pixels. The resulting reduction-only state dict feeds the runner's
frozen_pretrained (theta frozen) and finetune_real (theta trainable) arms.

Data path == runner's data path, by construction:
  * CoreDataset is imported FROM exp1_7_train (same module object, same
    label_map {3:0, 4:1, 2:2, 1:3}, same 336 pad / center-crop, same split
    file), and pixels are flattened from ``train_ds[i]`` -- the very tensors
    the runner feeds the model -- so a core larger than 336 px loses the same
    border pixels here as in the runner.
  * No input normalization anywhere: the runner feeds raw NPZ values
    (X_raw, X_d1, X_d2 stacked) to spectral_reduce and so do we.
  * Flattening order matches MLPSpectralReduction.forward
    (permute(0,2,3,1,4).reshape(B,H,W,C*S): channel-major [raw|d1|d2] rows)
    and forward() is a pure reshape around ``self.net``, so per-pixel training
    calls ``red.net(rows)``; an equivalence assert against forward() runs at
    start-up to make that claim checkable rather than trusted.

Optimizer: AdamW lr 1e-3 (constant), <= 30 epochs, early stop patience 5 on
val macro-F1. Batches of pixels, shuffled per epoch with a seeded generator;
optional seed-controlled subsample of the train pixels (--max_pixels, default
2M -- a no-op for breast fold 0 which has ~443k labeled train pixels).

Output (one file per seed):
  experiments_shortcut/pretrained/spectral_mlp_f{fold}_s{seed}.pt
    {'state_dict': reduction-only state dict (net.0.*, net.2.*),
     'pixel_val_f1': best val macro-F1, 'epoch': best epoch, 'seed': seed,
     + provenance: head_state_dict, arch, fold, data_dir, pixel counts,
       optimizer settings, per-epoch history, wall time}

Usage (one process, seeds sequential, data loaded once):
  python code/experiments/pretrain_spectral_mlp.py --seeds 0 1 2
  # smoke:
  python code/experiments/pretrain_spectral_mlp.py --seeds 0 --max_train_cores 8 \
      --max_epochs 2 --out_dir /tmp/pretrain_smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1_7_train import (                                   # noqa: E402
    IN_CHANNELS, NUM_CLASSES, NUM_SPECTRAL, SCRATCH, SPATIAL, THEORY_ROOT,
    CoreDataset, MLPSpectralReduction,
)

PRETRAINED_DIR = THEORY_ROOT / "experiments_shortcut" / "pretrained"


def pretrained_path(fold: int, seed: int, out_dir: Path = PRETRAINED_DIR) -> Path:
    return Path(out_dir) / f"spectral_mlp_f{fold}_s{seed}.pt"


# ---------------------------------------------------------------------------
# Data: flatten the runner's per-core tensors into labeled pixel rows
# ---------------------------------------------------------------------------

def flatten_labeled_pixels(ds: CoreDataset):
    """Rows (N, C*S) float32 + labels (N,) int64 over labeled pixels.

    Iterates ``ds[i]`` with augment=False, so padding/cropping is what the
    runner's spectral_reduce sees. (The runner flips augment back ON for its
    train loop -- exp1_7_train.py, right after the probe batch -- but its
    augmentation is purely spatial (flip / rot90), so the multiset of
    per-pixel (spectrum, label) pairs is unchanged; only WHICH border pixels
    the 336 center-crop drops can differ, for the few cores above 336 px.)
    Row layout is channel-major [raw(314) | d1(314) | d2(314)], as in
    MLPSpectralReduction.forward.
    """
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]                       # (C, H, W, S), (H, W)
        valid = y != 255
        if not bool(valid.any()):
            continue
        px = x.permute(1, 2, 0, 3)[valid]  # (n, C, S)
        xs.append(px.reshape(px.shape[0], -1).contiguous())
        ys.append(y[valid])
    return torch.cat(xs), torch.cat(ys)


def check_row_equivalence(red: MLPSpectralReduction, device) -> None:
    """Assert red(x) (runner path) == red.net(rows) (per-pixel path)."""
    x = torch.randn(2, IN_CHANNELS, 3, 5, NUM_SPECTRAL, device=device)
    with torch.no_grad():
        a = red(x)                                             # (B, K, H, W)
        rows = x.permute(0, 2, 3, 1, 4).reshape(-1, IN_CHANNELS * NUM_SPECTRAL)
        b = red.net(rows).reshape(2, 3, 5, -1).permute(0, 3, 1, 2)
    assert torch.allclose(a, b, atol=1e-5), "row path != forward path"


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_pixels(red, head, X, Y, bs=16384):
    red.eval(); head.eval()
    preds, loss_sum = [], 0.0
    for b in range(0, X.shape[0], bs):
        logits = head(red.net(X[b:b + bs]))
        loss_sum += float(F.cross_entropy(logits, Y[b:b + bs], reduction="sum").item())
        preds.append(logits.argmax(1))
    preds = torch.cat(preds).cpu().numpy()
    labs = Y.cpu().numpy()
    return {
        "loss": loss_sum / max(X.shape[0], 1),
        "macro_f1": f1_score(labs, preds, average="macro",
                             labels=list(range(NUM_CLASSES)), zero_division=0),
        "acc": accuracy_score(labs, preds),
    }


def train_seed(seed, Xtr, Ytr, Xva, Yva, args, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    red = MLPSpectralReduction(in_channels=IN_CHANNELS, num_spectral=NUM_SPECTRAL,
                               hidden=512, reduce_dim=64).to(device)
    head = nn.Linear(64, NUM_CLASSES).to(device)
    check_row_equivalence(red, device)
    opt = torch.optim.AdamW(list(red.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(seed)

    # seed-controlled pixel subsample (no-op unless N > max_pixels)
    n_all = Xtr.shape[0]
    if args.max_pixels > 0 and n_all > args.max_pixels:
        keep = torch.randperm(n_all, generator=gen)[: args.max_pixels].to(device)
        Xs, Ys = Xtr[keep], Ytr[keep]
    else:
        Xs, Ys = Xtr, Ytr
    n = Xs.shape[0]

    best_f1, best_epoch, best_state, best_head, best_val = -1.0, 0, None, None, None
    bad, history = 0, []
    t0 = time.time()
    for epoch in range(1, args.max_epochs + 1):
        red.train(); head.train()
        perm = torch.randperm(n, generator=gen).to(device)
        tr_loss, tr_n = 0.0, 0
        for b in range(0, n, args.batch_size):
            idx = perm[b:b + args.batch_size]
            logits = head(red.net(Xs[idx]))
            loss = F.cross_entropy(logits, Ys[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * idx.numel()
            tr_n += idx.numel()
        val = eval_pixels(red, head, Xva, Yva)
        history.append(dict(epoch=epoch, train_loss=tr_loss / max(tr_n, 1),
                            val_loss=val["loss"], val_macro_f1=val["macro_f1"],
                            val_acc=val["acc"]))
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, best_val, bad = val["macro_f1"], epoch, val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in red.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        print(f"[seed {seed}] epoch {epoch:2d}/{args.max_epochs} "
              f"train_loss={tr_loss / max(tr_n, 1):.4f} val_loss={val['loss']:.4f} "
              f"val_f1={val['macro_f1']:.4f} val_acc={val['acc']:.4f}"
              f"{' *' if improved else ''}  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= args.patience:
            print(f"[seed {seed}] early stop: no val-F1 improvement for "
                  f"{args.patience} epochs", flush=True)
            break
    wall = time.time() - t0

    out = pretrained_path(args.fold, seed, Path(args.out_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,            # reduction-only: net.0.*, net.2.*
        "pixel_val_f1": float(best_f1),
        "epoch": int(best_epoch),
        "seed": int(seed),
        "head_state_dict": best_head,
        "pixel_val_acc": float(best_val["acc"]),
        "pixel_val_loss": float(best_val["loss"]),
        "arch": "MLPSpectralReduction(942->512->GELU->64) + Linear(64->4)",
        "fold": int(args.fold), "data_dir": args.data_dir,
        "max_train_cores": int(args.max_train_cores),
        "n_train_pixels": int(n), "n_train_pixels_available": int(n_all),
        "n_val_pixels": int(Xva.shape[0]),
        "optimizer": "adamw", "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "max_epochs": args.max_epochs,
        "patience": args.patience, "epochs_run": len(history),
        "history": history, "wall_secs": round(wall, 1),
        "input_normalization": "none (raw NPZ X_raw/X_d1/X_d2, as in runner)",
        # str(): torch.__version__ is a TorchVersion instance, which the
        # weights_only=True unpickler the runner uses would reject.
        "torch_version": str(torch.__version__),
    }, out)
    size_mb = out.stat().st_size / 1e6
    print(f"[seed {seed}] BEST pixel val macro-F1 = {best_f1:.4f} (epoch {best_epoch}, "
          f"{len(history)} epochs run, {wall:.0f}s) -> {out} ({size_mb:.2f} MB)",
          flush=True)
    return dict(seed=seed, pixel_val_f1=float(best_f1), epoch=best_epoch,
                epochs_run=len(history), wall_secs=round(wall, 1), path=str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data_dir", default="/mnt/hdd2/u37314kd/data_breast_v2_pca23")
    ap.add_argument("--dataset_name", default="breast")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01,
                    help="AdamW default; pretraining only (runner's theta "
                         "group stays at 0.0)")
    ap.add_argument("--batch_size", type=int, default=4096, help="pixels/step")
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max_pixels", type=int, default=2_000_000,
                    help="seed-controlled subsample of train pixels; 0 = all")
    ap.add_argument("--max_train_cores", type=int, default=0,
                    help="smoke only: trim to N train + 4 val cores "
                         "(same trimming as exp1_7_train.py)")
    ap.add_argument("--out_dir", default=str(PRETRAINED_DIR))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    split_file = Path(args.data_dir) / f"splits_fold{args.fold}.json"
    if not split_file.exists():
        raise SystemExit(f"missing split file: {split_file}")
    if args.max_train_cores > 0:               # exp1_7_train.py trimming, verbatim
        full = json.loads(split_file.read_text())
        trimmed = {k: [] for k in full}
        trimmed["train"] = full["train"][: args.max_train_cores]
        trimmed["val"] = full["val"][:4]
        tmp = SCRATCH / (f"pretrain_split_{args.dataset_name}_f{args.fold}"
                         f"_n{args.max_train_cores}.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(trimmed))
        print(f"[data] TRIMMED split -> {tmp}", flush=True)
        split_file = tmp

    t0 = time.time()
    train_ds = CoreDataset(args.data_dir, str(split_file), "train",
                           spatial_size=SPATIAL, augment=False)
    val_ds = CoreDataset(args.data_dir, str(split_file), "val",
                         spatial_size=SPATIAL, augment=False)
    assert len(train_ds.wavenumbers) == NUM_SPECTRAL
    Xtr, Ytr = flatten_labeled_pixels(train_ds)
    Xva, Yva = flatten_labeled_pixels(val_ds)
    n_tr_cores, n_va_cores = len(train_ds), len(val_ds)
    del train_ds, val_ds                        # free the ~60 GB of whole cores
    print(f"[data] train {n_tr_cores} cores -> {Xtr.shape[0]:,} labeled px "
          f"(class counts {np.bincount(Ytr.numpy(), minlength=NUM_CLASSES).tolist()}); "
          f"val {n_va_cores} cores -> {Xva.shape[0]:,} labeled px "
          f"(class counts {np.bincount(Yva.numpy(), minlength=NUM_CLASSES).tolist()}); "
          f"load+flatten {time.time() - t0:.0f}s", flush=True)
    Xtr, Ytr, Xva, Yva = (t.to(device) for t in (Xtr, Ytr, Xva, Yva))

    results = [train_seed(s, Xtr, Ytr, Xva, Yva, args, device) for s in args.seeds]
    print("\n=== pretrained spectral MLP: pixel val macro-F1 per seed ===")
    for r in results:
        print(f"  seed {r['seed']}: pixel_val_f1={r['pixel_val_f1']:.4f}  "
              f"best_epoch={r['epoch']}  epochs_run={r['epochs_run']}  "
              f"wall={r['wall_secs']}s  {r['path']}")
    # Human-readable summary next to the .pt files (merged across invocations
    # so seeds run in separate processes still land in one file).
    summary_path = Path(args.out_dir) / f"pretrain_summary_f{args.fold}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for r in results:
        summary[f"seed_{r['seed']}"] = dict(
            r, n_train_pixels=int(Xtr.shape[0]), n_val_pixels=int(Xva.shape[0]),
            n_train_cores=n_tr_cores, n_val_cores=n_va_cores, lr=args.lr,
            weight_decay=args.weight_decay, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=args.patience,
            data_dir=args.data_dir, max_train_cores=args.max_train_cores)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[write] {summary_path}")
    if device.type == "cuda":
        print(f"[done] peak_gpu={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
