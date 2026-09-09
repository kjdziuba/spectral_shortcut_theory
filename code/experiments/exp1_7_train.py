"""Exp 1.7 / E3c — Retraining runner for the spectral-shortcut theory paper.

Retrains small F = g_phi(f_theta(X)) pipelines on real hyperspectral tissue
cores while logging the quantities that test Theorem 1 (curvature disparity /
gradient starvation) and Theorem 2 (frozen-theta flow):

  per step  : loss, grad_theta_norm, grad_phi_norm, EGR, r_rms
  per epoch : val metrics, parameter displacement (theta / phi),
              feature displacement on a fixed probe batch, and (every 5
              epochs) the operator norm ||d g_phi / d Z||_op at the current
              probe features via power iteration.

Arms (--arm):
  joint_linear  (A1): LinearSpectralReduction, jointly trained.
                      spectral_norm=False (Assumption-1 hygiene: no
                      pre-projection BatchNorm1d). theta := spectral_reduce.proj
                      ONLY; the post-projection BatchNorm2d goes in phi.
  frozen_random (A2): identical construction + same-seed init as joint_linear,
                      but ALL spectral_reduce params excluded from the
                      optimizer. requires_grad stays True so the COUNTERFACTUAL
                      gradient f_theta would have received is logged by EGR;
                      those grads are zeroed manually every step.
  frozen_pca    (A3): PCAReduction(64) fitted on the training cores, frozen,
                      counterfactual grads logged (forward re-implemented
                      without @torch.no_grad so autograd can reach theta).
  joint_mlp     (B) : per-pixel MLP 942->512->GELU->64, jointly trained.
  frozen_pretrained : E3c-local. Same MLPSpectralReduction class/dims as
                      joint_mlp, weights loaded from the per-seed pixel-level
                      pretrain (code/experiments/pretrain_spectral_mlp.py ->
                      experiments_shortcut/pretrained/spectral_mlp_f{fold}_s{seed}.pt),
                      then FROZEN exactly like frozen_random: excluded from the
                      optimizer, requires_grad kept True so counterfactual
                      theta grads are logged, grads zeroed manually each step.
                      (No BN inside the MLP, so train/eval forward is the same
                      deterministic map; frozen_random's BN2d bookkeeping has
                      no analogue here.)
  finetune_real     : E3c-local. Same pretrained init, theta TRAINABLE in the
                      zero-weight-decay group (joint_mlp's group treatment).

Optimizer (non-negotiable, review round): theta params sit in their OWN param
group with weight_decay=0.0 — a decay term on theta breaks the Theorem-2 flow.
phi gets weight_decay=0.01. AdamW lr=1e-4 (default) or SGD momentum=0.9 lr=1e-2.
Constant LR, fixed --epochs, NO early stopping.
--spectral_lr_mult M (default 1.0) multiplies the THETA group's lr only, for
the trainable-theta arms (joint_linear / joint_mlp / finetune_real); phi's lr
is never touched. With M == 1.0 the param groups are built exactly as before.
--run_label overrides the arm token in the output dir name (lane driver uses
it for e.g. joint_speclr0.1_h192_adamw_s0); default = arm, unchanged.

MATCHED-HYGIENE OPTIONS (reviewer round, v2). All OPT-IN: with every flag at
its default the optimisation is bit-identical to the pre-v2 script, and so are
epochs.csv and the pre-existing eight columns of steps.csv. config.json is the
one deliberate exception: it now always records the four new flags plus
n_bn_affine_* / clip_max_norm / base_lrs, so a default run is self-describing.
  --save_best         also write <run>/best.pt at the best val-macro-F1 epoch
                      (atomic tmp + os.replace) and record best_epoch /
                      best_val_f1 in config.json at the end. final.pt is
                      still written, unchanged.
  --bn_affine_mode    where the SPECTRAL REDUCTION's BatchNorm affine
                      (gamma/beta) parameters go.
                        arm_default : current per-arm behaviour (joint_linear
                                      trains spectral_reduce.norm in phi;
                                      frozen_random freezes it).
                        train_all   : trainable in EVERY arm, in the phi
                                      group -- frozen arms train them too.
                        freeze_all  : frozen in EVERY arm (out of the
                                      optimizer, grads zeroed like the other
                                      frozen params).
                      NOTE the reduction modules differ: LinearSpectralReduction
                      carries BatchNorm2d(64) = 128 affine params (joint_linear /
                      frozen_random); PCAReduction and MLPSpectralReduction
                      carry NO BatchNorm at all, so for frozen_pca / joint_mlp /
                      frozen_pretrained / finetune_real the count is 0 and the
                      mode is a no-op. The realised counts per group are written
                      to config.json (n_bn_affine_{total,theta,phi,frozen}).
  --clip_scope        which parameters the grad-norm clip is computed over.
                        optimizer : current -- all optimizer params jointly.
                        phi_only  : norm over the phi params only, in EVERY
                                    arm. theta is then left COMPLETELY
                                    UNCLIPPED (it is deliberately NOT scaled
                                    by phi's coefficient -- the point of the
                                    matched protocol is that phi sees the same
                                    clip in joint and frozen arms while theta's
                                    step is governed by its own lr alone).
                        none      : no clipping at all.
  --lr_schedule       none (constant, current) | cosine (per-EPOCH decay of
                      every param group's lr from its initial value to 0 over
                      --epochs: lr_e = lr_0 * 0.5*(1+cos(pi*(e-1)/E)) ).

VERIFIER-ROUND FIXES (v2, applied to the NEW arms/flags only — no existing
arm's optimisation changes):
  * --spectral_lr_mult now rejects NaN/inf as well as <= 0 (`nan <= 0` is
    False, so `--spectral_lr_mult nan` used to poison theta's lr from step 1).
  * an explicit --pretrained_path is validated on seed AND fold AND data_dir
    (previously only seed, and only as a warning), so a fold-1 or
    other-dataset encoder can no longer be loaded silently.
    --allow_pretrained_mismatch downgrades all three to warnings.
  * config.json now records the two KNOWN CONFOUNDS of the pretrained arms
    (pretrained_selection_caveat, pretrained_augment_caveat) plus the
    checkpoint's fold / data_dir / weight_decay, and the runner prints a
    [bias] line. The pretrain script picks its epoch by macro-F1 on the SAME
    fold val cores this runner scores, so frozen_pretrained / finetune_real
    carry a val-selection advantage the other arms do not; and it pretrained
    theta under AdamW weight_decay=0.01, which the runner's own protocol
    forbids for theta. Both must be stated wherever those arms are compared.
    These keys appear ONLY for the pretrained arms.
  * (fix round 2) the selection split / caveats are derived from the
    CHECKPOINT's fold+data_dir, not from args.fold. The previous version
    asserted "selected on the val cores this run scores" even when
    --allow_pretrained_mismatch had loaded another fold's or another
    dataset's theta, i.e. it wrote a false provenance record on exactly the
    path the flag exists for. config.json also records pretrained_mismatch
    and pretrained_selection_matches_run_val.
  * (fix round 2) clip_coef mirrors torch's clamp instead of min(): a bare
    min(1.0, nan) is 1.0, so a NaN grad norm used to be logged as "no clip"
    while torch had scaled every gradient by NaN.
  * (fix round 2) probe_features' docstring no longer claims feat_disp == 0
    for frozen arms; under --bn_affine_mode train_all the reduction's BN
    affine trains, so a frozen arm's probe features move.

steps.csv gained four columns (appended after the existing eight, which are
unchanged): grad_total_norm_preclip, clip_coef, upd_theta_norm, upd_phi_norm.
  grad_total_norm_preclip : the pre-clip total grad norm over the CLIP SCOPE's
      parameters (for --clip_scope none, over the optimizer params, so the
      column stays comparable across scopes).
  clip_coef : the coefficient actually applied, torch's own
      min(1, max_norm/(total_norm + 1e-6)); exactly 1.0 for --clip_scope none.
  upd_theta_norm / upd_phi_norm : L2 norm of the ACTUAL parameter change across
      optimizer.step(), from pre/post snapshots. Exactly 0 for a group that is
      not in the optimizer (every frozen arm's theta).

Usage:
  python code/experiments/exp1_7_train.py --arm joint_linear --width 192 --seed 0
  python code/experiments/exp1_7_train.py --arm joint_linear --width 192 --seed 0 \
      --spectral_lr_mult 0.1 --run_label joint_speclr0.1
  # matched hygiene:
  python code/experiments/exp1_7_train.py --arm frozen_random --width 48 --seed 0 \
      --bn_affine_mode train_all --clip_scope phi_only --save_best \
      --run_label frozen_random_m
  # smoke:
  python code/experiments/exp1_7_train.py --arm frozen_random --width 48 \
      --seed 0 --epochs 2 --max_train_cores 8
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

THEORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THEORY_ROOT / "code"))
SIDE = Path("/home/u37314kd/Projects/spectral_tokenization/side_project")
sys.path.insert(0, str(SIDE))

from egr.callback import EGRLogger                              # noqa: E402
from models.blockvit_v2 import BlockViTv2, PCAReduction         # noqa: E402
from data.core_dataset import CoreDataset                       # noqa: E402

NUM_HEADS = 12
NUM_CLASSES = 4
NUM_SPECTRAL = 314
IN_CHANNELS = 3
SPATIAL = 336
SCRATCH = Path("/tmp/claude-1008/spectral_shortcut_scratch")
PRETRAINED_DIR = THEORY_ROOT / "experiments_shortcut" / "pretrained"
PRETRAINED_ARMS = ("frozen_pretrained", "finetune_real")
TRAINABLE_THETA_ARMS = ("joint_linear", "joint_mlp", "finetune_real")
CLIP_MAX_NORM = 1.0        # unchanged from the pre-v2 hard-coded 1.0


# ---------------------------------------------------------------------------
# Spectral-reduction modules defined in-script
# ---------------------------------------------------------------------------

class MLPSpectralReduction(nn.Module):
    """Per-pixel MLP spectral reduction: 942 -> 512 -> GELU -> 64.

    Input (B, C, H, W, S) -> flatten each pixel's spectrum to a row -> MLP ->
    output (B, K, H, W). Processed row-by-row (like LinearSpectralReduction)
    to keep the peak *intermediate* tensors small; note autograd still retains
    the 512-dim hidden activation for every pixel.
    """

    def __init__(self, in_channels: int = 3, num_spectral: int = 314,
                 hidden: int = 512, reduce_dim: int = 64):
        super().__init__()
        self.reduce_dim = reduce_dim
        in_dim = in_channels * num_spectral
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, reduce_dim),
        )

    def forward(self, x):
        """(B, C, H, W, S) -> (B, K, H, W)"""
        B, C, H, W, S = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C * S)
        out_rows = [self.net(x[:, i, :, :]) for i in range(H)]   # each (B, W, K)
        return torch.stack(out_rows, dim=1).permute(0, 3, 1, 2)


class GradPCAReduction(PCAReduction):
    """PCAReduction whose forward participates in autograd.

    The parent's forward is decorated @torch.no_grad(), which would silence the
    counterfactual theta gradients EGR needs. Same math, same interface:
    (B, C, H, W, S) -> (B, K, H, W), frozen linear map y = (x - mean) @ W_pca.T.
    """

    def forward(self, x):  # noqa: D102 — grad-enabled clone of parent forward
        B, C, H, W, S = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C * S)
        out_rows = [self.proj(x[:, i, :, :]) for i in range(H)]
        return torch.stack(out_rows, dim=1).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Model / param-group construction
# ---------------------------------------------------------------------------

def load_pretrained_ckpt(path):
    """torch.load a pretrain_spectral_mlp.py checkpoint (weights_only=True).

    Explicit rather than relying on the torch>=2.6 default: the file must stay
    plain tensors/containers/primitives. (pretrain_spectral_mlp.py stores
    str(torch.__version__) for exactly this reason -- a bare TorchVersion is
    rejected by the weights_only unpickler.)
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def build_model(args) -> nn.Module:
    """Seed, build BlockViTv2(spectral_norm=False), swap the reduction per arm.

    Seeding happens immediately before construction and the arm-specific swap
    happens AFTER the base model is built, so joint_linear / frozen_random /
    (phi of) all arms share bit-identical initialization at equal seed.
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = BlockViTv2(
        in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
        num_spectral=NUM_SPECTRAL, reduce_dim=64, patch_tok_size=16,
        hidden_dim=args.width, num_layers=args.num_layers, num_heads=NUM_HEADS,
        mlp_ratio=4.0, dropout=0.0, spatial_size=SPATIAL,
        spectral_norm=False,   # Assumption-1 hygiene: no pre-projection BN1d
    )
    if args.arm == "frozen_pca":
        model.spectral_reduce = GradPCAReduction(
            n_components=64, in_channels=IN_CHANNELS, num_spectral=NUM_SPECTRAL)
    elif args.arm == "joint_mlp":
        model.spectral_reduce = MLPSpectralReduction(
            in_channels=IN_CHANNELS, num_spectral=NUM_SPECTRAL,
            hidden=512, reduce_dim=64)
    elif args.arm in PRETRAINED_ARMS:
        # Same construction as joint_mlp (identical RNG draw -> identical phi
        # init at equal seed), then the random init is overwritten in place.
        model.spectral_reduce = MLPSpectralReduction(
            in_channels=IN_CHANNELS, num_spectral=NUM_SPECTRAL,
            hidden=512, reduce_dim=64)
        ck = load_pretrained_ckpt(args.pretrained_path)
        model.spectral_reduce.load_state_dict(ck["state_dict"], strict=True)
    return model


def resolve_pretrained(args) -> dict:
    """Locate the per-seed pretrained spectral MLP (pretrained arms only).

    Sets args.pretrained_path (default: PRETRAINED_DIR/spectral_mlp_f{fold}_s{seed}.pt)
    and returns the checkpoint's provenance for config.json. For all other
    arms returns empty provenance and leaves args.pretrained_path = None.

    Verifier fix: the checkpoint is validated on seed AND fold AND data_dir,
    on the explicit-path branch too (it used to check only the seed, and only
    as a warning there, so a fold-1 / other-dataset encoder loaded silently).
    A field the checkpoint does not carry is skipped with a warning rather
    than failing; --allow_pretrained_mismatch downgrades every mismatch to a
    warning for a deliberate cross-condition study. The default path encodes
    fold+seed, so for a default-path run this can only fire on --data_dir.
    """
    meta = dict(pretrained_path=None, pretrained_pixel_val_f1=None,
                pretrained_epoch=None, pretrained_seed=None)
    if args.arm not in PRETRAINED_ARMS:
        if args.pretrained_path is not None:
            print(f"[warn] --pretrained_path ignored for arm {args.arm}", flush=True)
            args.pretrained_path = None
        return meta
    explicit = args.pretrained_path is not None
    if not explicit:
        args.pretrained_path = str(
            PRETRAINED_DIR / f"spectral_mlp_f{args.fold}_s{args.seed}.pt")
    p = Path(args.pretrained_path)
    if not p.exists():
        raise SystemExit(f"missing pretrained spectral MLP: {p} "
                         f"(run code/experiments/pretrain_spectral_mlp.py)")
    ck = load_pretrained_ckpt(p)
    mismatch = []
    for key, want, cast in (("seed", args.seed, int), ("fold", args.fold, int),
                            ("data_dir", args.data_dir, str)):
        if key not in ck:
            print(f"[warn] pretrained checkpoint carries no '{key}' field; "
                  f"cannot validate it ({p})", flush=True)
            continue
        if cast(ck[key]) != cast(want):
            mismatch.append(f"{key}: ckpt={ck[key]!r} run={want!r}")
    if mismatch:
        msg = (f"pretrained checkpoint does not match this run ({p}): "
               + "; ".join(mismatch))
        if args.allow_pretrained_mismatch:
            print(f"[warn] {msg}  [--allow_pretrained_mismatch]", flush=True)
        else:
            raise SystemExit(msg + "  [pass --allow_pretrained_mismatch to "
                                   "override deliberately]")

    # ---- WHICH split actually selected the pretrain epoch -----------------
    # v2-fix: this was hard-coded to f"val_fold{args.fold}" and the caveat
    # asserted unconditionally that theta had been selected on the val cores
    # THIS run scores. Under --allow_pretrained_mismatch that is false, and
    # false in the dangerous direction: another fold's checkpoint was FITTED
    # on cores that are in this run's val split (k-fold val sets are
    # disjoint, so fold-A's TRAIN set contains fold-B's VAL cores). Derive
    # the selection split from the CHECKPOINT and describe the
    # situation that actually holds.
    ck_fold = int(ck["fold"]) if "fold" in ck else None
    ck_data_dir = str(ck["data_dir"]) if "data_dir" in ck else None
    ck_wd = float(ck["weight_decay"]) if "weight_decay" in ck else None
    if ck_fold is None or ck_data_dir is None:
        same_split = None
        sel_split = None
        sel_caveat = (
            "UNKNOWN: the checkpoint carries no fold and/or data_dir, so "
            "which split selected the pretrain epoch cannot be determined; "
            "claim neither a val-selection advantage nor its absence")
    elif (ck_fold, ck_data_dir) == (args.fold, args.data_dir):
        same_split = True
        sel_split = f"val_fold{ck_fold} [{ck_data_dir}]"
        sel_caveat = (
            "pretrain_spectral_mlp.py picked this epoch by macro-F1 on the "
            "SAME fold val cores this run scores, so frozen_pretrained / "
            "finetune_real carry a val-selection advantage that joint_linear "
            "/ joint_mlp / frozen_random / frozen_pca do not; state it "
            "wherever those arms are compared")
    else:
        same_split = False
        sel_split = (f"val_fold{ck_fold} [{ck_data_dir}] -- NOT this run's "
                     f"val split (fold {args.fold} [{args.data_dir}])")
        sel_caveat = (
            "CROSS-CONDITION run (--allow_pretrained_mismatch): the pretrain "
            f"epoch was picked on val_fold{ck_fold} [{ck_data_dir}], NOT on "
            f"this run's fold-{args.fold} val cores, so the usual "
            "val-selection-advantage caveat does not apply. Check the other "
            "direction instead -- the pretrainer's TRAIN cores may contain "
            "cores that are in THIS run's val split (k-fold val sets are "
            "disjoint), which would be outright train-on-val leakage")
    if ck_wd is None:
        wd_caveat = ("the checkpoint records no weight_decay, so whether "
                     "theta was pretrained under a decay the runner's "
                     "protocol forbids (weight_decay_theta=0.0) is unknown")
    elif ck_wd == 0.0:
        wd_caveat = ("theta was pretrained under weight_decay=0.0, which "
                     "matches the runner's protocol (weight_decay_theta=0.0)")
    else:
        wd_caveat = (f"theta was pretrained under AdamW weight_decay={ck_wd}, "
                     "which the runner's own protocol forbids for theta "
                     "(weight_decay_theta=0.0)")
    meta.update(
        pretrained_path=str(p),
        pretrained_pixel_val_f1=float(ck["pixel_val_f1"]),
        pretrained_epoch=int(ck["epoch"]),
        pretrained_seed=int(ck["seed"]),
        # ---- provenance + the disclosed confounds (verifier round) --------
        pretrained_fold=ck_fold,
        pretrained_data_dir=ck_data_dir,
        pretrained_weight_decay=ck_wd,
        pretrained_mismatch=(mismatch or None),
        pretrained_selection_matches_run_val=same_split,
        pretrained_selection_split=sel_split,
        pretrained_selection_caveat=sel_caveat,
        pretrained_augment_caveat=(
            "pretraining flattened pixels with augment=False while the runner "
            "trains with augment=True (spatial flip/rot90 before the 336 "
            "center-crop), so for cores larger than 336 px the pretrainer "
            "never saw the border pixels the runner trains on"),
        pretrained_theta_weight_decay_caveat=wd_caveat,
    )
    return meta


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def bn_affine_names(model: nn.Module) -> set:
    """Fully-qualified names of the spectral reduction's BN affine params.

    Only BatchNorm modules INSIDE model.spectral_reduce, and only when
    affine=True. Returns e.g. {'spectral_reduce.norm.weight',
    'spectral_reduce.norm.bias'} for LinearSpectralReduction, and the EMPTY
    set for PCAReduction / MLPSpectralReduction (neither has a BatchNorm).
    """
    names = set()
    sr = getattr(model, "spectral_reduce", None)
    if sr is None:
        return names
    for mod_name, m in sr.named_modules():
        if isinstance(m, _BN_TYPES) and getattr(m, "affine", False):
            prefix = f"spectral_reduce.{mod_name}." if mod_name else "spectral_reduce."
            for pn, _ in m.named_parameters(recurse=False):
                names.add(prefix + pn)
    return names


def split_param_groups(model: nn.Module, arm: str,
                       bn_affine_mode: str = "arm_default"):
    """Return (theta, phi, frozen_extra) parameter lists.

    theta  : the f_theta parameters the theory tracks.
    phi    : everything that goes in the optimizer's decayed group.
    frozen_extra: params in NEITHER optimizer group (frozen arms only) whose
             grads must be zeroed manually every step — includes theta itself
             for frozen arms plus spectral_reduce.norm in frozen_random.

    bn_affine_mode != 'arm_default' short-circuits the per-arm rule for the
    spectral reduction's BN affine params ONLY (see module docstring); every
    other parameter follows the untouched per-arm logic below. With
    'arm_default' the routing set is empty, so the function is the pre-v2 one.
    """
    bn_names = bn_affine_names(model) if bn_affine_mode != "arm_default" else set()
    theta, phi, frozen_extra = [], [], []
    for name, p in model.named_parameters():
        if name in bn_names:
            # BN affine never belongs to theta (it is normalisation, not the
            # map f_theta); train_all -> optimizer's decayed group, freeze_all
            # -> out of the optimizer with its grads zeroed each step.
            (phi if bn_affine_mode == "train_all" else frozen_extra).append(p)
            continue
        in_sr = name.startswith("spectral_reduce")
        if arm == "joint_linear":
            # theta = the linear map only; the post-projection BatchNorm2d
            # (spectral_reduce.norm) is normalization, not the map -> phi.
            (theta if name.startswith("spectral_reduce.proj") else phi).append(p)
        elif arm in ("joint_mlp", "finetune_real"):
            (theta if in_sr else phi).append(p)
        elif arm in ("frozen_random", "frozen_pca"):
            if name.startswith("spectral_reduce.proj"):
                theta.append(p)
                frozen_extra.append(p)
            elif in_sr:
                frozen_extra.append(p)   # e.g. spectral_reduce.norm (BN2d)
            else:
                phi.append(p)
        elif arm == "frozen_pretrained":
            # theta = the whole MLP (as in joint_mlp); frozen (as in
            # frozen_random): out of the optimizer, counterfactual grads
            # logged, zeroed manually. The MLP has no BN, so nothing else
            # lands in frozen_extra.
            if in_sr:
                theta.append(p)
                frozen_extra.append(p)
            else:
                phi.append(p)
        else:
            raise ValueError(f"unknown arm {arm}")
    return theta, phi, frozen_extra


def base_lr(args) -> float:
    return 1e-4 if args.optimizer == "adamw" else 1e-2


def build_optimizer(args, theta, phi, arm):
    groups = []
    if arm in TRAINABLE_THETA_ARMS:
        g = {"params": theta, "weight_decay": 0.0}   # non-negotiable
        if args.spectral_lr_mult != 1.0:
            # theta-only lr; phi's group keeps the optimizer default. The key
            # is added ONLY for a non-unit multiplier so the mult=1.0 groups
            # are exactly the pre-E3c-local dicts.
            g["lr"] = base_lr(args) * args.spectral_lr_mult
        groups.append(g)
    elif args.spectral_lr_mult != 1.0:
        print(f"[warn] --spectral_lr_mult={args.spectral_lr_mult} has no effect "
              f"for frozen arm {arm} (theta is not in the optimizer)", flush=True)
    groups.append({"params": phi, "weight_decay": 0.01})
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=1e-4)
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=1e-2, momentum=0.9)
    raise ValueError(f"unknown optimizer {args.optimizer}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def group_l2_disp(params, ref):
    """||p(t) - p(0)|| over a param group (concatenated L2)."""
    s = 0.0
    with torch.no_grad():
        for p, p0 in zip(params, ref):
            s += float((p.detach() - p0).pow(2).sum().item())
    return s ** 0.5


def group_l2_norm(params):
    s = 0.0
    with torch.no_grad():
        for p in params:
            s += float(p.detach().pow(2).sum().item())
    return s ** 0.5


def snapshot_into(bufs, params):
    """bufs[i] <- params[i] (in place, no allocation). Pre-step snapshot."""
    with torch.no_grad():
        for b, p in zip(bufs, params):
            b.copy_(p.detach())


def update_norm(params, bufs) -> float:
    """||p_after - p_before||_2 over a group, one host sync (float64 accum).

    Exactly 0.0 for a group the optimizer does not own (frozen theta): the
    snapshot and the parameter are then the same numbers bit for bit.
    """
    if not params:
        return 0.0
    with torch.no_grad():
        acc = torch.zeros((), device=params[0].device, dtype=torch.float64)
        for p, b in zip(params, bufs):
            acc += (p.detach() - b).double().pow(2).sum()
        return float(acc.sqrt().item())


def grad_total_norm(params) -> torch.Tensor:
    """Total L2 grad norm over params, matching clip_grad_norm_'s value.

    Used only for --clip_scope none, where nothing is clipped but the column
    must still be populated.
    """
    norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
    if not norms:
        return torch.zeros(())
    return torch.norm(torch.stack(norms), 2)


def probe_features(model, probe_x, device):
    """Z = spectral_reduce(probe) under no_grad, in TRAIN-mode functional form.

    Train mode matches what g_phi sees during optimization (batch-stat BN for
    the linear arms). BN momentum is zeroed for the call so the probe forward
    never contaminates running statistics, then restored. Deterministic given
    parameters + the fixed probe batch, so a frozen arm gives feat_disp == 0
    under --bn_affine_mode arm_default / freeze_all. NOT under train_all: the
    reduction's BN affine is then in phi and trains, so a frozen arm's probe
    features DO move and feat_disp is no longer a theta-displacement proxy.
    """
    sr = model.spectral_reduce
    was_training = sr.training
    saved = []
    for m in sr.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            saved.append((m, m.momentum))
            m.momentum = 0.0
    sr.train()
    with torch.no_grad():
        Z = sr(probe_x.to(device))
    for m, mom in saved:
        m.momentum = mom
    sr.train(was_training)
    return Z


def input_jac_opnorm(model, Z, n_iter=15, seed=0):
    """||d g_phi / d Z||_op at probe features Z, via power iteration on J^T J.

    g_phi = the model tail after spectral_reduce; BlockViTv2.forward already
    routes 4D input (B, 64, H, W) past spectral_reduce, so tail(Z) = model(Z).
    JVP is computed with the double-backward trick (u = d(J^T r)/dr . v) —
    torch.func.jvp is avoided because forward-mode AD through the fused
    attention paths of nn.TransformerEncoder is not reliably supported.
    The forward runs under the MATH SDPA backend: the fused efficient/flash
    attention backwards have no second derivative ("derivative for
    aten::_scaled_dot_product_efficient_attention_backward is not
    implemented"), while the math backend is fully differentiable.
    Evaluated with the tail in eval() mode (deterministic BN, dropout=0).
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    was_training = model.training
    model.eval()
    Zl = Z.detach().requires_grad_(True)
    with sdpa_kernel([SDPBackend.MATH]):
        y = model(Zl)                                # (B, 4, H, W)
        r = torch.zeros_like(y, requires_grad=True)  # dual seed for JVP trick
        g = torch.autograd.grad(y, Zl, grad_outputs=r,
                                create_graph=True)[0]                        # J^T r

        gen = torch.Generator(device="cpu").manual_seed(seed)
        v = torch.randn(Zl.shape, generator=gen).to(Zl.device)
        v = v / v.norm()
        sigma = 0.0
        for _ in range(n_iter):
            u = torch.autograd.grad(g, r, grad_outputs=v,
                                    retain_graph=True)[0]                    # J v
            sigma = float(u.norm().item())           # v is unit: sigma = ||Jv||
            w = torch.autograd.grad(y, Zl, grad_outputs=u.detach(),
                                    retain_graph=True)[0]                    # J^T J v
            wn = w.norm().clamp_min(1e-30)
            v = (w / wn).detach()
    del y, r, g
    model.train(was_training)
    return sigma


def residual_rms(logits, labels):
    """RMS per-pixel residual over valid pixels: sqrt(mean ||softmax - onehot||^2)."""
    with torch.no_grad():
        valid = labels != 255
        n = int(valid.sum().item())
        if n == 0:
            return float("nan"), 0
        p = torch.softmax(logits, dim=1).permute(0, 2, 3, 1)[valid]   # (N, C)
        oh = F.one_hot(labels[valid], num_classes=NUM_CLASSES).float()
        r = (p - oh).pow(2).sum(dim=1).mean().sqrt()
        return float(r.item()), n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    total_loss, total_px = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        valid = y != 255
        n = int(valid.sum().item())
        if n == 0:
            continue
        logits = model(x)
        loss = F.cross_entropy(logits, y, ignore_index=255)
        total_loss += float(loss.item()) * n
        total_px += n
        preds.append(logits.argmax(1)[valid].cpu().numpy())
        labs.append(y[valid].cpu().numpy())
    preds, labs = np.concatenate(preds), np.concatenate(labs)
    return {
        "loss": total_loss / max(total_px, 1),
        "macro_f1": f1_score(labs, preds, average="macro",
                             labels=list(range(NUM_CLASSES)), zero_division=0),
        "acc": accuracy_score(labs, preds),
    }


def append_csv(path: Path, rows, fieldnames):
    # Re-mkdir defensively: the out tree is shared and external cleanup racing
    # a write has been observed to FileNotFoundError an otherwise healthy run.
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            wr.writeheader()
        wr.writerows(rows)
        f.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["joint_linear", "frozen_random", "frozen_pca",
                             "joint_mlp", "frozen_pretrained", "finetune_real"])
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spectral_lr_mult", type=float, default=1.0,
                    help="multiplies the THETA group's lr (joint_linear / "
                         "joint_mlp / finetune_real only); phi lr untouched")
    ap.add_argument("--pretrained_path", default=None,
                    help="pretrained arms: override the per-seed default "
                         "experiments_shortcut/pretrained/spectral_mlp_f{fold}_s{seed}.pt")
    ap.add_argument("--allow_pretrained_mismatch", action="store_true",
                    help="downgrade the pretrained-checkpoint seed/fold/"
                         "data_dir checks to warnings (deliberate "
                         "cross-condition study only)")
    ap.add_argument("--run_label", default=None,
                    help="output-dir token in place of the arm name "
                         "(e.g. joint_speclr0.1); default = arm")
    # ---- matched-hygiene options (v2). Defaults == pre-v2 behaviour. -------
    ap.add_argument("--save_best", action="store_true",
                    help="also write <run>/best.pt at the best val macro-F1 "
                         "epoch (atomic) and record best_epoch / best_val_f1 "
                         "in config.json")
    ap.add_argument("--bn_affine_mode", default="arm_default",
                    choices=["arm_default", "train_all", "freeze_all"],
                    help="routing of the spectral reduction's BatchNorm affine "
                         "params: per-arm (default) / phi in every arm / "
                         "frozen in every arm")
    ap.add_argument("--clip_scope", default="optimizer",
                    choices=["optimizer", "phi_only", "none"],
                    help="grad-clip scope: all optimizer params (default) / "
                         "phi only, theta left unclipped / no clipping")
    ap.add_argument("--lr_schedule", default="none", choices=["none", "cosine"],
                    help="none = constant lr (default); cosine = per-epoch "
                         "cosine decay of every group's lr over --epochs")
    ap.add_argument("--data_dir", default="/mnt/hdd2/u37314kd/data_breast_v2_pca23")
    ap.add_argument("--dataset_name", default="breast")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--out_root",
                    default=str(THEORY_ROOT / "experiments_shortcut" / "e3c"))
    ap.add_argument("--max_train_cores", type=int, default=0,
                    help="if >0, trim the split to this many train cores + 4 "
                         "val cores (smoke tests); 0 = full split")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.width % NUM_HEADS != 0:
        raise SystemExit(f"width {args.width} not divisible by num_heads {NUM_HEADS}")
    device = torch.device(args.device)
    # isfinite first: `float('nan') <= 0` is False, so a bare `<= 0` guard let
    # --spectral_lr_mult nan through and NaN-poisoned theta from step 1.
    if not math.isfinite(args.spectral_lr_mult) or args.spectral_lr_mult <= 0:
        raise SystemExit("--spectral_lr_mult must be a finite number > 0, "
                         f"got {args.spectral_lr_mult}")
    pretrained_meta = resolve_pretrained(args)

    label = args.run_label or args.arm
    out_dir = (Path(args.out_root) / f"{args.dataset_name}_f{args.fold}"
               / f"{label}_h{args.width}_{args.optimizer}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_csv = out_dir / "steps.csv"
    epochs_csv = out_dir / "epochs.csv"
    for p in (steps_csv, epochs_csv):   # fresh run overwrites stale logs
        if p.exists():
            p.unlink()

    # ---- split (optionally trimmed for smoke tests, exp1_8-style) ----------
    split_file = Path(args.data_dir) / f"splits_fold{args.fold}.json"
    if not split_file.exists():
        raise SystemExit(f"missing split file: {split_file}")
    if args.max_train_cores > 0:
        full = json.loads(split_file.read_text())
        trimmed = {k: [] for k in full}
        trimmed["train"] = full["train"][: args.max_train_cores]
        trimmed["val"] = full["val"][:4]
        tmp = SCRATCH / (f"e3c_split_{args.dataset_name}_f{args.fold}"
                         f"_n{args.max_train_cores}.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(trimmed))
        print(f"[data] TRIMMED split -> {tmp} "
              f"({len(trimmed['train'])} train / {len(trimmed['val'])} val)",
              flush=True)
        split_file = tmp

    train_ds = CoreDataset(args.data_dir, str(split_file), "train",
                           spatial_size=SPATIAL, augment=False)
    val_ds = CoreDataset(args.data_dir, str(split_file), "val",
                         spatial_size=SPATIAL, augment=False)
    # PROTOCOL (2026-08-27): DataLoader workers and page-locking OFF -- the
    # per-step GB-scale numpy churn plus pinning caused a 98%-stime kernel
    # stall at full heap (27min user vs 21.7h kernel). Identical setting
    # across all arms, so arm comparisons are unaffected.
    # CORRECTION (verifier round): the two datasets are CONSTRUCTED with
    # augment=False only so the fixed probe batch below is un-augmented;
    # train_ds.augment is flipped back to True immediately after the probe
    # (see "Fixed probe batch"). The TRAIN loop therefore DOES run with
    # spatial augmentation (flip/rot90, applied before the 336 center-crop);
    # val_ds stays un-augmented. Identical across all arms. The pre-v2
    # comment claimed augmentation was off and was simply wrong.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    assert len(train_ds.wavenumbers) == NUM_SPECTRAL, \
        f"expected S={NUM_SPECTRAL}, got {len(train_ds.wavenumbers)}"

    # ---- model, arm-specific setup ----------------------------------------
    model = build_model(args).to(device)

    if args.arm == "frozen_pca":
        model.spectral_reduce.fit(train_ds.cores)   # run.py:~607 pattern
        # PCAReduction freezes proj in __init__; re-enable requires_grad so
        # COUNTERFACTUAL gradients flow (it stays out of the optimizer).
        for p in model.spectral_reduce.proj.parameters():
            p.requires_grad_(True)

    theta, phi, frozen_extra = split_param_groups(model, args.arm,
                                                  args.bn_affine_mode)
    optimizer = build_optimizer(args, theta, phi, args.arm)
    opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    egr = EGRLogger.from_param_groups(theta, phi, require_grad_only=False)

    # BN-affine accounting (reported for EVERY run, including arm_default).
    _named = dict(model.named_parameters())
    bn_params = [_named[n] for n in sorted(bn_affine_names(model))]

    def _bn_count(group):
        ids = {id(p) for p in group}
        return sum(p.numel() for p in bn_params if id(p) in ids)

    bn_counts = dict(
        n_bn_affine_total=sum(p.numel() for p in bn_params),
        n_bn_affine_theta=_bn_count(theta),
        n_bn_affine_phi=_bn_count(phi),
        n_bn_affine_frozen=_bn_count(frozen_extra),
    )

    # Per-step update-norm snapshots: allocated once, refilled in place.
    theta_snap = [torch.empty_like(p) for p in theta]
    phi_snap = [torch.empty_like(p) for p in phi]
    clip_params = phi if args.clip_scope == "phi_only" else opt_params
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    C_f = sum(p.numel() for p in theta)
    C_g = sum(p.numel() for p in phi)
    C_total = sum(p.numel() for p in model.parameters())

    # ---- reference states (AFTER any PCA fit, BEFORE training) ------------
    theta0 = [p.detach().clone() for p in theta]
    phi0 = [p.detach().clone() for p in phi]
    theta0_norm = group_l2_norm(theta0)
    phi0_norm = group_l2_norm(phi0)

    # Fixed probe batch: first 2 train cores, un-augmented, kept on CPU.
    train_ds.augment = False
    probe_x = torch.stack([train_ds[i][0] for i in range(min(2, len(train_ds)))])
    train_ds.augment = True
    Z0 = probe_features(model, probe_x, device)
    Z0_norm = float(Z0.norm().item())

    config = dict(vars(args))
    config.update(
        C_f=C_f, C_g=C_g, C_total=C_total,
        n_frozen_extra=sum(p.numel() for p in frozen_extra),
        n_train_cores=len(train_ds), n_val_cores=len(val_ds),
        num_heads=NUM_HEADS, num_classes=NUM_CLASSES,
        num_spectral=NUM_SPECTRAL, spatial_size=SPATIAL,
        lr=1e-4 if args.optimizer == "adamw" else 1e-2,
        weight_decay_phi=0.01, weight_decay_theta=0.0,
        # E3c-local provenance: theta lr actually used (None when theta is
        # frozen), the multiplier, and the pretrained encoder (if any).
        spectral_lr_mult=args.spectral_lr_mult,
        spectral_lr=(base_lr(args) * args.spectral_lr_mult
                     if args.arm in TRAINABLE_THETA_ARMS else None),
        run_label=label,
        **pretrained_meta,
        # matched-hygiene provenance (v2)
        **bn_counts,
        clip_max_norm=CLIP_MAX_NORM,
        base_lrs=base_lrs,
        theta0_norm=theta0_norm, phi0_norm=phi0_norm, Z0_fro_norm=Z0_norm,
        torch_version=torch.__version__,
        device_name=(torch.cuda.get_device_name(0)
                     if device.type == "cuda" else "cpu"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)   # tree may be cleaned externally
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[cfg] arm={args.arm} h={args.width} {args.optimizer} seed={args.seed} "
          f"| C_f={C_f:,} C_g={C_g:,} total={C_total:,} -> {out_dir}", flush=True)
    if (args.bn_affine_mode != "arm_default" or args.clip_scope != "optimizer"
            or args.lr_schedule != "none" or args.save_best):
        print(f"[hyg] bn_affine_mode={args.bn_affine_mode} "
              f"clip_scope={args.clip_scope} lr_schedule={args.lr_schedule} "
              f"save_best={args.save_best} | bn_affine total="
              f"{bn_counts['n_bn_affine_total']} theta="
              f"{bn_counts['n_bn_affine_theta']} phi="
              f"{bn_counts['n_bn_affine_phi']} frozen="
              f"{bn_counts['n_bn_affine_frozen']} | base_lrs={base_lrs}",
              flush=True)
    if args.spectral_lr_mult != 1.0 or pretrained_meta["pretrained_path"]:
        print(f"[cfg] spectral_lr_mult={args.spectral_lr_mult} "
              f"spectral_lr={config['spectral_lr']} "
              f"pretrained={pretrained_meta['pretrained_path']} "
              f"(pixel_val_f1={pretrained_meta['pretrained_pixel_val_f1']})",
              flush=True)
    if pretrained_meta["pretrained_path"]:
        # Loud, per-run disclosure of the confounds recorded in config.json —
        # these arms are NOT a clean contrast. v2-fix: read the strings from
        # pretrained_meta (derived from the CHECKPOINT) instead of restating
        # args.fold, which was false whenever --allow_pretrained_mismatch
        # loaded another fold's / another dataset's theta.
        print(f"[bias] pretrain epoch selected on "
              f"{pretrained_meta['pretrained_selection_split']}", flush=True)
        for key in ("pretrained_selection_caveat",
                    "pretrained_theta_weight_decay_caveat",
                    "pretrained_augment_caveat"):
            print(f"[bias] {pretrained_meta[key]}", flush=True)
        if pretrained_meta.get("pretrained_mismatch"):
            print("[bias] checkpoint/run mismatch overridden: "
                  + "; ".join(pretrained_meta["pretrained_mismatch"]),
                  flush=True)

    step_fields = ["step", "epoch", "loss", "n_valid_px",
                   "grad_theta_norm", "grad_phi_norm", "egr", "r_rms",
                   # v2 additions, appended so the existing columns keep
                   # their names AND their positions
                   "grad_total_norm_preclip", "clip_coef",
                   "upd_theta_norm", "upd_phi_norm"]
    epoch_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1",
                    "val_acc", "theta_disp", "theta_disp_rel", "phi_disp",
                    "phi_disp_rel", "feat_disp", "feat_disp_rel",
                    "input_jac_op", "epoch_secs"]

    # ---- training loop -----------------------------------------------------
    step = 0
    best_val_f1, best_epoch = -float("inf"), None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.lr_schedule == "cosine":
            # per-EPOCH cosine over the whole run; factor 1.0 at epoch 1.
            f = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1)
                                      / max(args.epochs, 1)))
            for g, b in zip(optimizer.param_groups, base_lrs):
                g["lr"] = b * f
        model.train()
        step_rows = []
        tr_loss_sum, tr_px = 0.0, 0
        last_egr, last_rrms = float("nan"), float("nan")

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if int((y != 255).sum().item()) == 0:
                continue
            logits = model(x)
            loss = F.cross_entropy(logits, y, ignore_index=255)
            rrms, n_valid = residual_rms(logits, y)   # no_grad, pre-backward
            loss.backward()
            rec = egr.log_step(step)                  # AFTER backward, pre-zero
            if args.clip_scope == "none":
                total_norm = float(grad_total_norm(clip_params))
                clip_coef = 1.0
            else:
                # 'optimizer' passes exactly the pre-v2 argument list; the
                # return value was previously discarded, reading it changes
                # nothing about the gradients.
                total_norm = float(nn.utils.clip_grad_norm_(
                    clip_params, CLIP_MAX_NORM))       # AFTER EGR logging
                # torch applies clamp(max_norm/(total+1e-6), max=1.0); mirror
                # it exactly. v2-fix: a bare min() logged 1.0 for a NaN norm
                # (`nan < 1.0` is False) while torch had multiplied every grad
                # by NaN — the column must show the blow-up, not hide it.
                _cc = CLIP_MAX_NORM / (total_norm + 1e-6)
                clip_coef = _cc if math.isnan(_cc) else min(1.0, _cc)
            snapshot_into(theta_snap, theta)
            snapshot_into(phi_snap, phi)
            optimizer.step()
            upd_theta = update_norm(theta, theta_snap)
            upd_phi = update_norm(phi, phi_snap)
            optimizer.zero_grad(set_to_none=False)
            # zero_grad only touches optimizer params; frozen arms must zero
            # the counterfactual theta (+ any other non-optimizer) grads too.
            for p in frozen_extra:
                if p.grad is not None:
                    p.grad.zero_()

            step_rows.append(dict(
                step=step, epoch=epoch, loss=float(loss.item()),
                n_valid_px=n_valid,
                grad_theta_norm=rec["grad_theta_norm"],
                grad_phi_norm=rec["grad_phi_norm"],
                egr=rec["egr"], r_rms=rrms,
                grad_total_norm_preclip=total_norm, clip_coef=clip_coef,
                upd_theta_norm=upd_theta, upd_phi_norm=upd_phi,
            ))
            last_egr, last_rrms = rec["egr"], rrms
            tr_loss_sum += float(loss.item()) * n_valid
            tr_px += n_valid
            step += 1

        append_csv(steps_csv, step_rows, step_fields)   # flush every epoch
        train_loss = tr_loss_sum / max(tr_px, 1)

        val = evaluate(model, val_loader, device)
        if args.save_best and val["macro_f1"] > best_val_f1:
            best_val_f1, best_epoch = float(val["macro_f1"]), epoch
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "best.pt.tmp"
            torch.save({
                "epoch": epoch,
                "val_macro_f1": best_val_f1,
                "model_state_dict": model.state_dict(),
                "theta0": [t.cpu() for t in theta0],
                "config": config,
            }, tmp)
            os.replace(tmp, out_dir / "best.pt")   # atomic within the dir
        th_d = group_l2_disp(theta, theta0)
        ph_d = group_l2_disp(phi, phi0)
        Zt = probe_features(model, probe_x, device)
        feat_d = float((Zt - Z0).norm().item())
        del Zt

        jac_op = ""
        if epoch % 5 == 0 or epoch == 1:
            Zt = probe_features(model, probe_x, device)
            jac_op = input_jac_opnorm(model, Zt, n_iter=15, seed=args.seed)
            del Zt
            torch.cuda.empty_cache()

        secs = time.time() - t0
        append_csv(epochs_csv, [dict(
            epoch=epoch, train_loss=train_loss, val_loss=val["loss"],
            val_macro_f1=val["macro_f1"], val_acc=val["acc"],
            theta_disp=th_d, theta_disp_rel=th_d / max(theta0_norm, 1e-30),
            phi_disp=ph_d, phi_disp_rel=ph_d / max(phi0_norm, 1e-30),
            feat_disp=feat_d, feat_disp_rel=feat_d / max(Z0_norm, 1e-30),
            input_jac_op=jac_op, epoch_secs=round(secs, 2),
        )], epoch_fields)

        jac_str = f" jac_op={jac_op:.4g}" if jac_op != "" else ""
        # log-only; epochs.csv keeps its pre-v2 columns exactly
        lr_str = ("" if args.lr_schedule == "none" else
                  "  lr=" + ",".join(f"{g['lr']:.4g}"
                                     for g in optimizer.param_groups))
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_f1={val['macro_f1']:.4f}  egr={last_egr:.4g}  "
              f"r_rms={last_rrms:.4f}  th_disp_rel={th_d / max(theta0_norm, 1e-30):.4g}"
              f"{jac_str}{lr_str}  ({secs:.0f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)   # tree may be cleaned externally
    if args.save_best:
        # best_epoch / best_val_f1 appear ONLY for --save_best runs, so a
        # default run's config.json never advertises a checkpoint it does not
        # have (analyze_e3c.py reads both with .get -> NaN).
        # None (not -inf) if no epoch ever ran: -inf is not valid JSON.
        config.update(best_epoch=best_epoch,
                      best_val_f1=(best_val_f1 if best_epoch is not None
                                   else None))
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        if best_epoch is None:
            print("[best] no epoch completed; best.pt not written", flush=True)
        else:
            print(f"[best] epoch {best_epoch} val_macro_f1={best_val_f1:.4f} "
                  f"-> {out_dir / 'best.pt'}", flush=True)
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "theta0": [t.cpu() for t in theta0],
        "config": config,
    }, out_dir / "final.pt")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[done] {out_dir}  peak_gpu={peak:.2f} GB", flush=True)
    else:
        print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
