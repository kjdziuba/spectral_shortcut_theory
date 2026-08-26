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

Optimizer (non-negotiable, review round): theta params sit in their OWN param
group with weight_decay=0.0 — a decay term on theta breaks the Theorem-2 flow.
phi gets weight_decay=0.01. AdamW lr=1e-4 (default) or SGD momentum=0.9 lr=1e-2.
Constant LR, fixed --epochs, NO early stopping.

Usage:
  python code/experiments/exp1_7_train.py --arm joint_linear --width 192 --seed 0
  # smoke:
  python code/experiments/exp1_7_train.py --arm frozen_random --width 48 \
      --seed 0 --epochs 2 --max_train_cores 8
"""
from __future__ import annotations

import argparse
import csv
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
    return model


def split_param_groups(model: nn.Module, arm: str):
    """Return (theta, phi, frozen_extra) parameter lists.

    theta  : the f_theta parameters the theory tracks.
    phi    : everything that goes in the optimizer's decayed group.
    frozen_extra: params in NEITHER optimizer group (frozen arms only) whose
             grads must be zeroed manually every step — includes theta itself
             for frozen arms plus spectral_reduce.norm in frozen_random.
    """
    theta, phi, frozen_extra = [], [], []
    for name, p in model.named_parameters():
        in_sr = name.startswith("spectral_reduce")
        if arm == "joint_linear":
            # theta = the linear map only; the post-projection BatchNorm2d
            # (spectral_reduce.norm) is normalization, not the map -> phi.
            (theta if name.startswith("spectral_reduce.proj") else phi).append(p)
        elif arm == "joint_mlp":
            (theta if in_sr else phi).append(p)
        elif arm in ("frozen_random", "frozen_pca"):
            if name.startswith("spectral_reduce.proj"):
                theta.append(p)
                frozen_extra.append(p)
            elif in_sr:
                frozen_extra.append(p)   # e.g. spectral_reduce.norm (BN2d)
            else:
                phi.append(p)
        else:
            raise ValueError(f"unknown arm {arm}")
    return theta, phi, frozen_extra


def build_optimizer(args, theta, phi, arm):
    groups = []
    if arm in ("joint_linear", "joint_mlp"):
        groups.append({"params": theta, "weight_decay": 0.0})   # non-negotiable
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


def probe_features(model, probe_x, device):
    """Z = spectral_reduce(probe) under no_grad, in TRAIN-mode functional form.

    Train mode matches what g_phi sees during optimization (batch-stat BN for
    the linear arms). BN momentum is zeroed for the call so the probe forward
    never contaminates running statistics, then restored. Deterministic given
    parameters + the fixed probe batch, so frozen arms give feat_disp == 0.
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
                             "joint_mlp"])
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
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

    out_dir = (Path(args.out_root) / f"{args.dataset_name}_f{args.fold}"
               / f"{args.arm}_h{args.width}_{args.optimizer}_s{args.seed}")
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
                           spatial_size=SPATIAL, augment=True)
    val_ds = CoreDataset(args.data_dir, str(split_file), "val",
                         spatial_size=SPATIAL, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
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

    theta, phi, frozen_extra = split_param_groups(model, args.arm)
    optimizer = build_optimizer(args, theta, phi, args.arm)
    opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    egr = EGRLogger.from_param_groups(theta, phi, require_grad_only=False)

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
        theta0_norm=theta0_norm, phi0_norm=phi0_norm, Z0_fro_norm=Z0_norm,
        torch_version=torch.__version__,
        device_name=(torch.cuda.get_device_name(0)
                     if device.type == "cuda" else "cpu"),
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[cfg] arm={args.arm} h={args.width} {args.optimizer} seed={args.seed} "
          f"| C_f={C_f:,} C_g={C_g:,} total={C_total:,} -> {out_dir}", flush=True)

    step_fields = ["step", "epoch", "loss", "n_valid_px",
                   "grad_theta_norm", "grad_phi_norm", "egr", "r_rms"]
    epoch_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1",
                    "val_acc", "theta_disp", "theta_disp_rel", "phi_disp",
                    "phi_disp_rel", "feat_disp", "feat_disp_rel",
                    "input_jac_op", "epoch_secs"]

    # ---- training loop -----------------------------------------------------
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
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
            nn.utils.clip_grad_norm_(opt_params, 1.0)  # AFTER EGR logging
            optimizer.step()
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
            ))
            last_egr, last_rrms = rec["egr"], rrms
            tr_loss_sum += float(loss.item()) * n_valid
            tr_px += n_valid
            step += 1

        append_csv(steps_csv, step_rows, step_fields)   # flush every epoch
        train_loss = tr_loss_sum / max(tr_px, 1)

        val = evaluate(model, val_loader, device)
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
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_f1={val['macro_f1']:.4f}  egr={last_egr:.4g}  "
              f"r_rms={last_rrms:.4f}  th_disp_rel={th_d / max(theta0_norm, 1e-30):.4g}"
              f"{jac_str}  ({secs:.0f}s)", flush=True)

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
