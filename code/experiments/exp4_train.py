"""Experiment 4, Stage B: joint training on NATURAL 3x3 neighbourhoods (REFOCUS_PLAN section 8.2).

Model as Exp 3: linear encoder S->K per pixel (K = 12), ReLU PatchHead over the nine encoded
spectra (width M), four logits, class-weighted mean cross-entropy (weights proportional to
inverse class frequency, normalized to mean one), full-batch plain GD with one global rate,
declared head-rate multiplier in the `headlr` arm. Paired initialization within seed.

Arms (training context):
  nat      natural neighbours                                  (informative context)
  nat16    natural neighbours, every pixel projected to the top 16 PCs of the training centres
  shuf     neighbours replaced by random training centres, independent of the centre label
           (the uninformative-context comparator, jointly trained)
  shuf16   shuf with the 16-PC projection
  frozen   random encoder frozen, head trained on natural neighbours
  headlr   nat with head-rate multiplier kappa (secondary arm)

Evaluation (validation and test patients separately; every centre held fixed):
  iid          natural patches
  ctx_random   the eight neighbours replaced by random centres from OTHER cores of the same side
  homogeneous  all nine pixels equal to the centre (secondary)
Probe: four-class LDA on the encoder output of centre spectra, fitted on held-out training
centres, scored on evaluation centres (macro-F1) - linear accessibility of the class signal.
Recovery (--recover): a fresh paired-init head trained for 20,000 GD steps on ctx_random
TRAINING patches from a frozen saved encoder, scored on ctx_random evaluation patches.

Outputs: results/exp4/snap_<tag>.csv, traj_<tag>.csv, enc_<tag>.npz; results/exp4_summary.csv;
results/exp4/recovery_summary.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "experiments"))
import exp4_natural_context as nc  # noqa: E402

CLASSES, CLASS_ORDER, S_FEAT, NC = nc.CLASSES, nc.CLASS_ORDER, nc.S_FEAT, nc.NC   # re-bound in main() after set_dataset


def summary_path() -> Path:
    return nc.OUT_DIR.parent / f"{nc.OUT_DIR.name}_summary.csv"
K = 12
N_PC = 16
LR = 1e-3
MAX_STEPS = 40_000
LOSS_STAR = 0.30
THRESHOLDS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.10]
STOP_LOSS = 0.10
DIVERGE_LOSS = 5.0
LOG_EVERY = 25
N_PROBE_FIT = 4_000
N_EVAL_MAX = 6_000
WIDTH = 32
SEEDS = [0, 1, 2]
HEAD_MULT_SECONDARY = 1 / 256
RECOVERY_STEPS = 20_000
CONDITIONS = ("iid", "ctx_random", "homogeneous")
ARMS = {
    "nat":    dict(train_cond="natural", pcs=False, frozen=False),
    "nat16":  dict(train_cond="natural", pcs=True,  frozen=False),
    "shuf":   dict(train_cond="shuffled", pcs=False, frozen=False),
    "shuf16": dict(train_cond="shuffled", pcs=True,  frozen=False),
    "frozen": dict(train_cond="natural", pcs=False, frozen=True),
    "headlr": dict(train_cond="natural", pcs=False, frozen=False),
    # no bottleneck: identity encoder (frozen), head reads the nine raw standardized spectra;
    # the analogue of O'Leary et al.'s "fixed" (no reduction) models  [amendment 8.5]
    "natfull":  dict(train_cond="natural", pcs=False, frozen=True, identity=True),
    "shuffull": dict(train_cond="shuffled", pcs=False, frozen=True, identity=True),
}


# ------------------------------------------------------------------ data ---
class Data:
    """Standardized natural patches on device, split by side; PCA basis of training centres."""

    def __init__(self, device):
        P, y, y9, core, side, rc = nc.load_cache()
        cls_index = {c: i for i, c in enumerate(CLASS_ORDER)}
        self.yi = torch.from_numpy(np.array([cls_index[int(v)] for v in y])).long()
        self.core = core; self.side = side
        tr = side == "train"
        Xc = P[tr, 4, :].astype(np.float64)
        self.mean = Xc.mean(0); self.std = Xc.std(0) + 1e-8
        Xs = (Xc - self.mean) / self.std
        _, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
        self.V16 = torch.from_numpy(np.ascontiguousarray(Vt[:N_PC].T)).float().to(device)   # (S, 16)
        self.evr16 = float((S[:N_PC] ** 2).sum() / (S ** 2).sum())
        self.P = {}
        for s in ("train", "val", "test"):
            m = side == s
            Ps = (P[m].astype(np.float32) - self.mean.astype(np.float32)) / self.std.astype(np.float32)
            self.P[s] = torch.from_numpy(Ps).to(device)
            setattr(self, f"y_{s}", self.yi[torch.from_numpy(m)].to(device))
            setattr(self, f"core_{s}", core[m])
        self.device = device
        n_tr = int(tr.sum())
        counts = torch.bincount(self.y_train, minlength=NC).float()
        w = counts.sum() / (NC * counts.clamp(min=1))
        self.class_w = (w / w.mean()).to(device)
        print(f"[data] train {n_tr} val {len(self.y_val)} test {len(self.y_test)}; class counts (train) "
              f"{[int(c) for c in counts]}; EVR of top {N_PC} PCs {self.evr16:.4f}", flush=True)

    def project(self, X: torch.Tensor, pcs: bool) -> torch.Tensor:
        return X @ self.V16 if pcs else X

    def shuffled(self, s: str, idx: torch.Tensor, g: torch.Generator, avoid_same_core: bool) -> torch.Tensor:
        """Patches of side `s` at `idx` with the eight neighbours replaced by random centres of side `s`
        (independent of the centre label); optionally from a different core."""
        P = self.P[s]; n = len(idx)
        src = torch.randint(0, len(P), (n, 8), generator=g)
        if avoid_same_core:
            core = getattr(self, f"core_{s}")
            for _ in range(5):
                same = torch.from_numpy(core[src.numpy()] == core[idx.numpy()][:, None])
                if not same.any():
                    break
                src[same] = torch.randint(0, len(P), (int(same.sum()),), generator=g)
        out = P[idx].clone()
        nb = P[src.to(P.device), 4, :]                       # (n, 8, S)
        out[:, [0, 1, 2, 3, 5, 6, 7, 8], :] = nb
        return out

    def homogeneous(self, s: str, idx: torch.Tensor) -> torch.Tensor:
        c = self.P[s][idx, 4, :]
        return c[:, None, :].expand(-1, 9, -1).contiguous()


# ----------------------------------------------------------------- model ---
class Encoder(nn.Module):
    def __init__(self, S: int, K: int):
        super().__init__()
        self.proj = nn.Linear(S, K, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(S))

    def forward(self, X):
        return self.proj(X)


class PatchHead(nn.Module):
    def __init__(self, K: int, M: int, n_cls: int | None = None):
        super().__init__()
        n_cls = n_cls or NC
        self.fc1 = nn.Linear(9 * K, M)
        self.fc2 = nn.Linear(M, n_cls, bias=False)

    def forward(self, Z):
        return self.fc2(F.relu(self.fc1(Z.reshape(Z.shape[0], -1))))


# --------------------------------------------------------------- metrics ---
@torch.no_grad()
def macro_f1(pred: torch.Tensor, y: torch.Tensor, n_cls: int | None = None) -> tuple[float, list[float]]:
    n_cls = n_cls or NC
    cm = torch.zeros(n_cls, n_cls, dtype=torch.float64, device=pred.device)
    cm.index_put_((y, pred), torch.ones_like(y, dtype=torch.float64), accumulate=True)
    tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1e-12), torch.zeros_like(tp))
    return float(f1.mean()), [float(v) for v in f1]


@torch.no_grad()
def lda4_fit(Z: torch.Tensor, y: torch.Tensor, ridge: float = 1e-4, n_cls: int | None = None):
    n_cls = n_cls or NC
    Z = Z.double(); mus = torch.stack([Z[y == c].mean(0) for c in range(n_cls)])
    R = torch.cat([Z[y == c] - mus[c] for c in range(n_cls)])
    Sw = R.T @ R / len(R); Sw = Sw + ridge * Sw.diag().mean() * torch.eye(len(Sw), device=Z.device, dtype=Z.dtype)
    Wd = torch.linalg.solve(Sw, mus.T)                        # (d, n_cls)
    b = -0.5 * (mus * Wd.T).sum(1)                            # (n_cls,)
    return Wd, b


@torch.no_grad()
def lda4_predict(Wd, b, Z: torch.Tensor) -> torch.Tensor:
    return (Z.double() @ Wd + b).argmax(1)


# ------------------------------------------------------------------ run ---
def run_one(arm: str, width: int, head_mult: float, seed: int, lr: float, max_steps: int, data: Data) -> pd.DataFrame:
    cfg = ARMS[arm]; device = data.device
    g = torch.Generator().manual_seed(7000 + seed)
    n_tr = len(data.y_train)
    perm = torch.randperm(n_tr, generator=g)
    probe_fit, train_idx = perm[:N_PROBE_FIT], perm[N_PROBE_FIT:]
    if cfg["train_cond"] == "natural":
        Xtr = data.P["train"][train_idx.to(device)]
    else:
        Xtr = data.shuffled("train", train_idx, torch.Generator().manual_seed(8000 + seed), avoid_same_core=False)
    Xtr = data.project(Xtr, cfg["pcs"]); ytr = data.y_train[train_idx.to(device)]
    S_in = Xtr.shape[-1]
    evals = {}
    for sd in ("val", "test"):
        n = len(getattr(data, f"y_{sd}"))
        gs = torch.Generator().manual_seed(9000 + seed)
        idx = torch.randperm(n, generator=gs)[:N_EVAL_MAX]
        ye = getattr(data, f"y_{sd}")[idx.to(device)]
        evals[(sd, "iid")] = (data.project(data.P[sd][idx.to(device)], cfg["pcs"]), ye)
        evals[(sd, "ctx_random")] = (data.project(data.shuffled(sd, idx, torch.Generator().manual_seed(9500 + seed), True), cfg["pcs"]), ye)
        evals[(sd, "homogeneous")] = (data.project(data.homogeneous(sd, idx), cfg["pcs"]), ye)
        evals[(sd, "centres")] = idx
    # probe data: encoder output of centre spectra (projected the same way as the arm's input)
    Zfit_in = data.project(data.P["train"][probe_fit.to(device), 4, :], cfg["pcs"]); yfit = data.y_train[probe_fit.to(device)]

    torch.manual_seed(seed); np.random.seed(seed)
    K_arm = S_in if cfg.get("identity") else K
    enc = Encoder(S_in, K_arm).to(device); head = PatchHead(K_arm, width).to(device)
    if cfg.get("identity"):
        with torch.no_grad():
            enc.proj.weight.copy_(torch.eye(S_in, device=device))
    if cfg["frozen"]:
        enc.proj.weight.requires_grad_(False)
    W0 = enc.proj.weight.detach().clone()

    def feat(X):  # identity arms: skip the (frozen, identity) 942x942 multiply
        return X if cfg.get("identity") else enc(X)
    groups = []
    if not cfg["frozen"]:
        groups.append({"params": [enc.proj.weight], "lr": lr})
    groups.append({"params": list(head.parameters()), "lr": lr * head_mult})
    opt = torch.optim.SGD(groups, momentum=0.0)
    h = f"_h{head_mult:g}" if head_mult != 1 else ""
    tag = f"{arm}_M{width}{h}_s{seed}"
    pending = sorted(THRESHOLDS, reverse=True)
    traj_rows, snap_rows, W_snaps = [], [], {}
    t0 = time.time(); status = "running"; gth = gph = float("nan")

    @torch.no_grad()
    def probe(sd: str) -> float:
        Wd, b = lda4_fit(feat(Zfit_in), yfit)
        idx = evals[(sd, "centres")].to(device)
        Zev = feat(data.project(data.P[sd][idx, 4, :], cfg["pcs"]))
        return macro_f1(lda4_predict(Wd, b, Zev), getattr(data, f"y_{sd}")[idx])[0]

    @torch.no_grad()
    def score(Xe, ye):
        pred = head(feat(Xe)).argmax(1)
        mf1, per = macro_f1(pred, ye)
        return float((pred == ye).float().mean()), mf1, per

    def snapshot(label, step, loss_v, acc_v, f1_v):
        Wt = enc.proj.weight.detach()
        row = dict(arm=arm, width=width, head_mult=head_mult, seed=seed, threshold=label, step=step, loss=loss_v,
                   acc_train=acc_v, f1_train=f1_v, gnorm_theta=gth, gnorm_phi=gph,
                   disp=float((Wt - W0).norm() / W0.norm()), wall_s=time.time() - t0,
                   probe_val=probe("val"), probe_test=probe("test"))
        for (sd, cond), val in evals.items():
            if cond == "centres":
                continue
            a, mf, per = score(*val)
            row[f"acc_{cond}_{sd}"] = a; row[f"f1_{cond}_{sd}"] = mf
            for i, c in enumerate(CLASS_ORDER):
                row[f"f1_{cond}_{sd}_{CLASSES[c]}"] = per[i]
        snap_rows.append(row)
        if not cfg.get("identity"):
            W_snaps[f"W_{label}"] = Wt.cpu().numpy().copy()
        print(f"  [{tag}] {label!s:>8} step {step:6d} loss {loss_v:.4f} f1tr {f1_v:.3f} probe_val {row['probe_val']:.3f} "
              f"val iid {row['f1_iid_val']:.3f} ctxrnd {row['f1_ctx_random_val']:.3f} homog {row['f1_homogeneous_val']:.3f}", flush=True)

    step = 0
    while True:
        opt.zero_grad(set_to_none=True)
        logits = head(feat(Xtr)); loss = F.cross_entropy(logits, ytr, weight=data.class_w)
        loss_v = float(loss)
        with torch.no_grad():
            pred = logits.argmax(1); acc_v = float((pred == ytr).float().mean()); f1_v = macro_f1(pred, ytr)[0]
        if not math.isfinite(loss_v) or loss_v > DIVERGE_LOSS:
            status = "diverged"; snapshot("diverged", step, loss_v, acc_v, f1_v); break
        loss.backward()
        gth = float(enc.proj.weight.grad.norm()) if enc.proj.weight.grad is not None else 0.0
        gph = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in head.parameters() if p.grad is not None))
        if step == 0:
            snapshot("init", 0, loss_v, acc_v, f1_v)
        while pending and loss_v < pending[0]:
            snapshot(pending.pop(0), step, loss_v, acc_v, f1_v)
        if loss_v < STOP_LOSS:
            status = "stopped"; snapshot("final", step, loss_v, acc_v, f1_v); break
        if step >= max_steps:
            status = "max_steps"; snapshot("final", step, loss_v, acc_v, f1_v); break
        if step % LOG_EVERY == 0:
            traj_rows.append(dict(step=step, loss=loss_v, acc_train=acc_v, f1_train=f1_v, gnorm_theta=gth, gnorm_phi=gph,
                                  disp=float((enc.proj.weight.detach() - W0).norm() / W0.norm())))
        opt.step(); step += 1

    snaps = pd.DataFrame(snap_rows); snaps["status"] = status; snaps["lr"] = lr
    snaps["reached_star"] = bool((snaps["threshold"] == LOSS_STAR).any())
    nc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(traj_rows).to_csv(nc.OUT_DIR / f"traj_{tag}.csv", index=False)
    snaps.to_csv(nc.OUT_DIR / f"snap_{tag}.csv", index=False)
    np.savez_compressed(nc.OUT_DIR / f"enc_{tag}.npz", W0=W0.cpu().numpy(), **W_snaps)
    print(f"[{tag}] {status} at step {step}, {time.time() - t0:.1f}s", flush=True)
    return snaps


# ------------------------------------------------------------- recovery ---
def recover(arm: str, width: int, head_mult: float, seed: int, enc_label: str, data: Data, lr: float = LR,
            steps: int = RECOVERY_STEPS) -> dict:
    """Fresh paired-init head trained on ctx_random TRAINING patches from a frozen saved encoder."""
    cfg = ARMS[arm]; device = data.device
    h = f"_h{head_mult:g}" if head_mult != 1 else ""
    tag = f"{arm}_M{width}{h}_s{seed}"
    path = nc.OUT_DIR / f"enc_{tag}.npz"
    if not path.exists():
        return dict(arm=arm, width=width, head_mult=head_mult, seed=seed, encoder=enc_label, status="missing")
    z = np.load(path)
    key = "W0" if enc_label == "init" else f"W_{enc_label}"
    if key not in z.files:
        return dict(arm=arm, width=width, head_mult=head_mult, seed=seed, encoder=enc_label, status="missing")
    W = torch.from_numpy(z[key]).to(device)
    g = torch.Generator().manual_seed(7000 + seed)
    perm = torch.randperm(len(data.y_train), generator=g)
    train_idx = perm[N_PROBE_FIT:]
    Xtr = data.project(data.shuffled("train", train_idx, torch.Generator().manual_seed(2500 + seed), False), cfg["pcs"])
    ytr = data.y_train[train_idx.to(device)]
    with torch.no_grad():
        Ztr = Xtr @ W.T
    torch.manual_seed(100 + seed)
    head = PatchHead(K, width).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.0)
    t0 = time.time(); loss_v = float("nan")
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(Ztr), ytr, weight=data.class_w); loss_v = float(loss)
        if loss_v < STOP_LOSS:
            break
        loss.backward(); opt.step()
    out = dict(arm=arm, width=width, head_mult=head_mult, seed=seed, encoder=enc_label, status="done",
               steps=step + 1, train_loss=loss_v, wall_s=time.time() - t0)
    with torch.no_grad():
        for sd in ("val", "test"):
            n = len(getattr(data, f"y_{sd}"))
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(9000 + seed))[:N_EVAL_MAX]
            ye = getattr(data, f"y_{sd}")[idx.to(device)]
            conds = {"ctx_random": data.shuffled(sd, idx, torch.Generator().manual_seed(9500 + seed), True),
                     "iid": data.P[sd][idx.to(device)], "homogeneous": data.homogeneous(sd, idx)}
            for cond, Xe in conds.items():
                pred = head(data.project(Xe, cfg["pcs"]) @ W.T).argmax(1)
                mf, per = macro_f1(pred, ye)
                out[f"acc_{cond}_{sd}"] = float((pred == ye).float().mean()); out[f"f1_{cond}_{sd}"] = mf
                for i, c in enumerate(CLASS_ORDER):
                    out[f"f1_{cond}_{sd}_{CLASSES[c]}"] = per[i]
    print(f"[recover {tag} enc={enc_label}] steps {out['steps']} loss {loss_v:.4f} val ctx_random f1 {out['f1_ctx_random_val']:.3f} "
          f"iid f1 {out['f1_iid_val']:.3f}", flush=True)
    return out


# --------------------------------------------------------------- collect ---
def collect() -> pd.DataFrame:
    rows = [pd.read_csv(f) for f in sorted(nc.OUT_DIR.glob("snap_*.csv"))]
    df = pd.concat(rows, ignore_index=True)
    init = df[df["threshold"] == "init"].set_index(["arm", "width", "head_mult", "seed"])
    for col in ["probe_val", "probe_test"]:
        df[col + "_gain"] = df[col] - init.loc[list(zip(df["arm"], df["width"], df["head_mult"], df["seed"])), col].values
    df.to_csv(summary_path(), index=False)
    star = df[df["threshold"] == str(LOSS_STAR)]
    cols = ["probe_val", "probe_val_gain", "f1_iid_val", "f1_ctx_random_val", "f1_homogeneous_val", "step"]
    print(f"[collect] {len(df)} rows -> {summary_path()}\nAt L* = {LOSS_STAR} (validation, mean over seeds):")
    print(star.groupby(["arm", "width", "head_mult"])[cols].mean().round(3).to_string())
    fin = df[df["threshold"] == "final"]
    print("At budget end / stop (validation, mean over seeds):")
    print(fin.groupby(["arm", "width", "head_mult"])[cols + ["loss"]].mean().round(3).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recover", action="store_true", help="head retraining from saved encoders (after the runs)")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs whose snap file exists")
    ap.add_argument("--copy", default=None, help="breast: nodenoise|pca23; paviau: raw")
    ap.add_argument("--dataset", default="breast", choices=list(nc.DATASETS))
    a = ap.parse_args()
    nc.set_dataset(a.dataset, a.copy)
    global CLASSES, CLASS_ORDER, S_FEAT, NC
    CLASSES, CLASS_ORDER, S_FEAT, NC = nc.CLASSES, nc.CLASS_ORDER, nc.S_FEAT, nc.NC
    print(f"[exp4_train] data copy {nc.COPY}: cache {nc.CACHE} -> {nc.OUT_DIR}", flush=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.collect:
        collect(); return
    data = Data(device)
    if a.recover:
        rows = []
        for seed in a.seeds:
            for arm, hm in [("nat", 1.0), ("headlr", HEAD_MULT_SECONDARY), ("nat16", 1.0), ("shuf", 1.0)]:
                for enc_label in ("init", str(LOSS_STAR), "final"):
                    if arm == "nat" and enc_label == "init":
                        rows.append(recover(arm, a.width, hm, seed, enc_label, data))
                    elif enc_label != "init":
                        rows.append(recover(arm, a.width, hm, seed, enc_label, data))
        pd.DataFrame(rows).to_csv(nc.OUT_DIR / "recovery_summary.csv", index=False)
        print(f"[recover] wrote {nc.OUT_DIR / 'recovery_summary.csv'}"); return
    for seed in a.seeds:
        for arm in a.arms:
            hm = HEAD_MULT_SECONDARY if arm == "headlr" else 1.0
            h = f"_h{hm:g}" if hm != 1 else ""
            if a.skip_existing and (nc.OUT_DIR / f"snap_{arm}_M{a.width}{h}_s{seed}.csv").exists():
                print(f"[exp4_train] skip {arm}_M{a.width}{h}_s{seed}", flush=True); continue
            run_one(arm, a.width, hm, seed, a.lr, a.max_steps, data)
    collect()


if __name__ == "__main__":
    main()
