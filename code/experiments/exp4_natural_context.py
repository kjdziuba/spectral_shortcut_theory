"""Experiment 4: natural context, usage test (REFOCUS_PLAN amendment section 8).

Stage A (this file): the premise check and the data.
  --build-cache  For the fold-0 train/val/test cores of the NO-DENOISING breast QCL copy
                 (full-rank spectra; the Exp 3 cache came from the PCA-23-denoised copy,
                 which is rank 23), sample labelled pixels whose 3x3 neighbourhood lies
                 inside the tissue mask and store the natural 3x3x942 patch, the 3x3 label
                 block, core id, split side and (row, col). Four classes, per core per class
                 caps 150/100/100 (train/val/test), seed 1234.
  --pixel-sweep  Per-pixel information sweep on the CENTRE spectra: per-feature
                 standardization and PCA fitted on training-patient centres; for each number
                 of principal components k, a shrinkage Fisher discriminant, a balanced
                 logistic regression and a small ReLU MLP (fixed schedule, no selection on
                 the evaluation patients) are fitted on training centres and scored on the
                 validation and test patients (four-class accuracy, macro-F1, per-class F1).
                 Writes results/exp4/pixel_sweep.csv and results/exp4/neighbourhood_stats.json.

Stage B (exp4_train.py): joint training on the natural patches, the context-random and
retraining measures, and the same sweep through the joint model.

Rationale: O'Leary et al. (2026) and Mueller et al. (2023) infer spectral redundancy from
the insensitivity of a spatial-spectral model's in-distribution accuracy to spectral
compression. Stage A asks whether a per-pixel classifier on these spectra gains from
components beyond the top sixteen at all (the premise); Stage B asks whether the jointly
trained spatial model exploits that gain (their observation) and whether it is recoverable
from the encoder (unused, not unnecessary).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import os
REPO = Path(__file__).resolve().parents[2]
FOLD = int(os.environ.get("EXP_FOLD", "0"))        # §12: five-fold extension; fold 0 = the registered paths
_F = "" if FOLD == 0 else f"_fold{FOLD}"
COPIES = {  # data copy -> (data directory, results directory). Same cores, same split file.
    "nodenoise": (Path("/mnt/hdd2/u37314kd/data_breast_v2_nodenoising"), REPO / "results" / f"exp4{_F}"),
    "pca23": (Path("/mnt/hdd2/u37314kd/data_breast_v2_pca23"), REPO / "results" / f"exp4_pca23{_F}"),
}
COPY = "nodenoise"
DATA_DIR, OUT_DIR = COPIES[COPY]
CACHE = OUT_DIR / "cache_fold0_natural.npz"
SPLITS = Path(f"/mnt/hdd2/u37314kd/data_breast_v2_pca23/splits_fold{FOLD}.json")  # same cores, same split
DATASETS = {  # dataset -> copies, classes, class order (index 0..NC-1), S, k grid (REFOCUS_PLAN §8 breast, §9 paviau)
    "breast": dict(copies=COPIES, classes={1: "NormalEpi", 2: "NormalStroma", 3: "CancerEpi", 4: "CAS"},
                   order=(3, 4, 2, 1), s_feat=942, ks=(2, 4, 8, 16, 23, 32, 64, 128, 256, 942)),
    "paviau": dict(copies={"raw": (None, REPO / "results" / "exp4_paviau")},
                   classes={1: "Asphalt", 2: "Meadows", 3: "Gravel", 4: "Trees", 5: "PaintedMetal", 6: "BareSoil",
                            7: "Bitumen", 8: "Bricks", 9: "Shadows"},
                   order=(1, 2, 3, 4, 5, 6, 7, 8, 9), s_feat=103, ks=(2, 4, 8, 16, 23, 32, 64, 103)),
}
DATASET = "breast"


def set_copy(name: str) -> None:
    """Select the data copy (breast: nodenoise = pre-registered Stage A/B; pca23 = replication 8.4; paviau: raw)."""
    global COPY, DATA_DIR, OUT_DIR, CACHE
    COPY = name
    DATA_DIR, OUT_DIR = COPIES[name]
    CACHE = OUT_DIR / "cache_fold0_natural.npz"


def set_dataset(name: str, copy: str | None = None) -> None:
    """Select the dataset (breast tissue, §8; Pavia University, §9) and one of its data copies."""
    global DATASET, COPIES, CLASSES, CLASS_ORDER, S_FEAT, KS, NC
    d = DATASETS[name]
    DATASET = name; COPIES = d["copies"]; CLASSES = d["classes"]; CLASS_ORDER = d["order"]
    S_FEAT = d["s_feat"]; KS = d["ks"]; NC = len(CLASS_ORDER)
    set_copy(copy or next(iter(COPIES)))
CLASSES = {1: "NormalEpi", 2: "NormalStroma", 3: "CancerEpi", 4: "CAS"}
CLASS_ORDER = (3, 4, 2, 1)          # companion 4-class mapping: CancerEpi 0, CAS 1, NormalStroma 2, NormalEpi 3
N_PER_CORE_CLASS = {"train": 150, "val": 100, "test": 100}
S_FEAT = 942
KS = (2, 4, 8, 16, 23, 32, 64, 128, 256, 942)
NC = len(CLASS_ORDER)
SEED = 1234


def load_split_ids() -> dict:
    return json.load(open(SPLITS))


# ----------------------------------------------------------------------------- cache
def build_cache() -> None:
    if DATASET != "breast":
        raise SystemExit("build hyperspectral caches with code/experiments/hsi_data.py")
    splits = load_split_ids()
    rng = np.random.default_rng(SEED)
    P, Y, L9, C, SD, RC = [], [], [], [], [], []
    t0 = time.time()
    for side in ("train", "val", "test"):
        for cid in splits[side]:
            f = DATA_DIR / f"{cid}.npz"
            if not f.exists():
                print(f"[cache] missing {f.name}", flush=True)
                continue
            z = np.load(f)
            y = z["y"].astype(np.uint8)
            mask = z["tissue_mask"].astype(bool)
            H, W = y.shape
            # interior pixels whose 8 neighbours are all tissue
            ok = np.ones_like(mask)
            ok[[0, -1], :] = False
            ok[:, [0, -1]] = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    ok[1:-1, 1:-1] &= mask[1 + dr:H - 1 + dr, 1 + dc:W - 1 + dc]
            feats = None
            n_core = 0
            for lab in CLASSES:
                idx = np.argwhere((y == lab) & ok)
                if len(idx) == 0:
                    continue
                take = min(len(idx), N_PER_CORE_CLASS[side])
                sel = idx[rng.choice(len(idx), take, replace=False)]
                if feats is None:
                    feats = np.concatenate([z["X_raw"], z["X_d1"], z["X_d2"]], axis=-1).astype(np.float32)  # (H, W, 942)
                patches = np.empty((take, 9, S_FEAT), dtype=np.float32)
                labs9 = np.empty((take, 9), dtype=np.uint8)
                j = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        patches[:, j] = feats[sel[:, 0] + dr, sel[:, 1] + dc]
                        labs9[:, j] = y[sel[:, 0] + dr, sel[:, 1] + dc]
                        j += 1
                P.append(patches); Y.append(np.full(take, lab, dtype=np.uint8)); L9.append(labs9)
                C.append(np.full(take, cid)); SD.append(np.full(take, side)); RC.append(sel.astype(np.int16))
                n_core += take
            print(f"[cache] {side} {cid}: {n_core} patches ({time.time() - t0:.0f}s)", flush=True)
    P = np.concatenate(P); Y = np.concatenate(Y); L9 = np.concatenate(L9)
    C = np.concatenate(C); SD = np.concatenate(SD); RC = np.concatenate(RC)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, P=P, y=Y, y9=L9, core=C, side=SD, rc=RC)
    for s in ("train", "val", "test"):
        m = SD == s
        counts = {CLASSES[c]: int((Y[m] == c).sum()) for c in CLASSES}
        print(f"[cache] {s}: {m.sum()} patches, {counts}, cores {len(np.unique(C[m]))}")
    print(f"[cache] wrote {CACHE} ({P.nbytes / 1e9:.2f} GB)")


def load_cache():
    z = np.load(CACHE, allow_pickle=True)
    return z["P"], z["y"], z["y9"], z["core"], z["side"], z["rc"]


def neighbourhood_stats(y: np.ndarray, y9: np.ndarray, side: np.ndarray) -> dict:
    """How homogeneous natural 3x3 neighbourhoods are (labels of the 8 neighbours vs centre)."""
    out = {}
    nb = np.delete(y9, 4, axis=1)                       # (n, 8) neighbour labels
    for s in ("train", "val", "test"):
        m = side == s
        same = (nb[m] == y[m][:, None])
        other_labelled = (nb[m] != y[m][:, None]) & (nb[m] != 0)
        out[s] = {
            "n": int(m.sum()),
            "frac_all8_same_label": float(same.all(1).mean()),
            "frac_any_other_labelled_class": float(other_labelled.any(1).mean()),
            "mean_frac_same_label": float(same.mean()),
            "mean_frac_unlabelled": float((nb[m] == 0).mean()),
        }
    return out


# ----------------------------------------------------------------------------- pixel sweep
def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels) -> tuple[float, list[float]]:
    from sklearn.metrics import f1_score
    per = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return float(per.mean()), [float(v) for v in per]


def _fit_mlp(Ztr, ytr, Zev: dict, device, seed: int = 0, hidden: int = 256, epochs: int = 40, lr: float = 1e-3):
    """Small ReLU MLP, class-balanced CE, Adam, fixed schedule (no selection on evaluation data)."""
    import torch
    torch.manual_seed(seed)
    Xt = torch.from_numpy(Ztr).float().to(device); yt = torch.from_numpy(ytr).long().to(device)
    n_cls = int(yt.max()) + 1
    counts = torch.bincount(yt, minlength=n_cls).float()
    w = (counts.sum() / (n_cls * counts.clamp(min=1))).to(device)
    net = torch.nn.Sequential(torch.nn.Linear(Ztr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, n_cls)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = 512
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g).to(device)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(net(Xt[b]), yt[b], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    preds = {}
    with torch.no_grad():
        for k, Z in Zev.items():
            preds[k] = net(torch.from_numpy(Z).float().to(device)).argmax(1).cpu().numpy()
    return preds


def pixel_sweep(device_str: str = "cuda", seeds: int = 3) -> None:
    import pandas as pd
    import torch
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    P, y, y9, core, side = load_cache()[:5]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = neighbourhood_stats(y, y9, side)
    (OUT_DIR / "neighbourhood_stats.json").write_text(json.dumps(stats, indent=2))
    print("[sweep] neighbourhood stats:", json.dumps(stats, indent=1), flush=True)

    Xc = P[:, 4, :].astype(np.float64)                  # centre spectra
    cls_index = {c: i for i, c in enumerate(CLASS_ORDER)}
    yi = np.array([cls_index[int(v)] for v in y])
    tr, va, te = (side == "train"), (side == "val"), (side == "test")
    mean, std = Xc[tr].mean(0), Xc[tr].std(0) + 1e-8
    Xs = (Xc - mean) / std
    # PCA on training centres
    U, S, Vt = np.linalg.svd(Xs[tr] - Xs[tr].mean(0), full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    V = Vt.T
    print(f"[sweep] train {tr.sum()} val {va.sum()} test {te.sum()}; cumulative EVR at 8/16/23/64: "
          f"{evr[:8].sum():.4f} {evr[:16].sum():.4f} {evr[:23].sum():.4f} {evr[:64].sum():.4f}", flush=True)
    rows = []
    for k in KS:
        Z = Xs @ V[:, :k]
        Ztr, ytr = Z[tr], yi[tr]
        ev = {"val": Z[va], "test": Z[te]}
        # (1) shrinkage LDA, uniform priors
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(NC, 1.0 / NC)).fit(Ztr, ytr)
        # (2) balanced logistic regression
        lr = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0).fit(Ztr, ytr)
        for name, model in (("lda", lda), ("logreg", lr)):
            for s, Zs in ev.items():
                yt_ = yi[va] if s == "val" else yi[te]
                pred = model.predict(Zs)
                mf1, per = _macro_f1(yt_, pred, range(NC))
                rows.append(dict(k=k, model=name, seed=0, split=s, acc=float((pred == yt_).mean()), macro_f1=mf1,
                                 **{f"f1_{CLASSES[c]}": per[i] for i, c in enumerate(CLASS_ORDER)}))
        # (3) MLP, several seeds
        for sd in range(seeds):
            preds = _fit_mlp(Ztr.astype(np.float32), ytr, {s: Zs.astype(np.float32) for s, Zs in ev.items()}, device, seed=sd)
            for s, pred in preds.items():
                yt_ = yi[va] if s == "val" else yi[te]
                mf1, per = _macro_f1(yt_, pred, range(NC))
                rows.append(dict(k=k, model="mlp", seed=sd, split=s, acc=float((pred == yt_).mean()), macro_f1=mf1,
                                 **{f"f1_{CLASSES[c]}": per[i] for i, c in enumerate(CLASS_ORDER)}))
        df_k = pd.DataFrame([r for r in rows if r["k"] == k])
        summ = df_k.groupby(["model", "split"])[["acc", "macro_f1"]].mean().round(3)
        print(f"[sweep] k={k}\n{summ.to_string()}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "pixel_sweep.csv", index=False)
    piv = df.groupby(["model", "split", "k"])["macro_f1"].mean().unstack("k").round(3)
    print("[sweep] macro-F1 by number of principal components:\n" + piv.to_string(), flush=True)
    print(f"[sweep] wrote {OUT_DIR / 'pixel_sweep.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--pixel-sweep", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--copy", default=None, help="data copy (breast: nodenoise|pca23; paviau: raw)")
    ap.add_argument("--dataset", default="breast", choices=list(DATASETS))
    a = ap.parse_args(argv)
    set_dataset(a.dataset, a.copy)
    print(f"[exp4] data copy {COPY}: {DATA_DIR} -> {OUT_DIR}", flush=True)
    if a.build_cache:
        build_cache()
    if a.pixel_sweep:
        pixel_sweep(a.device, a.seeds)
    if not (a.build_cache or a.pixel_sweep):
        print(__doc__)


if __name__ == "__main__":
    main()
