"""Public hyperspectral scenes as a reproducible replication of Experiments 3 and 4
(REFOCUS_PLAN section 9). Builds caches in the exact formats the Exp 3 / Exp 4 code reads.

Scenes (from /mnt/hdd2/u37314kd/data_hyperspectral, .mat files as distributed by the EHU
"Hyperspectral Remote Sensing Scenes" page; fetched from the HybridSN GitHub mirror):
  paviau   ROSIS Pavia University, 610 x 340 x 103, 9 classes
  indian   AVIRIS Indian Pines (corrected), 145 x 145 x 200, 16 classes

Spatial "patient" analogue: the scene is cut into square TILES of side `tile`; tiles are
assigned to train / val / test at random (seed) so that every kept class has at least
`min_px` labelled pixels in each split; a patch is admitted only if its whole 3 x 3
neighbourhood lies inside one tile, so no evaluation neighbourhood overlaps a training one.
Residual spatial autocorrelation across tile borders is a known limitation of single-scene
benchmarks and is stated as such.

Preprocessing: reflectance counts cast to float; per-feature standardization and PCA are
fitted on the training split by the experiment code (as for the tissue spectra). No
per-pixel normalization, no denoising, no derivatives.

  --report            print per-class, per-split pixel and tile counts for the chosen split
  --build-pixel-cache write results/exp3_<scene>/cache_fold0.npz  (X, label, core, side)
  --build-patch-cache write results/exp4_<scene>/cache_fold0_natural.npz (P, y, y9, core, side, rc)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/mnt/hdd2/u37314kd/data_hyperspectral")
SCENES = {
    "paviau": dict(cube="PaviaU.mat", gt="PaviaU_gt.mat",
                   classes={1: "Asphalt", 2: "Meadows", 3: "Gravel", 4: "Trees", 5: "PaintedMetal",
                            6: "BareSoil", 7: "Bitumen", 8: "Bricks", 9: "Shadows"}),
    "indian": dict(cube="Indian_pines_corrected.mat", gt="Indian_pines_gt.mat",
                   classes={i: f"c{i}" for i in range(1, 17)}),
}
FRACTIONS = (0.6, 0.2, 0.2)          # tiles to train / val / test


def load_scene(name: str):
    s = SCENES[name]
    cube = sio.loadmat(DATA / s["cube"]); gt = sio.loadmat(DATA / s["gt"])
    X = cube[[k for k in cube if not k.startswith("__")][0]].astype(np.float32)
    y = gt[[k for k in gt if not k.startswith("__")][0]].astype(np.uint8)
    return X, y, s["classes"]


def tile_ids(H: int, W: int, tile: int) -> np.ndarray:
    r = np.arange(H)[:, None] // tile; c = np.arange(W)[None, :] // tile
    return (r * ((W + tile - 1) // tile) + c).astype(np.int32)


def interior_mask(y: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Labelled pixels whose 3x3 neighbourhood lies inside the same tile (and the image)."""
    H, W = y.shape
    ok = np.zeros_like(y, dtype=bool); ok[1:-1, 1:-1] = y[1:-1, 1:-1] > 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            ok[1:-1, 1:-1] &= tiles[1 + dr:H - 1 + dr, 1 + dc:W - 1 + dc] == tiles[1:-1, 1:-1]
    return ok


def tile_split(y: np.ndarray, tiles: np.ndarray, ok: np.ndarray, seed: int, min_px: int, classes: dict,
               fractions=FRACTIONS, tries: int = 2000):
    """Random tile assignment; the first seed offset for which every class has >= min_px admissible
    pixels in every split is used (the classes that never satisfy it are dropped and reported)."""
    n_tiles = int(tiles.max()) + 1
    labelled_tiles = np.unique(tiles[ok])
    keep = dict(classes)
    for attempt in range(tries):
        rng = np.random.default_rng(seed * 100003 + attempt)
        perm = rng.permutation(labelled_tiles)
        n = len(perm); a = int(fractions[0] * n); b = int((fractions[0] + fractions[1]) * n)
        side_of = np.full(n_tiles, "", dtype=object)
        side_of[perm[:a]] = "train"; side_of[perm[a:b]] = "val"; side_of[perm[b:]] = "test"
        counts = {c: {s: int(((y == c) & ok & np.isin(tiles, perm[sl])).sum())
                      for s, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n)))}
                  for c in keep}
        bad = [c for c in keep if min(counts[c].values()) < min_px]
        if not bad:
            return side_of, counts, attempt, []
    # infeasible for some classes: drop the ones that fail most often at the last assignment
    dropped = bad
    for c in dropped:
        keep.pop(c)
    return side_of, {c: counts[c] for c in keep}, attempt, dropped


def report(scene: str, tile: int, seed: int, min_px: int):
    X, y, classes = load_scene(scene)
    tiles = tile_ids(*y.shape, tile); ok = interior_mask(y, tiles)
    side_of, counts, attempt, dropped = tile_split(y, tiles, ok, seed, min_px, classes)
    print(f"[{scene}] tile {tile}px, seed {seed}, attempt {attempt}; dropped classes: {[classes[c] for c in dropped]}")
    n_tiles = {s: int((side_of == s).sum()) for s in ("train", "val", "test")}
    print(f"  tiles: {n_tiles}; admissible labelled pixels {int(ok.sum())} of {int((y > 0).sum())}")
    for c, d in counts.items():
        print(f"  {classes[c]:14s} " + "  ".join(f"{s} {d[s]:6d}" for s in ("train", "val", "test")))
    return X, y, classes, tiles, ok, side_of, counts, dropped


def build_caches(scene: str, tile: int, seed: int, min_px: int, per_tile_class: dict, pixel: bool, patch: bool):
    X, y, classes, tiles, ok, side_of, counts, dropped = report(scene, tile, seed, min_px)
    keep = [c for c in classes if c not in dropped]
    rng = np.random.default_rng(1234)
    H, W, S = X.shape
    Xs, ys, cores, sides = [], [], [], []
    P, Y, L9, C, SD, RC = [], [], [], [], [], []
    for t in np.unique(tiles[ok]):
        side = side_of[t]
        if side == "":
            continue
        for c in keep:
            idx = np.argwhere((y == c) & ok & (tiles == t))
            if len(idx) == 0:
                continue
            take = min(len(idx), per_tile_class[side])
            sel = idx[rng.choice(len(idx), take, replace=False)]
            core = f"t{t:04d}"
            if pixel:
                Xs.append(X[sel[:, 0], sel[:, 1]]); ys.append(np.full(take, c, dtype=np.int16))
                cores.append(np.full(take, core)); sides.append(np.full(take, side))
            if patch:
                pt = np.empty((take, 9, S), dtype=np.float32); l9 = np.empty((take, 9), dtype=np.uint8); j = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        pt[:, j] = X[sel[:, 0] + dr, sel[:, 1] + dc]; l9[:, j] = y[sel[:, 0] + dr, sel[:, 1] + dc]; j += 1
                P.append(pt); Y.append(np.full(take, c, dtype=np.uint8)); L9.append(l9)
                C.append(np.full(take, core)); SD.append(np.full(take, side)); RC.append(sel.astype(np.int16))
    meta = dict(scene=scene, tile=tile, seed=seed, min_px=min_px, per_tile_class=per_tile_class,
                classes={int(k): v for k, v in classes.items()}, dropped=[int(c) for c in dropped],
                counts={int(k): v for k, v in counts.items()}, S=int(S))
    if pixel:
        out = REPO / "results" / f"exp3_{scene}"; out.mkdir(parents=True, exist_ok=True)
        Xa = np.concatenate(Xs); la = np.concatenate(ys); ca = np.concatenate(cores); sa = np.concatenate(sides)
        np.savez_compressed(out / "cache_fold0.npz", X=Xa, label=la, core=ca, side=sa)
        (out / "split_meta.json").write_text(json.dumps(meta, indent=2))
        for s in ("train", "val", "test"):
            m = sa == s
            print(f"[pixel cache] {s}: {m.sum()} px, tiles {len(np.unique(ca[m]))}, "
                  f"{ {classes[c]: int((la[m] == c).sum()) for c in keep} }")
        print(f"[pixel cache] wrote {out / 'cache_fold0.npz'}")
    if patch:
        out = REPO / "results" / f"exp4_{scene}"; out.mkdir(parents=True, exist_ok=True)
        Pa = np.concatenate(P); Ya = np.concatenate(Y); L9a = np.concatenate(L9)
        Ca = np.concatenate(C); SDa = np.concatenate(SD); RCa = np.concatenate(RC)
        np.savez(out / "cache_fold0_natural.npz", P=Pa, y=Ya, y9=L9a, core=Ca, side=SDa, rc=RCa)
        (out / "split_meta.json").write_text(json.dumps(meta, indent=2))
        for s in ("train", "val", "test"):
            m = SDa == s
            print(f"[patch cache] {s}: {m.sum()} patches, tiles {len(np.unique(Ca[m]))}")
        print(f"[patch cache] wrote {out / 'cache_fold0_natural.npz'} ({Pa.nbytes / 1e9:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="paviau", choices=list(SCENES))
    ap.add_argument("--tile", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-px", type=int, default=60)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--build-pixel-cache", action="store_true")
    ap.add_argument("--build-patch-cache", action="store_true")
    ap.add_argument("--per-tile-class", default="train:400,val:200,test:200")
    a = ap.parse_args()
    ptc = {k: int(v) for k, v in (kv.split(":") for kv in a.per_tile_class.split(","))}
    if a.report and not (a.build_pixel_cache or a.build_patch_cache):
        report(a.scene, a.tile, a.seed, a.min_px)
    if a.build_pixel_cache or a.build_patch_cache:
        build_caches(a.scene, a.tile, a.seed, a.min_px, ptc, a.build_pixel_cache, a.build_patch_cache)


if __name__ == "__main__":
    main()
