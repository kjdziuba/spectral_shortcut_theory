"""E3c analysis — every theory-facing number from the 20-run breast matrix.

Reads experiments_shortcut/e3c/breast_f0/*/ and reports, per run:
  - final val macro-F1 (mean of last 5 epochs; robust to end noise)
  - theta/phi relative displacement at T
  - canonical envelope fit: mu-hat = sup{mu : r(t) <= C-hat/(1+mu t) all t},
    C-hat = smoothed r(0). Computed as min_t (C-hat/r_t - 1)/t on the
    rolling-median residual (window 25 steps) -- the sup over admissible mu
    demanded by review finding ChatGPT-7. Time unit = optimizer step.
  - realized gain B-hat = sup_t ||grad_theta||/||r|| (and p99, less
    outlier-fragile); theorem instantiation (B-hat * C-hat / mu-hat)
    * log(C-hat/r_T) vs the measured displacement (needs no flow-time
    conversion: both sides use the same step clock through mu-hat).
  - dissipation share: sum||grad_theta||^2 / sum(||grad_theta||^2+||grad_phi||^2)
    (descriptive under AdamW; nearer theorem-grade for the SGD pair)
  - EGR at epochs {1,5,20,60} (per-epoch median) and containment probe
    jac_op = ||dg_phi/dZ||_op at first/last probe epochs.

Prints arm x width tables and the three pre-registered comparisons:
joint vs frozen-random (Thm-2 functional read), joint vs frozen-PCA
(falsifier), SGD pair. Writes results/e3c_analysis_runs.csv.

v2 (reviewer round) additions, all backwards compatible: runs produced with
the matched-hygiene flags carry extra steps.csv columns and extra config.json
keys. Pre-existing runs have neither, and every new field is read with .get /
an `in` guard, so their rows keep exactly the columns and values they had.
New CSV columns: best_epoch, val_f1_best_saved (config.json, --save_best runs
only), clip_coef, upd_theta_norm, upd_phi_norm (per-run means over steps.csv).

Verifier-round fixes: '_speclr<M>' is appended only for the arms whose theta
is actually in the optimizer (a stray multiplier on a frozen arm used to
create a phantom arm identical to the plain one), and the spectral-lr
sensitivity block now covers SGD runs too instead of silently dropping them.
Neither changes any output for the pre-existing runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E3C = ROOT / "experiments_shortcut" / "e3c" / "breast_f0"

# Mirrors exp1_7_train.TRAINABLE_THETA_ARMS: the only arms whose theta is in
# the optimizer, hence the only arms where --spectral_lr_mult does anything.
TRAINABLE_THETA_ARMS = ("joint_linear", "joint_mlp", "finetune_real")


def envelope_mu(r: np.ndarray, warmup: int = 10) -> tuple[float, float]:
    """Canonical sup-estimator for the residual envelope.

    r: rolling-median-smoothed residual per step. Returns (mu_hat, C_hat).
    mu_hat = min over t>=warmup of (C_hat/r_t - 1)/t  (pointwise constraint
    C_hat/(1+mu t) >= r_t solved for mu, sup over admissible = min over t).
    """
    C = float(r[:warmup].max())
    t = np.arange(len(r), dtype=float)
    mask = (t >= warmup) & (r > 1e-8)
    if not mask.any():
        return float("nan"), C
    mus = (C / r[mask] - 1.0) / t[mask]
    return float(max(mus.min(), 0.0)), C


def hygiene_suffix(cfg: dict) -> str:
    """'' for a default-hygiene run, else a tag describing the deviation.

    A matched-hygiene run trains DIFFERENT parameters under a DIFFERENT clip
    from the plain arm of the same name, so it must not be averaged into it.
    Every pre-existing run lacks these keys and gets '' -> label unchanged.
      train_all + phi_only  -> '_m'      (the lane's <arm>_m token)
      anything else         -> explicit '_bnT'/'_bnF'/'_clipphi'/'_noclip'
      cosine schedule       -> '_cosine' appended
    """
    bn = cfg.get("bn_affine_mode", "arm_default")
    clip = cfg.get("clip_scope", "optimizer")
    sched = cfg.get("lr_schedule", "none")
    if (bn, clip, sched) == ("arm_default", "optimizer", "none"):
        return ""
    parts = []
    if (bn, clip) == ("train_all", "phi_only"):
        parts.append("m")
    else:
        if bn != "arm_default":
            parts.append({"train_all": "bnT", "freeze_all": "bnF"}[bn])
        if clip != "optimizer":
            parts.append({"phi_only": "clipphi", "none": "noclip"}[clip])
    if sched != "none":
        parts.append(sched)
    return "_" + "_".join(parts) if parts else ""


def arm_label(cfg: dict) -> str:
    """cfg['arm'], suffixed '_speclr<M>' when the run recorded a non-unit
    spectral lr multiplier (E3c-local speclr arms), then a hygiene tag for a
    matched-protocol run. Runs without either key (every pre-existing run)
    keep the bare arm name, so grouping is unchanged.

    Verifier fix: the suffix is applied ONLY for the trainable-theta arms.
    The runner warns that the multiplier is a no-op when theta is not in the
    optimizer, so a frozen-arm run launched with a stray --spectral_lr_mult
    used to surface here as a distinct phantom arm that was behaviourally
    identical to the plain one; it is now folded into the plain arm.
    """
    arm = cfg["arm"]
    mult = cfg.get("spectral_lr_mult")
    if mult is not None and float(mult) != 1.0:
        if arm in TRAINABLE_THETA_ARMS:
            arm = f"{arm}_speclr{float(mult):g}"
        else:
            print(f"[note] {arm}: spectral_lr_mult={mult} is a runner no-op "
                  f"for a frozen arm; folded into '{arm}'", file=sys.stderr)
    return arm + hygiene_suffix(cfg)


def col_mean(steps: pd.DataFrame, name: str) -> float:
    """Mean of a steps.csv column, NaN when the column predates v2."""
    if name not in steps.columns:
        return float("nan")
    return float(pd.to_numeric(steps[name], errors="coerce").mean())


def analyze_run(d: Path, min_epochs: int = 60) -> dict | None:
    try:
        cfg = json.loads((d / "config.json").read_text())
        steps = pd.read_csv(d / "steps.csv").drop_duplicates(
            subset=["step"], keep="last")
        epochs = pd.read_csv(d / "epochs.csv").drop_duplicates(
            subset=["epoch"], keep="last").sort_values("epoch")
    except Exception as e:
        print(f"[skip] {d.name}: {e}", file=sys.stderr)
        return None
    if epochs.epoch.max() < min_epochs:
        print(f"[skip] {d.name}: incomplete ({epochs.epoch.max()} epochs)",
              file=sys.stderr)
        return None

    r_smooth = steps.r_rms.rolling(25, min_periods=5, center=True).median()
    r_smooth = r_smooth.bfill().ffill().to_numpy()
    mu, C = envelope_mu(r_smooth)

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = (steps.grad_theta_norm / steps.r_rms).replace(
            [np.inf, -np.inf], np.nan).dropna()
    Bhat, Bp99 = float(gain.max()), float(gain.quantile(0.99))

    th2 = steps.grad_theta_norm.to_numpy() ** 2
    ph2 = steps.grad_phi_norm.to_numpy() ** 2
    diss_share = float(th2.sum() / max(th2.sum() + ph2.sum(), 1e-30))

    r_T = float(r_smooth[-25:].mean())
    bound = (Bp99 * C / mu) * np.log(max(C / max(r_T, 1e-8), 1.0)) \
        if mu > 0 else float("nan")

    # theta_disp in epochs.csv is absolute; recover relative via config norm
    # if stored, else use the runner's printed relative (theta0_norm in cfg).
    th0 = cfg.get("theta0_norm")
    last = epochs.iloc[-1]
    th_rel = (last.theta_disp / th0) if th0 else float("nan")
    th_abs = float(last.theta_disp)

    egr_ep = steps.groupby("epoch").egr.median()
    jac = epochs.dropna(subset=["input_jac_op"]) \
        if "input_jac_op" in epochs else pd.DataFrame()

    f1_final = float(epochs.val_macro_f1.tail(5).mean())
    return dict(
        arm=arm_label(cfg), width=cfg["width"], seed=cfg["seed"],
        optimizer=cfg.get("optimizer", "adamw"),
        val_f1=f1_final, val_f1_best=float(epochs.val_macro_f1.max()),
        theta_disp_abs=th_abs, theta_disp_rel=float(th_rel),
        mu_hat=mu, C_hat=C, r_final=r_T,
        B_hat=Bhat, B_p99=Bp99, thm2_bound=bound,
        diss_share_theta=diss_share,
        egr_e1=float(egr_ep.get(1, np.nan)), egr_e5=float(egr_ep.get(5, np.nan)),
        egr_e20=float(egr_ep.get(20, np.nan)), egr_e60=float(egr_ep.get(60, np.nan)),
        jac_first=float(jac.input_jac_op.iloc[0]) if len(jac) else np.nan,
        jac_last=float(jac.input_jac_op.iloc[-1]) if len(jac) else np.nan,
        # ---- v2 columns; NaN for every pre-existing run ----------------
        best_epoch=cfg.get("best_epoch", np.nan),
        val_f1_best_saved=cfg.get("best_val_f1", np.nan),
        clip_coef=col_mean(steps, "clip_coef"),
        upd_theta_norm=col_mean(steps, "upd_theta_norm"),
        upd_phi_norm=col_mean(steps, "upd_phi_norm"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3c_dir", default=str(E3C),
                    help="run tree to analyse (default: the breast_f0 matrix)")
    ap.add_argument("--out", default=str(ROOT / "results" / "e3c_analysis_runs.csv"),
                    help="output CSV (default: results/e3c_analysis_runs.csv)")
    ap.add_argument("--min_epochs", type=int, default=60,
                    help="skip runs shorter than this (default 60)")
    args = ap.parse_args()

    rows = [r for d in sorted(Path(args.e3c_dir).iterdir()) if d.is_dir()
            if (r := analyze_run(d, args.min_epochs)) is not None]
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[write] {out}  ({len(df)} runs)\n")

    adamw = df[df.optimizer == "adamw"]
    g = adamw.groupby(["arm", "width"]).agg(
        f1=("val_f1", "mean"), f1_sd=("val_f1", "std"),
        th_rel=("theta_disp_rel", "mean"),
        mu=("mu_hat", "mean"), B99=("B_p99", "mean"),
        bound=("thm2_bound", "mean"), disp=("theta_disp_abs", "mean"),
        diss=("diss_share_theta", "mean"),
        egr1=("egr_e1", "mean"), egr60=("egr_e60", "mean"),
        jacL=("jac_last", "mean"),
    ).reset_index()
    print("=== arm x width (AdamW, mean over seeds) ===")
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n=== pre-registered comparisons (final val F1, mean+/-sd) ===")
    for w in sorted(adamw.width.unique()):
        sub = adamw[adamw.width == w]
        line = [f"h={w}:"]
        for arm in ("joint_linear", "frozen_random", "frozen_pca",
                    "frozen_pretrained", "finetune_real"):
            s = sub[sub.arm == arm].val_f1
            if len(s):
                line.append(f"{arm}={s.mean():.4f}+/-{s.std():.4f}")
        print("  " + "  ".join(line))
        jj = sub[sub.arm == "joint_linear"].val_f1
        fr = sub[sub.arm == "frozen_random"].val_f1
        fp = sub[sub.arm == "frozen_pca"].val_f1
        jm = sub[sub.arm == "joint_mlp"].val_f1
        fpt = sub[sub.arm == "frozen_pretrained"].val_f1
        ft = sub[sub.arm == "finetune_real"].val_f1
        if len(jj) and len(fr):
            print(f"      joint - frozen_random = {jj.mean() - fr.mean():+.4f}"
                  f"   (Thm-2 functional read: ~0 predicted)")
        if len(jj) and len(fp):
            print(f"      joint - frozen_pca    = {jj.mean() - fp.mean():+.4f}"
                  f"   (falsifier: strongly positive would refute)")
        # E3c-local arms (pretrained per-pixel MLP); silent unless present.
        if len(jm) and len(fpt):
            print(f"      joint_mlp - frozen_pretrained = {jm.mean() - fpt.mean():+.4f}"
                  f"   (does a GOOD frozen encoder match joint training?)")
        if len(ft) and len(fpt):
            print(f"      finetune_real - frozen_pretrained = {ft.mean() - fpt.mean():+.4f}"
                  f"   (does theta flow add anything on top of pretraining?)")
        # Verifier fix: this block used to read `sub` (AdamW only), so an SGD
        # speclr run never appeared in it. Read the full frame at this width
        # and label the optimizer. Prints nothing when no speclr run exists,
        # so pre-existing output is unchanged.
        spec = df[(df.width == w) & df.arm.str.contains("_speclr")]
        if len(spec):
            print("      spectral-lr sensitivity:")
            for (a, o), s in spec.groupby(["arm", "optimizer"]).val_f1:
                print(f"        {a:<28s} [{o:<5s}] val_f1={s.mean():.4f}"
                      f"+/-{s.std():.4f} (n={len(s)})")

    sgd = df[df.optimizer == "sgd"]
    if len(sgd):
        print("\n=== SGD pair (theorem's regime) ===")
        print(sgd[["arm", "width", "seed", "val_f1", "theta_disp_rel",
                   "mu_hat", "B_p99", "thm2_bound", "theta_disp_abs",
                   "diss_share_theta"]].to_string(index=False,
                  float_format=lambda v: f"{v:.4g}"))

    print("\n=== Thm-2 bound vs measured displacement (joint arms) ===")
    trainable = df[df.arm.str.startswith(("joint_linear", "finetune_real"))]
    for _, r in trainable.iterrows():
        ratio = r.thm2_bound / max(r.theta_disp_abs, 1e-30)
        tag = "" if r.arm == "joint_linear" else f" [{r.arm}]"
        print(f"  {r.optimizer} h={r.width} s={r.seed}{tag}: measured ||dtheta||="
              f"{r.theta_disp_abs:.3f}  bound={r.thm2_bound:.3f}  "
              f"bound/measured={ratio:.2f}")


if __name__ == "__main__":
    main()
