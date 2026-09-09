#!/bin/bash
# E3c lane driver: runs configs sequentially, continues on failure.
# Usage: run_lane.sh <lane_name> "<arm_token>:<width>:<seed>:<optimizer>" ...
#
# Arm tokens (the token names the output dir + log, so every token is unique):
#   joint_linear | frozen_random | frozen_pca | joint_mlp   -- unchanged
#   frozen_pretrained | finetune_real                        -- E3c-local (pretrained MLP)
#   joint_speclr<M>    -> --arm joint_linear  --spectral_lr_mult <M>   e.g. joint_speclr0.1, joint_speclr10
#   finetune_speclr<M> -> --arm finetune_real --spectral_lr_mult <M>   e.g. finetune_speclr0.1
#   mlp_speclr<M>      -> --arm joint_mlp     --spectral_lr_mult <M>
# speclr tokens pass --run_label <token> so the runner writes
#   experiments_shortcut/e3c/breast_f0/<token>_h<W>_<opt>_s<seed>  (e.g. joint_speclr0.1_h192_adamw_s0)
# Plain tokens pass no extra flags: their command line is byte-identical to before.
#
# MATCHED-HYGIENE TOKENS (v2, reviewer round).  Suffix '_m' on any arm token:
#   <arm>_m  -> --arm <arm> --bn_affine_mode train_all --clip_scope phi_only \
#               --save_best --run_label <arm>_m
#   e.g. joint_linear_m, frozen_random_m, frozen_pca_m, joint_mlp_m,
#        frozen_pretrained_m, finetune_real_m
#   joint_cos_m -> joint_linear + the _m flags + --lr_schedule cosine
# The token (not the arm) names the output dir, so joint_linear_m_h48_adamw_s0
# never collides with joint_linear_h48_adamw_s0.
#
# Env overrides (every default reproduces the pre-v2 command line exactly):
#   EPOCHS=<n>      epochs per run                 (default 60)
#   OUT_ROOT=<dir>  --out_root for the runner      (default: unset -> the
#                   runner's own default experiments_shortcut/e3c; the flag is
#                   then not passed at all, so existing tokens are unchanged)
#   LOG_DIR=<dir>   where the per-run .log goes    (default experiments_shortcut/logs)
LANE=$1; shift
cd /home/u37314kd/Projects/spectral_shortcut_theory
EPOCHS=${EPOCHS:-60}
OUT_ROOT=${OUT_ROOT:-}
LOG_DIR=${LOG_DIR:-experiments_shortcut/logs}
mkdir -p "$LOG_DIR"
ROOT_ARGS=(); BASE=experiments_shortcut/e3c
if [[ -n "$OUT_ROOT" ]]; then ROOT_ARGS=(--out_root "$OUT_ROOT"); BASE=$OUT_ROOT; fi
MATCHED=(--bn_affine_mode train_all --clip_scope phi_only --save_best)
for cfg in "$@"; do
  IFS=: read -r TOKEN W SEED OPT <<< "$cfg"
  EXTRA=()
  case "$TOKEN" in
    # joint_cos_m must precede the generic *_m branch (it also ends in _m).
    joint_cos_m)      ARM=joint_linear;   EXTRA=("${MATCHED[@]}" --lr_schedule cosine --run_label "$TOKEN") ;;
    joint_speclr*_m)  ARM=joint_linear;   MULT=${TOKEN#joint_speclr}; MULT=${MULT%_m}
                      EXTRA=("${MATCHED[@]}" --spectral_lr_mult "$MULT" --run_label "$TOKEN") ;;
    finetune_speclr*_m) ARM=finetune_real; MULT=${TOKEN#finetune_speclr}; MULT=${MULT%_m}
                      EXTRA=("${MATCHED[@]}" --spectral_lr_mult "$MULT" --run_label "$TOKEN") ;;
    *_m)              ARM=${TOKEN%_m};    EXTRA=("${MATCHED[@]}" --run_label "$TOKEN") ;;
    joint_speclr*)    ARM=joint_linear;  EXTRA=(--spectral_lr_mult "${TOKEN#joint_speclr}"    --run_label "$TOKEN") ;;
    finetune_speclr*) ARM=finetune_real; EXTRA=(--spectral_lr_mult "${TOKEN#finetune_speclr}" --run_label "$TOKEN") ;;
    mlp_speclr*)      ARM=joint_mlp;     EXTRA=(--spectral_lr_mult "${TOKEN#mlp_speclr}"      --run_label "$TOKEN") ;;
    *)                ARM=$TOKEN ;;
  esac
  DIR=$BASE/breast_f0/${TOKEN}_h${W}_${OPT}_s${SEED}
  # Guard (2026-09-09): never delete a finished run unless FORCE=1.
  if [[ -s "$DIR/epochs.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[$LANE] SKIP  $cfg (run dir exists with epochs.csv; set FORCE=1 to overwrite) $(date +%H:%M:%S)"; continue
  fi
  rm -rf "$DIR"
  echo "[$LANE] START $cfg $(date +%H:%M:%S)"
  python code/experiments/exp1_7_train.py --arm "$ARM" --width "$W" --seed "$SEED" \
    --optimizer "$OPT" --epochs "$EPOCHS" "${ROOT_ARGS[@]}" "${EXTRA[@]}" \
    > "$LOG_DIR"/${TOKEN}_h${W}_${OPT}_s${SEED}.log 2>&1 \
    && echo "[$LANE] DONE  $cfg $(date +%H:%M:%S)" \
    || echo "[$LANE] FAIL  $cfg $(date +%H:%M:%S)"
done
echo "[$LANE] LANE COMPLETE $(date +%H:%M:%S)"
