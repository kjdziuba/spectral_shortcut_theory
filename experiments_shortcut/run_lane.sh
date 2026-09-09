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
#   <speclr token>_m -> the speclr mapping PLUS the _m flags
#        e.g. joint_speclr0.1_m, finetune_speclr10_m, mlp_speclr0.1_m
#   joint_cos_m -> joint_linear + the _m flags + --lr_schedule cosine
# The token (not the arm) names the output dir, so joint_linear_m_h48_adamw_s0
# never collides with joint_linear_h48_adamw_s0.
#
# TOKEN VALIDATION (verifier round): the token is fully resolved and checked
# BEFORE anything is deleted. A token that maps to an unknown arm, a speclr
# token with a missing/non-numeric/non-positive multiplier, or a malformed
# <token>:<W>:<seed>:<opt> quadruple is reported as BADTOKEN and skipped with
# the target directory left untouched. Previously `joint_speclr` (no number)
# rm -rf'd the target dir and only then died in argparse.
#
# Env overrides (every default reproduces the pre-v2 command line exactly):
#   EPOCHS=<n>      epochs per run                 (default 60)
#   OUT_ROOT=<dir>  --out_root for the runner      (default: unset -> the
#                   runner's own default experiments_shortcut/e3c; the flag is
#                   then not passed at all, so existing tokens are unchanged)
#   LOG_DIR=<dir>   where the per-run .log goes    (default experiments_shortcut/logs)
#   FORCE=1         overwrite a run dir that already holds an epochs.csv
LANE=$1; shift
cd /home/u37314kd/Projects/spectral_shortcut_theory
EPOCHS=${EPOCHS:-60}
OUT_ROOT=${OUT_ROOT:-}
LOG_DIR=${LOG_DIR:-experiments_shortcut/logs}
mkdir -p "$LOG_DIR"
ROOT_ARGS=(); BASE=experiments_shortcut/e3c
if [[ -n "$OUT_ROOT" ]]; then ROOT_ARGS=(--out_root "$OUT_ROOT"); BASE=$OUT_ROOT; fi
MATCHED=(--bn_affine_mode train_all --clip_scope phi_only --save_best)
KNOWN_ARMS="joint_linear frozen_random frozen_pca joint_mlp frozen_pretrained finetune_real"
for cfg in "$@"; do
  IFS=: read -r TOKEN W SEED OPT <<< "$cfg"
  EXTRA=(); MULT=""; SCHED=""; IS_M=0; BASE_TOK=$TOKEN
  if [[ "$TOKEN" == *_m ]]; then IS_M=1; BASE_TOK=${TOKEN%_m}; fi
  case "$BASE_TOK" in
    # joint_cos exists only as the matched token joint_cos_m; the bare form
    # falls through to the unknown-arm check below rather than silently
    # becoming joint_linear without the matched flags.
    joint_cos)        if (( IS_M )); then ARM=joint_linear; SCHED=cosine
                      else ARM=$BASE_TOK; fi ;;
    joint_speclr*)    ARM=joint_linear;  MULT=${BASE_TOK#joint_speclr} ;;
    finetune_speclr*) ARM=finetune_real; MULT=${BASE_TOK#finetune_speclr} ;;
    mlp_speclr*)      ARM=joint_mlp;     MULT=${BASE_TOK#mlp_speclr} ;;
    *)                ARM=$BASE_TOK ;;
  esac

  # ---- validate BEFORE touching the filesystem ---------------------------
  BAD=""
  if [[ -z "$TOKEN" || -z "$W" || -z "$SEED" || -z "$OPT" ]]; then
    BAD="expected <token>:<width>:<seed>:<optimizer>"
  elif [[ " $KNOWN_ARMS " != *" $ARM "* ]]; then
    BAD="unknown arm '$ARM'"
  elif [[ "$BASE_TOK" == *speclr* ]] && [[ -z "$MULT" ]]; then
    BAD="speclr token carries no multiplier"
  elif [[ -n "$MULT" ]] && { ! [[ "$MULT" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] \
                             || ! awk -v m="$MULT" 'BEGIN{exit !(m+0>0)}'; }; then
    BAD="speclr multiplier '$MULT' is not a finite positive number"
  fi
  if [[ -n "$BAD" ]]; then
    echo "[$LANE] BADTOKEN $cfg ($BAD); nothing deleted, nothing run $(date +%H:%M:%S)"
    continue
  fi

  if (( IS_M )); then EXTRA+=("${MATCHED[@]}"); fi
  if [[ -n "$SCHED" ]]; then EXTRA+=(--lr_schedule "$SCHED"); fi
  if [[ -n "$MULT" ]]; then EXTRA+=(--spectral_lr_mult "$MULT"); fi
  # Plain tokens (TOKEN == ARM) pass no --run_label, exactly as before.
  if [[ "$TOKEN" != "$ARM" ]]; then EXTRA+=(--run_label "$TOKEN"); fi

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
