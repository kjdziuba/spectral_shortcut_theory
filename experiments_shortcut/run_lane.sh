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
LANE=$1; shift
cd /home/u37314kd/Projects/spectral_shortcut_theory
for cfg in "$@"; do
  IFS=: read -r TOKEN W SEED OPT <<< "$cfg"
  EXTRA=()
  case "$TOKEN" in
    joint_speclr*)    ARM=joint_linear;  EXTRA=(--spectral_lr_mult "${TOKEN#joint_speclr}"    --run_label "$TOKEN") ;;
    finetune_speclr*) ARM=finetune_real; EXTRA=(--spectral_lr_mult "${TOKEN#finetune_speclr}" --run_label "$TOKEN") ;;
    mlp_speclr*)      ARM=joint_mlp;     EXTRA=(--spectral_lr_mult "${TOKEN#mlp_speclr}"      --run_label "$TOKEN") ;;
    *)                ARM=$TOKEN ;;
  esac
  DIR=experiments_shortcut/e3c/breast_f0/${TOKEN}_h${W}_${OPT}_s${SEED}
  rm -rf "$DIR"
  echo "[$LANE] START $cfg $(date +%H:%M:%S)"
  python code/experiments/exp1_7_train.py --arm "$ARM" --width "$W" --seed "$SEED" \
    --optimizer "$OPT" --epochs 60 "${EXTRA[@]}" \
    > experiments_shortcut/logs/${TOKEN}_h${W}_${OPT}_s${SEED}.log 2>&1 \
    && echo "[$LANE] DONE  $cfg $(date +%H:%M:%S)" \
    || echo "[$LANE] FAIL  $cfg $(date +%H:%M:%S)"
done
echo "[$LANE] LANE COMPLETE $(date +%H:%M:%S)"
