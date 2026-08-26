#!/bin/bash
# E3c lane driver: runs configs sequentially, continues on failure.
# Usage: run_lane.sh <lane_name> "<arm>:<width>:<seed>:<optimizer>" ...
LANE=$1; shift
cd /home/u37314kd/Projects/spectral_shortcut_theory
for cfg in "$@"; do
  IFS=: read -r ARM W SEED OPT <<< "$cfg"
  DIR=experiments_shortcut/e3c/breast_f0/${ARM}_h${W}_${OPT}_s${SEED}
  rm -rf "$DIR"
  echo "[$LANE] START $cfg $(date +%H:%M:%S)"
  python code/experiments/exp1_7_train.py --arm "$ARM" --width "$W" --seed "$SEED" \
    --optimizer "$OPT" --epochs 60 \
    > experiments_shortcut/logs/${ARM}_h${W}_${OPT}_s${SEED}.log 2>&1 \
    && echo "[$LANE] DONE  $cfg $(date +%H:%M:%S)" \
    || echo "[$LANE] FAIL  $cfg $(date +%H:%M:%S)"
done
echo "[$LANE] LANE COMPLETE $(date +%H:%M:%S)"
