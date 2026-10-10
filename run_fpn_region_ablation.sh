#!/usr/bin/env bash
set -euo pipefail

# Run from RCLCOD repo root, after install_fpn_region_ablation.py.
# Set MODES="seed adaptive" to skip a completed baseline.
# Each run has isolated checkpoints/logs; all have the same seed, data, sampler.
MODES="${MODES:-baseline seed adaptive}"
for MODE in ${MODES}; do
  echo "===== FPN region experiment: ${MODE} ====="
  python basemain_continuous.py \
    --config "configs/rcl_fpn_region/fpn_${MODE}.py" \
    --model-name PvtV2B4_FPN_RegionAblation \
    --output-dir "realsam2/FPN_region_${MODE}"
done
