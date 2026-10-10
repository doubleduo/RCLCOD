#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_fpn_pool_ual.sh a1_high_only
#        bash run_fpn_pool_ual.sh all
MODE="${1:-a1_high_only}"
if [[ "$MODE" == "all" ]]; then
  modes=(a0_no_ual a1_high_only a2_both_full a3_middle_low a4_middle_edge)
else
  modes=("$MODE")
fi
for mode in "${modes[@]}"; do
  cfg="configs/fpn_pool_ual/${mode}.py"
  if [[ ! -f "$cfg" ]]; then echo "Missing $cfg; first run install_and_make_configs.py" >&2; exit 1; fi
  echo "===== FPN UAL ${mode} ====="
  python basemain_continuous.py --config "$cfg" \
    --model-name PvtV2B4_FPN_PoolUAL \
    --output-dir "realsam2/FPN_pool_ual_${mode}"
done
