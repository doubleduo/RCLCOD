#!/usr/bin/env bash
set -euo pipefail
# Pass any shared CLI arguments, e.g. --checkpoint PATH --max-images 20.
output_dir="./SGT/SAM1_Refine_V1"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--output-dir" ]] && ((i+1<${#args[@]})); then
    output_dir="${args[$((i+1))]}"
  fi
done
for abl in A0 A1 A2 A3 A4; do
  python sam1_generate_refine.py --config "configs/${abl}.json" "$@"
done
python compare_ablations.py --output-dir "$output_dir"
