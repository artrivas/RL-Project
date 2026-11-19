#!/usr/bin/env bash
# Explicit expensive reproduction; the finalization launcher NEVER calls this.
set -euo pipefail
cd "$(dirname "$0")/.."
raw="${1:?Usage: reproduce_m5.sh RAW_DATA_DIR NEW_OUTPUT_DIR [cpu|cuda]}"
output="${2:?Provide a NEW output directory}"
device="${3:-cpu}"
test ! -e "$output" || { echo "Refusing to overwrite $output"; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -m unittest discover -s src/retail/tests -v
python3 -m src.retail.suite --data-dir "$raw" \
  --config docker/m5_final_assumptions.json --item-id FOODS_3_090 \
  --settings docker/m5_final_settings.json --seeds 3 4 5 --episodes 300 \
  --device "$device" --output "$output/suite" --execute
bash scripts/launch_m5_final.sh "$output/final" "$output" "$raw"
