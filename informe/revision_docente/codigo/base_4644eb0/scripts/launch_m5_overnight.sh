#!/usr/bin/env bash
# Run on the Pod from any directory; keep the checkout unchanged while training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
batch="${1:-/workspace/rl-project-validation/m5-night-01}"
if [[ -e "$batch" || -e "$batch.log" || -e "$batch.pid" ]]; then
  echo "Batch or launch record exists: inspect it before starting another." >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$batch")"
python3 -B scripts/test_m5_overnight.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Exclusive log creation also prevents two concurrent launchers duplicating work.
set -o noclobber
nohup python3 -u scripts/m5_overnight.py --output "$batch" --max-hours 8 --execute \
  > "$batch.log" 2>&1 < /dev/null &
batch_pid=$!
printf '%s\n' "$batch_pid" > "$batch.pid"
sleep 5
if kill -0 "$batch_pid" 2>/dev/null; then
  echo "BATCH RUNNING: PID=$batch_pid"
  echo "You may close SSH. Log: $batch.log"
  tail -n 15 "$batch.log"
else
  echo "Batch exited during startup; inspect this log:" >&2
  cat "$batch.log"
  exit 1
fi
