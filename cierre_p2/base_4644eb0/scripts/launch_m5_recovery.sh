#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
batch="${1:-/workspace/rl-project-validation/m5-night-01}"
recovery="${2:-$batch/recovery-01}"
if [[ -e "$recovery" || -e "$recovery.log" || -e "$recovery.pid" ]]; then
  echo "Recovery or launch record exists; inspect it before retrying." >&2
  exit 1
fi
test -d "$batch"
python3 -B scripts/test_resume_m5_overnight.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -o noclobber
nohup python3 -u scripts/resume_m5_overnight.py \
  --batch "$batch" --output "$recovery" --max-hours 4 --execute \
  > "$recovery.log" 2>&1 < /dev/null &
recovery_pid=$!
printf '%s\n' "$recovery_pid" > "$recovery.pid"
sleep 8
if kill -0 "$recovery_pid" 2>/dev/null; then
  echo "RECOVERY RUNNING: PID=$recovery_pid"
  echo "You may close SSH. Log: $recovery.log"
  tail -n 25 "$recovery.log"
else
  echo "Recovery exited during startup. Inspect log:" >&2
  cat "$recovery.log"
  exit 1
fi
