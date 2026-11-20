#!/usr/bin/env bash
# Execute inside a prepared remote checkout; does not provision a Pod or install packages.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
python -m src.retail.preflight --require-cuda --report notebooks/artifacts/p2_via_b/preflight.json
echo "Numerical tests passed. Run a tiny --demo training job on the GPU before M5."
