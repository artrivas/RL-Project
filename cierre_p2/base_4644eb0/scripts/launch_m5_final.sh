#!/usr/bin/env bash
# Run inside the existing Ubuntu Pod. No training, no Pod lifecycle operations.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-/workspace/rl-project-validation/m5-final-01}"
batch="${2:-/workspace/rl-project-validation/m5-night-01}"
raw="${3:-/workspace/data/m5/raw}"
mkdir -p "$(dirname "$out")"
# Lock spans dependency setup and evaluation; a second launch cannot race.
exec 9>"${out}.launch.lock"
flock -n 9 || { echo "El cierre ya está activo para $out"; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/m5-matplotlib PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
if ! python3 -c 'import torch; assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,6)' >/dev/null 2>&1; then
  python3 -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
fi
if ! python3 -c 'import numpy, matplotlib, pypdf' >/dev/null 2>&1; then
  python3 -m pip install --no-cache-dir 'numpy>=1.26,<3' matplotlib==3.10.7 pypdf==6.1.3
fi
if ! command -v pdflatex >/dev/null 2>&1 || ! kpsewhich IEEEtran.cls >/dev/null 2>&1 || ! kpsewhich spanish.ldf >/dev/null 2>&1 || ! kpsewhich ptmr8t.tfm >/dev/null 2>&1; then
  if [ "$(id -u)" != 0 ]; then
    echo "Falta LaTeX. Instala texlive-latex-recommended texlive-publishers texlive-lang-spanish texlive-fonts-recommended y repite."
    exit 1
  fi
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends texlive-latex-recommended texlive-publishers texlive-lang-spanish texlive-fonts-recommended
fi
python3 -m unittest discover -s src/retail/tests -v
python3 -m unittest discover -s scripts -p 'test_finalize_m5.py' -v
python3 scripts/finalize_m5.py --batch "$batch" --data-dir "$raw" --output "$out"
