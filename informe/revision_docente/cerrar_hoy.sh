#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export M5_RAW_DIR="${M5_RAW_DIR:-$PWD/../../notebooks/data/m5/raw}"
if [[ ! -d "$M5_RAW_DIR" ]]; then export M5_RAW_DIR="$PWD/salidas_cierre/datos"; fi
mkdir -p "$M5_RAW_DIR" salidas_cierre
export M5_WORKERS="${M5_WORKERS:-3}"
docker info --format 'Docker Server {{.ServerVersion}}'
docker compose -f compose.cierre.yaml up --build --abort-on-container-exit --exit-code-from cierre

