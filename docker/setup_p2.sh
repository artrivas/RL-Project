#!/usr/bin/env bash
# Run from Ubuntu/WSL: bash docker/setup_p2.sh [--require-data]
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--require-data" ) ]]; then
    echo "Usage: bash docker/setup_p2.sh [--require-data]" >&2
    exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

command -v docker >/dev/null 2>&1 || {
    echo "Docker CLI is missing in this Ubuntu environment. Install Docker Engine + Compose or enable Docker Desktop WSL integration." >&2
    exit 1
}
docker compose version >/dev/null
docker info >/dev/null || {
    echo "Docker is not reachable. Start its service/Desktop and check permissions; this script does not change system permissions." >&2
    exit 1
}
export P2_UID="$(id -u)"
export P2_GID="$(id -g)"
compose=(docker compose -f "$repo_root/docker/docker-compose.p2.yml")
mkdir -p notebooks/data/m5/raw notebooks/artifacts/p2_via_b/setup notebooks/p2
echo "Disk available on the project filesystem:"
df -h .
"${compose[@]}" config --quiet
"${compose[@]}" build retail
"${compose[@]}" run --rm --no-deps retail python -m pip check
"${compose[@]}" run --rm --no-deps retail python -m unittest discover -s src/retail/tests -v
"${compose[@]}" run --rm --no-deps retail python -m pytest -q -p no:cacheprovider src/tests
doctor_args=(--smoke --report notebooks/artifacts/p2_via_b/setup/environment.json)
if [[ "${1:-}" == "--require-data" ]]; then
    doctor_args+=(--require-data)
fi
"${compose[@]}" run --rm --no-deps retail python -m src.retail.doctor "${doctor_args[@]}"
"${compose[@]}" run --rm -T --no-deps retail python -m pip freeze > notebooks/artifacts/p2_via_b/setup/requirements.resolved.txt
docker image inspect rl-project-p2:local --format '{{.Id}}' > notebooks/artifacts/p2_via_b/setup/image-id.txt
"${compose[@]}" up -d --wait --wait-timeout 120 retail
echo "P2 environment checks passed. Jupyter: http://localhost:${P2_JUPYTER_PORT:-8889}"
echo "Authentication token: docker compose -f docker/docker-compose.p2.yml logs retail"
echo "Data status is recorded in notebooks/artifacts/p2_via_b/setup/environment.json. Missing data is allowed only without --require-data."
