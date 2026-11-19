#!/usr/bin/env bash
set -euo pipefail
cd /opt/retail
mkdir -p /salidas
dest=$(mktemp -d /salidas/m5-XXXXXXXX)
echo "Resultados en $dest"
python /opt/entrada/calibrar_m5.py --descargar --datos /datos --salida "$dest/calibracion.json"
bash scripts/reproduce_m5.sh /datos "$dest/campana" cpu
python /opt/entrada/graficar_g0.py "$dest/campana/suite" "$dest"

