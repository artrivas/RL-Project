#!/usr/bin/env bash
set -euo pipefail
cd /opt/retail
exec 9>/salidas/cierre.lock
flock -n 9 || { echo "La campaÃ±a ya estÃ¡ en ejecuciÃ³n"; exit 1; }
# Compilar el manuscrito revisado de inmediato para que pueda leerse durante el entrenamiento.
mkdir -p /salidas/lectura
(cd /revision && pdflatex -halt-on-error -interaction=batchmode -output-directory=/salidas/lectura informe_p2_revision.tex && pdflatex -halt-on-error -interaction=batchmode -output-directory=/salidas/lectura informe_p2_revision.tex)
echo "Informe compilado: salidas_cierre/lectura/informe_p2_revision.pdf"
if [[ ! -f /salidas/calibracion.json ]]; then
  python /opt/entrada/calibrar_m5.py --descargar --datos /datos --salida /salidas/calibracion.json
fi
python -m unittest discover -s src/retail/tests -v
python -m unittest discover -s scripts -p test_finalize_m5.py -v
python /revision/codigo/campana_docente.py

