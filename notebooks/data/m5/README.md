# Datos M5 para P2

Colocar en `raw/` los archivos originales, sin renombrarlos:

- `sales_train_evaluation.csv` (d_1…d_1941).
- `sell_prices.csv`.
- `calendar.csv`.

Fuente: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
Repositorio de los organizadores: https://github.com/Mcompetitions/M5-methods

No confundir `sales_train_validation.csv` (1,913 días) con el archivo de evaluación que espera nuestro split. Los CSV y archivos comprimidos dentro de `raw/` se excluyen de Git. No guardar credenciales de Kaggle en el proyecto. La descarga no está automatizada ni se da por completada.

`python -m src.retail.doctor --skip-dependencies --require-data` revisa presencia y esquema de cabeceras. **No audita todavía valores, fechas reales, cobertura de las tres tiendas, duplicados, joins ni filtraciones de información**; esas verificaciones corresponden al siguiente incremento del loader.
