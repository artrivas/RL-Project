# P2 Vía B: cierre experimental

Se evaluaron 24 políticas congeladas y H0 en el último año (d1577-d1941).
Tres semillas de desarrollo: 3, 4, 5; 109500 interacciones reales por política.
No se ejecutó entrenamiento adicional. Resultados descriptivos; no prueban significancia ni convergencia.

La familia con mayor retorno medio de validación fue P0.
Su retorno medio de test es 96590.33, frente a H0=98915.49; diferencia=-2325.15.
Las comparaciones completas están en results_table.csv y paired_ablations.csv.
0/24 corridas satisfacen conjuntamente los dos umbrales de stockouts y el beneficio frente a los proxies.
Esto no certifica superar beneficio o quiebres históricos reales de Walmart.

## Archivos principales

- informe_p2.tex y informe_p2.pdf: artículo IEEE, máximo 8 páginas.
- validation_returns.png, training_returns.png, losses.png: curvas y dispersión.
- test_results.png y results_table.csv: comparación final.
- learning_diagnostics.csv: eficiencia al umbral exploratorio y estabilidad.
- model_test_errors.csv: error de predicción de demanda en trayectorias de test, sin ajustes.
- daily_store.csv: trazas de los tres agentes.
- demo.html: replay interactivo de PPO semilla 3, elección fijada antes de test.
- analisis_p2.ipynb: cuaderno de lectura reproducible del paquete.
- protocol.json y SHA256SUMS.json: datos, fuentes, checkpoints y hashes.
- delivery_checklist.json: requisitos satisfechos, límites y pendientes.

## Alcance y limitaciones

La elasticidad de 1.5 es supuesta, no estimada. Las ventas son un proxy de demanda.
Un solo SKU y un único año no permiten generalizar a Walmart.
No hay observación de inventarios, pedidos históricos ni costes de compra.
El plan inicial reservaba semillas 0,1,2 para confirmación; se documenta su omisión por cierre de alcance.
El umbral de eficiencia es descriptivo, definido después del desarrollo y antes de test.
Las curvas de entrenamiento suman recompensas sin descuento; el G0 descontado sólo se calcula en evaluación.
La cola de entrenamiento mezcla ventanas temporales; la cola de validación usa la misma ventana.
0 corridas cambian el retorno de validación más de 0.01 al pasar a CPU; consultar cpu_validation_check.csv.
El contraejemplo de hacking es sintético; no se atribuye automáticamente a la política N0 en M5.
No hay evidencia de aprobación docente de las aproximaciones históricas.

## Revisión antes de entregar

Revisar visualmente el PDF y confirmar autores heredados de P1.
Leer las conclusiones junto con target_checks.csv; terminar el proceso no equivale a cumplir el objetivo cuantitativo.
La sustentación oral sigue pendiente. No se ha enviado nada a Canvas ni integrado la rama a main.
