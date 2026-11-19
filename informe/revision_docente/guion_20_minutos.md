# Sustentación de 20 minutos

Leonardo Candio y Dimael Rivas

## 00:00–00:40 · Precios e inventario con aprendizaje por refuerzo

Presentamos la campaña de control colaborativo y la revisión de su referencia a partir de ventas M5.

## 00:40–01:45 · Problema y alcance de la evidencia

Fijamos el precio observado e inducimos la reposición con las ventas de entrenamiento. Los niveles de cobertura del 90 y 95 por ciento son supuestos del equipo. Se desconocen los inventarios históricos diarios.

## 01:45–02:45 · Datos y protocolo temporal

Entrenamos en los primeros 1211 días. El último año estaba reservado durante el entrenamiento. La revisión de la referencia y la reproducción tabular reutilizan este año. PPO se seleccionó en validación.

## 02:45–03:50 · Estado, acciones y recompensa

La observación local contiene existencias, tránsito, pronóstico de 28 días, precio, calendario, horizonte restante e inventario regional. No garantiza Markovianidad completa. Hay cinco multiplicadores y tres objetivos de cobertura: 15 acciones. El volumen solicitado se calcula con inventario y pronóstico, por eso puede cambiar aunque el índice de acción no cambie. Las ventas históricas se modulan con elasticidad supuesta 1.5. Los despachos de hoy llegan mañana. La recompensa regional suma ingresos menos almacenamiento y faltantes, sin el bono de shaping en las métricas reportadas.

Fuentes: informe_p2.pdf; protocol.json.

## 03:50–04:55 · Comparación y ablaciones

El baseline tabular reutiliza el algoritmo de P1, con tablas nuevas entrenadas para retail. DDQN representa la familia de valor y PPO la familia Actor-Crítico. Los agentes tienen redes locales. La condición egoísta recibe recompensa local, las demás reciben la regional. V1 aísla PBRS, V2 aísla Dyna y V3 combina ambos. N0 añade un bono ingenuo. Se mantiene el presupuesto de interacciones reales, pero no el número de actualizaciones ni el coste computacional.

Fuentes: informe_p2.pdf; protocol.json.

## 04:55–06:15 · PBRS: invariancia del objetivo descontado

El potencial penaliza el desequilibrio de cobertura y la cobertura menor de tres días. Se usa la diferencia descontada de potencial. Sumando las recompensas descontadas, los términos interiores se cancelan. Con potencial terminal cero queda el retorno original menos el potencial inicial. Dado el mismo estado inicial, esa cantidad no depende de las decisiones futuras, así que no cambia el orden de las políticas. La identidad usa el estado completo y el objetivo descontado. No garantiza que redes y agentes independientes aprendan más rápido, ni que mejore el retorno sin descuento a presupuesto finito.

Fuentes: informe_p2.pdf; protocol.json.
Referencia: Ng, Harada y Russell (1999), ICML, pp. 278–287.

## 06:15–07:10 · Bono ingenuo y cambio de preferencia

Este es un experimento controlado de dos pasos, separado de M5. Servir obtiene mayor retorno base que acaparar. El bono ingenuo recompensa cobertura y hace que acaparar parezca mejor. Con PBRS ambas trayectorias reciben la misma corrección total de más uno porque el potencial inicial es menos uno, y la preferencia original se conserva. No afirmamos que N0 haya explotado este mecanismo en M5: sus resultados empíricos se analizan por separado.

Fuentes: informe_p2.pdf; protocol.json.

## 07:10–08:15 · Dyna híbrido: demanda aprendida

El modelo aprende log de uno más demanda usando observación, acción e identidad de tienda. Sus etiquetas proceden de transiciones reales del simulador en entrenamiento. La demanda posterior a la acción es privilegiada para ajustar el modelo, no se entrega a la política antes de decidir. Inventario y recompensa siguen ecuaciones exactas. Dyna genera transiciones sintéticas de un paso para actualizar DDQN. Esas transiciones no vuelven a entrenar el modelo ni se mezclan con el replay real. Más experiencia sintética puede introducir sesgo y no garantiza más retorno.

Fuentes: informe_p2.pdf; protocol.json.

## 08:15–09:30 · Aprendizaje en validación

El retorno sin descuento coincide en los 900 episodios tabulares reproducidos. Eso permitió recuperar la curva de G₀ para el tabular. Los registros profundos conservan solo el total sin descuento por episodio.

## 09:30–11:00 · Test con siete días de cobertura (H0 al 95 %)

La referencia de esta tabla corresponde al 95 %. Ninguna corrida cumple conjuntamente con ese comparador y el agente egoísta. La diapositiva 14 muestra el escenario del 90 %.

## 11:00–11:55 · Ablaciones: efectos observados en test

Las diferencias están emparejadas por semilla y son descriptivas. DDQN con Dyna pierde en promedio 6591.5 unidades de retorno frente a DDQN. PBRS solo mejora dos de tres semillas pero su diferencia media es negativa. La diferencia de DDQN cooperativo frente al egoísta está dominada por una corrida egoísta muy mala: sólo una de tres diferencias es positiva. No se concluye superioridad estadística ni se atribuye el resultado a una sola causa.

Fuentes: paired_ablations.csv; test_summary.json.

## 11:55–12:50 · Eficiencia y estabilidad

El umbral descriptivo es 72143.5, el 80 por ciento del máximo empírico de validación. Se fijó antes de abrir test, pero después del desarrollo. Las evaluaciones ocurren cada 3650 pasos: un cruce en la primera evaluación no identifica el instante exacto ni diferencia métodos que cruzaron antes. DDQN, PBRS y Dyna tienen la misma mediana de primer cruce. Esta resolución no acredita aceleración adicional. La estabilidad usa los últimos diez checkpoints de validación.

Fuentes: learning_diagnostics.csv; protocol.json.

## 12:50–13:45 · El modelo de demanda fuera de muestra

El MAE medio del modelo es menor que el del pronóstico de 28 días en ambas condiciones. Sin embargo, el RMSE es mayor. La mejora en error absoluto convive con errores grandes más penalizados por RMSE. Esto no demuestra la causa del menor retorno con Dyna. Las métricas se calculan sobre la trayectoria propia de cada política, con demandas dependientes del precio. No son un benchmark idéntico entre V2 y V3. Son predicciones de un paso y no prueban fidelidad de rollouts largos.

Fuentes: model_test_errors.csv.

## 13:45–15:30 · La meta cambia con la reposición inducida

Al 90 % se reponen siete días en dos tiendas y tres en CA_3. Al 95 % se reponen siete en las tres. El segundo escenario reproduce mejor las ventas de CA_3 y es más difícil de superar. Explicar ambas comparaciones.

## 15:30–17:50 · Demostración en vivo del agente tabular

Abrir la demo y ejecutar un día. La política calcula su acción a partir del stock y del pronóstico. Avanzar siete días y cambiar el período. Se usa T0 para demostrar inferencia. PPO continúa siendo la política seleccionada mediante validación.

## 17:50–20:00 · Conclusiones y límites

Dos semillas de PPO alcanzan la meta bajo H0 al 90 %. Con H0 al 95 % no la alcanza ninguna. La sensibilidad de validación con elasticidades asociativas redujo el retorno tabular de 86 419,91 a 83 022,89.
