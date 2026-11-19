# Sustentación de 20 minutos

Autores: Leonardo Candio y Dimael Rivas, conservados del informe.

## 1. Precios e inventario con aprendizaje por refuerzo (40 segundos)

Presentamos una comparación de controladores para tres tiendas reales de California dentro de un simulador construido con M5. La pregunta es si las redes, la recompensa compartida y la experiencia sintética mejoran el control frente a una referencia de reposición. El estudio está terminado con resultados favorables y desfavorables, sin reentrenamiento después de abrir test.

## 2. Problema y alcance de la evidencia (70 segundos)

Cada tienda decide precio y reposición, pero las tres dependen de un centro de distribución con capacidad finita. Utilizamos ventas, precios y calendario reales. Los inventarios iniciales, capacidades, costes y elasticidades pertenecen al simulador. No observamos la política de inventario de Walmart. Por ello H0 es una reconstrucción con precio histórico y base-stock. La meta literal de superar quiebres y beneficio históricos no se puede verificar con estos archivos.

## 3. Datos y protocolo temporal (80 segundos)

Trabajamos con FOODS_3_090, elegido por ventas agregadas únicamente en entrenamiento. Las tiendas son CA_1, CA_2 y CA_3. El último año se reservó para test. Cada método entrenable usa las semillas 3, 4 y 5, con 300 episodios de 365 días. Son semillas de desarrollo, no una campaña confirmatoria independiente. Se usó el último checkpoint, sin elegir semillas según test. PPO fue la familia con mayor retorno medio de validación, y su semilla 3 quedó fijada para la demo principal.

## 4. Estado, acciones y recompensa (90 segundos)

La observación local contiene existencias, tránsito, pronóstico de 28 días, precio, calendario, horizonte restante e inventario regional. No garantiza Markovianidad completa. Hay cinco multiplicadores y tres objetivos de cobertura: 15 acciones. El volumen solicitado se calcula con inventario y pronóstico, por eso puede cambiar aunque el índice de acción no cambie. Las ventas históricas se modulan con elasticidad supuesta 1.5. Los despachos de hoy llegan mañana. La recompensa regional suma ingresos menos almacenamiento y faltantes, sin el bono de shaping en las métricas reportadas.

## 5. Comparación y ablaciones (80 segundos)

El baseline tabular reutiliza el algoritmo de P1, con tablas nuevas entrenadas para retail. DDQN representa la familia de valor y PPO la familia Actor-Crítico. Los agentes tienen redes locales. La condición egoísta recibe recompensa local, las demás reciben la regional. V1 aísla PBRS, V2 aísla Dyna y V3 combina ambos. N0 añade un bono ingenuo. Se mantiene el presupuesto de interacciones reales, pero no el número de actualizaciones ni el coste computacional.

## 6. PBRS: invariancia del objetivo descontado (100 segundos)

El potencial penaliza el desequilibrio de cobertura y la cobertura menor de tres días. Se usa la diferencia descontada de potencial. Sumando las recompensas descontadas, los términos interiores se cancelan. Con potencial terminal cero queda el retorno original menos el potencial inicial. Dado el mismo estado inicial, esa cantidad no depende de las decisiones futuras, así que no cambia el orden de las políticas. La identidad usa el estado completo y el objetivo descontado. No garantiza que redes y agentes independientes aprendan más rápido, ni que mejore el retorno sin descuento a presupuesto finito.

## 7. Bono ingenuo y cambio de preferencia (70 segundos)

Este es un experimento controlado de dos pasos, separado de M5. Servir obtiene mayor retorno base que acaparar. El bono ingenuo recompensa cobertura y hace que acaparar parezca mejor. Con PBRS ambas trayectorias reciben la misma corrección total de más uno porque el potencial inicial es menos uno, y la preferencia original se conserva. No afirmamos que N0 haya explotado este mecanismo en M5: sus resultados empíricos se analizan por separado.

## 8. Dyna híbrido: demanda aprendida (90 segundos)

El modelo aprende log de uno más demanda usando observación, acción e identidad de tienda. Sus etiquetas proceden de transiciones reales del simulador en entrenamiento. La demanda posterior a la acción es privilegiada para ajustar el modelo, no se entrega a la política antes de decidir. Inventario y recompensa siguen ecuaciones exactas. Dyna genera transiciones sintéticas de un paso para actualizar DDQN. Esas transiciones no vuelven a entrenar el modelo ni se mezclan con el replay real. Más experiencia sintética puede introducir sesgo y no garantiza más retorno.

## 9. Aprendizaje en validación (90 segundos)

La curva muestra medias de tres semillas para tabular, DDQN y PPO. Las bandas de una desviación estándar y las ablaciones están en el informe y las figuras originales. El eje cuenta interacciones reales. PPO acaba con mayor retorno de validación entre agentes entrenados, pero por debajo de la heurística. Las curvas de entrenamiento suman recompensas sin descuento sobre ventanas diferentes. Las pérdidas tampoco son una escala común de calidad entre algoritmos.

## 10. Test: retorno y quiebres (110 segundos)

En test cambia el orden observado en validación. Q-learning es el mejor agente entrenado por retorno medio y queda 0.27 por ciento por debajo de H0. PPO queda 2.35 por ciento por debajo de H0 en retorno, pero tiene el mejor fill rate medio y la menor dispersión de retorno entre métodos aprendidos. Esto no autoriza cambiar retrospectivamente la familia elegida en validación. La tasa de quiebres cuenta días con pérdidas, el fill rate cuenta unidades. S0 tiene gran dispersión. Ninguna corrida satisface conjuntamente ambos umbrales de reducción y el beneficio.

## 11. Ablaciones: efectos observados en test (80 segundos)

Las diferencias están emparejadas por semilla y son descriptivas. DDQN con Dyna pierde en promedio 6591.5 unidades de retorno frente a DDQN. PBRS solo mejora dos de tres semillas pero su diferencia media es negativa. La diferencia de DDQN cooperativo frente al egoísta está dominada por una corrida egoísta muy mala: sólo una de tres diferencias es positiva. No se concluye superioridad estadística ni se atribuye el resultado a una sola causa.

## 12. Eficiencia y estabilidad (70 segundos)

El umbral descriptivo es 72143.5, el 80 por ciento del máximo empírico de validación. Se fijó antes de abrir test, pero después del desarrollo. Las evaluaciones ocurren cada 3650 pasos: un cruce en la primera evaluación no identifica el instante exacto ni diferencia métodos que cruzaron antes. DDQN, PBRS y Dyna tienen la misma mediana de primer cruce. Esta resolución no acredita aceleración adicional. La estabilidad usa los últimos diez checkpoints de validación.

## 13. El modelo de demanda fuera de muestra (70 segundos)

El MAE medio del modelo es menor que el del pronóstico de 28 días en ambas condiciones. Sin embargo, el RMSE es mayor. La mejora en error absoluto convive con errores grandes más penalizados por RMSE. Esto no demuestra la causa del menor retorno con Dyna. Las métricas se calculan sobre la trayectoria propia de cada política, con demandas dependientes del precio. No son un benchmark idéntico entre V2 y V3. Son predicciones de un paso y no prueban fidelidad de rollouts largos.

## 14. Demostración de las políticas congeladas (120 segundos)

Abrir demo_p2.html. La primera trayectoria sigue siendo PPO semilla 3, elegida antes de test. Mostrar Primer quiebre: día 27. PPO mantiene acciones constantes: CA_1 usa multiplicador 0.95 y las otras 1.10, todas con objetivo de siete días. El pedido cambia por inventario y pronóstico, pero no hay cambios diarios de precio aprendidos en esta trayectoria. No existe racionamiento regional aquí. Cambiar al ejemplo DDQN semilla 3 y pulsar Primer racionamiento: día 63, d1639. La suma de pedidos supera despachos y éstos llegan mañana. Este segundo caso se eligió después del test sólo para explicar el mecanismo.

## 15. Conclusiones y límites (40 segundos)

La heurística sigue siendo una referencia fuerte y mayor complejidad no garantizó mayor retorno con este presupuesto. Q-learning obtuvo el mejor retorno medio aprendido en test. PPO se había seleccionado por validación. La experiencia sintética no demostró una ventaja final consistente. La meta conjunta no se alcanzó. La elasticidad es supuesta y M5 no identifica los quiebres ni beneficio históricos. Presentamos la evaluación del simulador con esos límites explícitos.

Duración total: 1200 segundos. La demo está incluida en los 20 minutos.
