# Calibración de la referencia histórica de M5

Desde la raíz del repositorio en Ubuntu o WSL, ejecuta la calibración con uv. El archivo `cierre_p2/requirements.txt` fija la única dependencia externa a la versión de NumPy del proyecto.

```bash
cd ~/RL-Project
uv run --python 3.11 --with-requirements cierre_p2/requirements.txt cierre_p2/calibrar_m5.py --descargar
```

El comando obtiene de Zenodo los tres CSV originales que falten y valida su SHA-256 con el protocolo de las corridas. Si ya están en el Pod o en otro directorio, usar `--datos /workspace/data/m5/raw`. Escribe `cierre_p2/calibracion_m5.json`. La copia de las fuentes de retail en `base_4644eb0/` corresponde al commit registrado en el protocolo. Este directorio contiene una selección de archivos del proyecto.

El ajuste usa FOODS_3_090 en las tres tiendas durante d1–d1211. Por tienda se ajusta una regresión de Poisson con enlace logarítmico sobre precio en logaritmos, tendencia, estacionalidad anual, día de semana, eventos y SNAP. El signo de la magnitud de elasticidad sigue la convención del simulador, −coeficiente de precio. Se conservan los ceros observados, se reporta el soporte del precio y se hacen tres evaluaciones temporales internas de 182 días. Precio constante o colinealidad completa dan un resultado indeterminado. Pocos cambios de precio, signos inestables y demanda censurada limitan la interpretación. El informe diagnóstico conserva los signos observados para revisión antes de incorporar estimaciones a los entrenamientos.

Para H0 se usa el precio histórico y se elige por tienda el menor objetivo de cobertura disponible entre 0, 3 y 7 días que habría cubierto las ventas acumuladas de dos días en el 90 % de los casos de entrenamiento. La ventana representa el día de revisión y la llegada al día siguiente. El cálculo usa el pronóstico anterior de 28 días y la regla de pedido del simulador. El porcentaje corresponde a cobertura de ventas históricas en entrenamiento. El servicio realizado con capacidad regional finita se mide después en validación. El 90 % es un supuesto de trabajo con sensibilidad al 80 % y 95 %. Si ninguna acción alcanza el objetivo se informa el resultado indefinido. La validación usa los días 1212–1576 y mantiene la elasticidad existente en el simulador para H0, cuyo multiplicador es siempre uno. Para un mismo recorrido se calcula la sensibilidad económica al multiplicar cada coste por 0,5, 1 o 2. Estos escenarios expresan incertidumbre sobre los costes. Las ventas históricas limitan la información disponible sobre demanda perdida real.

El desglose separado `desglose_H0_v1.json` reconstruye la contabilidad de la referencia anterior en el año de test ya publicado, sin usarla para calibrar. Los precios observados son semanales. Los ingresos aproximados usan las unidades históricas, recuperables en la traza de H0 al multiplicador 1. El beneficio de H0 usa las ventas que permite la simulación y sus costes supuestos.

Las pruebas específicas se ejecutan con `uv run --python 3.11 --with-requirements cierre_p2/requirements.txt cierre_p2/test_calibrar_m5.py`. Para la campaña revisada queda revisar los diagnósticos numéricos con M5, fijar la configuración y evaluar los agentes bajo las mismas reglas. La reproducción Docker completa y la inferencia en vivo requieren verificación independiente.

## Resultado de la calibración y comparación en validación

Las huellas de los tres CSV coinciden con las registradas en `m5-final-01.zip`. Con los días 1 a 1211, el umbral principal del 90 % induce objetivos de inventario de 7, 7 y 3 días para CA_1, CA_2 y CA_3. La sensibilidad al 95 % induce 7 días para cada tienda y reproduce H0 de la campaña original. Ambos escenarios usan los precios semanales de M5. El nivel de cobertura deseado y los costes son supuestos del análisis. El inventario histórico real de Walmart no está observado.

| Política | Retorno simulado en validación | Días-tienda con quiebre entre los elegibles |
|---|---:|---:|
| H0 al 90 % | 54 776,80 | 18,64 % |
| H0 al 95 % | 88 129,71 | 2,11 % |
| PPO cooperativo existente, media de tres semillas | 86 091,71 | 2,23 % |

Los tres resultados corresponden a los días 1212 a 1576. Los de PPO proceden de la fila de validación de P0 en `results_table.csv` del paquete original. En CA_3 el error absoluto medio entre las ventas simuladas de H0 y las observadas pasa de 36,93 unidades diarias con tres días de inventario a 1,41 con siete. Conviene conservar ambas referencias al comunicar la comparación.

El diagnóstico de precios se puede reproducir con `uv run --python 3.11 --with-requirements cierre_p2/requirements.txt cierre_p2/diagnostico_precios.py --salida precios_revision.json` (usar una ruta de salida nueva). Hubo 25 semanas con precios distintos entre tiendas y alguna venta, sobre 173 semanas de entrenamiento. La elasticidad común con tendencia por tienda cambia de −2,22 en las primeras 95 semanas a +2,20 al incluir las 173. En el simulador una elasticidad positiva reduce la demanda al elevar el precio. El cambio de signo limita el uso de esta asociación para predecir qué habría sucedido a otros precios. Las comparaciones de las políticas existentes conservan el supuesto de elasticidad de su campaña.

## Comprobación de la meta conjunta en el último año

Reanálisis posterior a la campaña, en los días 1577 a 1941. Se simularon las dos reglas H0 inducidas con entrenamiento y se contrastaron con las métricas congeladas del ZIP original. El criterio por corrida exige retorno superior a H0 y al menos 30 % de reducción relativa de quiebres frente a H0 y frente a S0 de la misma semilla. Las políticas conservan la elasticidad supuesta de su entrenamiento. El año de prueba ya se había examinado.

| Referencia | Retorno H0 | Quiebres H0 | Corridas que cumplen entre las 24 entrenadas | Semillas PPO que cumplen |
|---|---:|---:|---:|---|
| H0 al 90 % | 83 353,74 | 11,88 % | 7 | 3 y 4 |
| H0 al 95 % | 98 915,49 | 1,26 % | 0 | Ninguna |

Con H0 al 90 %, las siete corridas son T0, V0, P0, V2, V3 y N0 de la semilla 3, más P0 de la semilla 4. PPO obtuvo reducciones de quiebres frente a S0 del 94,42 %, 42,86 % y 20,00 % en las semillas 3, 4 y 5, respectivamente. Frente a H0 al 90 %, las reducciones fueron 87,61 %, 89,38 % y 89,38 %. H0 al 95 % reproduce el retorno y la tasa de quiebres de la referencia archivada.
