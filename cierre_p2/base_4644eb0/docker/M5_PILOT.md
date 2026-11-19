# Piloto M5 de desarrollo

## Evidencia recibida del Pod

El usuario ejecutó el commit ba3ad670fd697c38bc9a5fd9c5101ea99c723303:
53 tests, cero fallos, cero errores, cero omisiones. Preflight con CUDA y smoke
de DDQN/PPO/DDQN+PBRS+Dyna completados en RTX 5080 (16 GB), Python 3.12.3,
PyTorch 2.8.0+cu129. Registro remoto:
`/workspace/rl-project-validation/p2-rII1jB/validation.log`.
Esta evidencia es la salida comunicada por el usuario; no una ejecución directa del conector.

Los tres CSV de Zenodo 10203108 pasaron MD5. La auditoría posterior seleccionó
FOODS_3_090 por ventas de train, con precios completos en CA_1/CA_2/CA_3.
SHA256 y estadísticas están en `/workspace/data/m5/selected-sku-audit.json`.
Los días con ventas cero no identifican por sí solos quiebres de stock.

## Preparación y ejecución

Desde el checkout actualizado:
```bash
python3 scripts/m5_pilot.py --data-dir /workspace/data/m5/raw \
  --audit /workspace/data/m5/selected-sku-audit.json \
  --output /workspace/rl-project-validation/m5-pilot-01 --execute
```

Sin --execute solo prepara. La carpeta debe ser nueva. Cinco trabajos secuenciales:
H0 y T0/V0/P0/V3, semilla de desarrollo 3; ocho episodios de 365 días por agente.
Cada agente usa 2.920 pasos regionales reales; evaluación en validación cada cuatro
episodios. No hay evaluación de test ni ranking confirmatorio.
Las ventanas se emparejan por semilla. Dyna añade cómputo explícito.
Cada trabajo tiene un límite de 30 minutos y conserva su log y artefactos disponibles.
Ante una interrupción no relanzar a ciegas: revisar logs/checkpoints; T0 no reanuda.

Supuestos iniciales, calculados exclusivamente con train:
- Capacidad local común: techo de 7 veces la mayor media diaria local (1.050).
- Capacidad regional: techo de 7 veces la suma de medias (1.768).
- Suministro diario regional: techo de la suma de medias (253).
- Stock inicial local común: techo de 3 veces la mayor media (450).
- Stock inicial regional: techo de 3 veces la suma (758).
- Mantenimiento: 1% del precio mediano train por unidad y día (0,0128).
- Penalización por demanda perdida: precio mediano train por unidad (1,28).
- Elasticidad supuesta: 1,5 en cada tienda.

Son decisiones de un escenario piloto, no parámetros observados de Walmart.
La capacidad/stock local común produce distinta cobertura por tienda.
Los costos y elasticidades requieren sensibilidad; no se fijan como verdaderos.
Se usa horizonte anual para evitar comparar episodios de 28 días con una
evaluación de 365 días. El piloto mide estabilidad, tiempos y comportamiento
preliminar; ocho episodios no garantizan convergencia.

El script comprueba los hashes contra la auditoría antes de ejecutar.
Guarda configuraciones completas, auditoría original, plan, logs, checkpoints,
métricas de validación, pasos y tiempos totales en summary.json.
El tiempo total incluye carga de CSV, evaluación y guardado. El tiempo de
entrenamiento por episodio también queda en los artefactos de cada agente.

Después: revisar métricas y curvas, formular los ajustes del piloto y sensibilidad,
congelar protocolo/presupuesto, ejecutar semillas finales 0/1/2 y evaluar test al final.
