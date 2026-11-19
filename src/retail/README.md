# P2 — Vía B: contratos antes de implementar

Estado: **setup y diagnóstico disponibles; no hay todavía entorno ni runner de entrenamiento retail**.
La especificación y la matriz de requisitos están en [el plan P2](../../informe/plan_p2_via_b.md).
`experiment_plan.json` es un plan estructurado: los `null` representan decisiones pendientes y no deben convertirse silenciosamente en valores por defecto.

## Setup disponible

- `bash docker/setup_p2.sh`: construir, comprobar dependencias/tests y arrancar Jupyter desde Ubuntu.
- `python -m src.retail.doctor --smoke`: comprobar dependencias e integración CPU con CartPole.
- `python -m src.retail.doctor --skip-dependencies --require-data`: sólo presencia/cabeceras de M5, usando Python estándar.
- `python -m unittest discover -s src/retail/tests -v`: tests del diagnóstico.

Consultar el [README raíz](../../README.md) para los requisitos, rutas de datos y estado de verificación. La comprobación de cabeceras no equivale a auditar el dataset, y el smoke DQN/PPO no equivale a implementar los agentes retail.

## Orden de implementación

| Módulo previsto | Responsabilidad / criterio de aceptación |
|---|---|
| `data.py` | Validar los tres CSV, filtrar tiendas y SKU(s), unir sin duplicados, construir split y manifiesto. Todo ajuste usa train. |
| `env.py` | Una transición regional con tres acciones simultáneas; inventario, racionamiento y tránsito conservativos; recompensa base separada. |
| `baselines.py` | Precio histórico + base-stock y adaptación de `src.agents.QLearningAgent`; nunca cargar tablas de fútbol. |
| `value.py` | Tres DDQN locales; replay y target network; control egoísta con idénticas restricciones. |
| `policy.py` | PPO con GAE y acciones locales; sincronizar el día regional antes de avanzar. |
| `shaping.py` | Potencial acotado, corrección terminal e identidad telescópica; bono ingenuo separado. |
| `world_model.py`, `dyna.py` | Demanda aprendida + reglas exactas; rollout conjunto; distinguir transiciones del simulador y sintéticas. |
| `train.py`, `evaluate.py` | Semillas, checkpoints, presupuesto y test bloqueado; nunca ajustar durante la evaluación final. |
| `analysis.py` | Retornos base, quiebres, eficiencia, estabilidad, pérdidas y bandas ±1σ. |

Estos nombres describen archivos **por crear**. No se agregan imports vacíos ni comandos de entrenamiento que aparenten estar implementados.

## Primer incremento verificable

1. Resolver ubicación de `sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv` y registrar hashes.
2. Auditar fechas, claves, disponibilidad y SKU con train; completar supuestos del plan.
3. Implementar el entorno determinista y H0; probar conservación de unidades, pedidos simultáneos, escasez, lead time, límites y ausencia de futuro en observaciones.
4. Registrar un rollout legible por día/tienda antes de implementar redes.

Los tests de setup están en `src/retail/tests/` y los análisis futuros irán en `notebooks/p2/`. Los artefactos van en `notebooks/artifacts/p2_via_b/`, con esquema distinto de P1. `docker/docker-compose.p2.yml` prepara el entorno de desarrollo sin `rcssserver`; todavía no ejecuta entrenamiento retail. El Compose raíz actual sigue siendo P1.

El cambio de dominio requiere un nuevo estado/acción/recompensa. El código tabular puede reutilizarse, pero deben comprobarse los terminales: el horizonte comercial con tiempo en estado no hereda automáticamente el bootstrap por truncamiento del fútbol.
