# P2 — Vía B: entorno retail y baselines

Primer incremento implementado: carga M5, simulador regional, H0 y Q-learning T0.
DDQN, PPO, PBRS y Dyna siguen pendientes. No hay resultados experimentales M5.

El código de P1 se conserva en la versión de `main`. T0 importa
`src.agents.QLearningAgent` sin modificarlo y crea tres tablas nuevas.
La rama `dimael` contiene trabajo adicional que no forma parte de esta rama.

## Verificación y ejecución

Desde la raíz del checkout, con Python y NumPy:

```bash
python -m unittest discover -s src/retail/tests -v
python -m src.retail.rollout --demo --policy h0 --output notebooks/artifacts/p2_via_b/demo_h0
python -m src.retail.rollout --demo --policy random --seed 3 --output notebooks/artifacts/p2_via_b/demo_random
```

Los datos de `--demo` son artificiales y sus métricas no son evidencia M5.
Los comandos rechazan carpetas de salida existentes. Cada corrida guarda configuración,
huellas SHA-256 de código y datos y métricas sin shaping; los rollouts guardan también
transiciones JSONL y un CSV diario por tienda.

Para M5, colocar los tres CSV indicados en `notebooks/data/m5/README.md`.
El siguiente comando es de evaluación de H0, no de entrenamiento:

```bash
python -m src.retail.rollout --data-dir notebooks/data/m5/raw --config /ruta/assumptions.json --item-id ITEM_ID --split validation --output notebooks/artifacts/p2_via_b/h0_validation
```

`assumptions.json` debe proporcionar TODOS estos campos:

| Campo | Significado |
|---|---|
| `store_capacity`, `regional_capacity` | Capacidades enteras positivas |
| `regional_supply` | Máximo de unidades externas recibidas cada día |
| `initial_store_stock`, `initial_regional_stock` | Inventario inicial explícito |
| `holding_cost`, `shortage_cost` | Costos supuestos por unidad |
| `elasticities` | Tres elasticidades no negativas, una por tienda |

Opcionales: `price_multipliers`, `target_days`, `history_days`.
La configuración artificial de `demo_config()` sirve para probar software;
no es una calibración recomendada para M5. El plan general tiene campos nulos
pendientes y no se acepta como configuración del simulador.

T0 tiene un punto de entrada listo para ejecutar posteriormente en el entorno remoto:

```bash
python -m src.retail.train_tabular --data-dir notebooks/data/m5/raw --config /ruta/assumptions.json --item-id ITEM_ID --seed 0 --episodes 100 --horizon 28 --output notebooks/artifacts/p2_via_b/t0_seed0
```

T0 usa ventanas de entrenamiento contenidas en d_1–d_1211, recompensa regional,
exploración lineal de 1.0 a 0.05, y evalúa una vez en validación d_1212–d_1576.
Guarda las tablas NPZ y el log por episodio. No toca test ni selecciona hiperparámetros.
El presupuesto de 100 episodios es un valor operativo inicial, no el presupuesto final
del estudio. El estado tabular es una agregación parcial (cobertura local/regional,
demanda pasada, día de semana y tiempo restante); no se presupone suficiente ni óptimo.
Las tres tablas comparten recompensa pero no parámetros.
Aún falta reanudación de entrenamiento y evaluación final desde checkpoint.

## Contrato del simulador

- Un SKU, tiendas CA_1, CA_2 y CA_3, acciones locales precio × objetivo de cobertura.
- Antes de decidir llegan los pedidos del día anterior y la reposición regional,
  limitada por la capacidad. Las tres tiendas deciden sobre el mismo estado regional.
- Se pide el déficit hasta el objetivo, limitado por la capacidad local. Se raciona
  proporcionalmente con restos enteros; los empates favorecen CA_1, luego CA_2 y CA_3.
  Este desempate introduce una pequeña asimetría conocida.
- Los envíos de hoy permanecen en tránsito hasta mañana y no satisfacen demanda hoy.
- Demanda contrafactual = ventas históricas × multiplicador elevado a menos elasticidad,
  redondeada al entero más cercano (mitades hacia arriba). Es un supuesto de simulación,
  no una estimación causal de demanda no censurada.
- La previsión operativa es la media de ventas observadas en los últimos 28 días;
  arranca con historia anterior al episodio y después incorpora ventas simuladas.
  Bajo stockouts puede subestimar demanda; no observa ventas futuras ni demanda perdida.
- Los episodios tienen horizonte finito real con tiempo restante observable;
  no hay bootstrap terminal ni valor de rescate. Cambiar esto requiere otro contrato.
- Recompensa base: ingresos menos mantenimiento del inventario final y penalización
  de demanda perdida. No incluye costo de compra desconocido. Es un
  **proxy de beneficio operativo simulado**, no beneficio histórico Walmart.
- H0 es precio histórico + base-stock, una política de referencia construida.
  No reconstruye la política de reposición histórica.

## Datos y alcance de la auditoría

`data.py` selecciona, si no se indica uno, el SKU de mayor venta total en train,
con ventas positivas en las tres tiendas; desempata por ID. La selección no consulta
valores de validación/test. Se recomienda congelar el ID en todas las corridas.
La auditoría valida claves de ventas de las tres tiendas, calendario y todos los valores
y joins de precio del SKU seleccionado. No certifica todos los SKU del dataset.

Un precio faltante hace fallar la carga: no se rellena con precios futuros. Un SKU
seleccionado automáticamente puede no tener precios desde d_1; en ese caso se debe
revisar su disponibilidad y fijar otro SKU mediante un criterio de entrenamiento
documentado. No se cambia automáticamente de SKU según rendimiento de validación/test.

Train: d_1–d_1211; validación: d_1212–d_1576; test reservado: d_1577–d_1941.
Las ventas no son demanda no censurada, ni M5 permite observar quiebres reales.
Las reducciones de quiebres se medirán dentro del simulador contra H0 y S0.
La reducción porcentual es indefinida si el baseline tiene cero quiebres.

## Estado y continuación

- Pruebas unitarias: conservación, capacidad, simultaneidad, lead time, precios,
  ausencia de futuro, parsing/joins, reproducibilidad, terminales y congelación de Q.
- Validación realizada con fixtures artificiales; faltan los CSV M5.
- Docker/Compose son una receta opcional de desarrollo; no han sido construidos
  en esta sesión. El Pod se preparará después. No se ha lanzado entrenamiento real.
- Próximos incrementos: calibrar/fijar supuestos, DDQN y PPO con presupuesto común,
  PBRS y bono ingenuo, Dyna híbrido, checkpoints reanudables y análisis multisemilla.

Ver [el protocolo propuesto](../../informe/plan_p2_via_b.md).
