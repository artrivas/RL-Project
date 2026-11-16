# P2 — Vía B: implementación y validación

La rama contiene carga M5, simulador regional, H0, T0, DDQN, PPO, PBRS,
bono ingenuo, Dyna de un paso y análisis multisemilla. **El código profundo
todavía requiere validación numérica con PyTorch; no hay resultados M5.**

P1 se conserva sin cambios. T0 importa `src.agents.QLearningAgent` y crea
tres tablas nuevas. Esta rama parte de main; no integra commits de dimael.

## Estado comprobado

Las pruebas sin PyTorch cubren datos, conservación de inventario, simultaneidad,
lead time, terminales tabulares, ausencia de futuro, PBRS, política congelada,
selección de SKU y construcción de la matriz experimental.
Las pruebas profundas están en `tests/test_deep.py`: DDQN, GAE, clipping PPO,
actualización de redes, separación de datos Dyna, evaluación congelada y equivalencia
entre reanudar y entrenar sin interrupción. Se omiten cuando falta PyTorch.
**Una suite con omisiones no certifica el código profundo.**

En el entorno de implementación falta PyTorch y su instalación falló por resolución
de red. Docker y CUDA no se han probado aquí. El siguiente paso es ejecutar el
preflight y el smoke pequeños en un entorno con PyTorch, antes de cualquier campaña.

## Dependencias y primer chequeo remoto

Python 3.11/3.12, NumPy y PyTorch 2.8.0; Matplotlib para gráficos.
`docker/requirements.p2.core.txt` declara NumPy. La instalación de PyTorch debe usar
la rueda oficial compatible con la GPU/CUDA del Pod; la receta Docker de desarrollo
existente instala **CPU**, no debe asumirse una imagen de entrenamiento GPU.
No se requiere Gymnasium ni Stable-Baselines3 para nuestros agentes retail.

Desde un checkout de esta rama:

```bash
python -m unittest discover -s src/retail/tests -v
python -m src.retail.preflight --require-cuda --report notebooks/artifacts/p2_via_b/preflight.json
python -m src.retail.smoke_deep --device cuda --output notebooks/artifacts/p2_via_b/gpu_smoke
```

El preflight falla si falta una dependencia, falla un test o se omite alguno.
El smoke entrena DDQN, PPO y DDQN+PBRS+Dyna durante ocho pasos reales cada uno,
incluyendo guardar y reanudar. Son fixtures artificiales, no experimentos M5.
Los resultados de CPU/GPU no se presuponen idénticos.

## Datos y supuestos

Se necesitan `sales_train_evaluation.csv`, `calendar.csv` y `sell_prices.csv`.
La auditoría selecciona, si no se fija uno, el SKU de mayor venta en train presente
con venta positiva en CA_1, CA_2 y CA_3. Desempata por ID. No usa valores de
validación/test para seleccionar. Valida calendario y los valores y joins del SKU
elegido, no todos los SKU. Precios faltantes causan error; no se rellenan con futuro.
Un SKU de selección automática puede carecer de precios desde d_1: se debe revisar
su disponibilidad y documentar otro criterio de selección basado en train.

Splits: train d_1–d_1211, validación d_1212–d_1576, test d_1577–d_1941.
Congelar el SKU para toda la matriz. El test se usa con el evaluador explícito al final.
La normalización del encoder se ajusta solo con train y se guarda en el checkpoint.

M5 no observa inventarios, reposición, costos, beneficio neto ni demanda perdida.
Un JSON `assumptions.json` debe fijar:

| Campo | Significado |
|---|---|
| `store_capacity`, `regional_capacity` | Capacidades enteras positivas |
| `regional_supply` | Máximo de unidades externas recibidas cada día |
| `initial_store_stock`, `initial_regional_stock` | Inventario inicial |
| `holding_cost`, `shortage_cost` | Costos supuestos por unidad |
| `elasticities` | Tres elasticidades no negativas |

Opcionales: `price_multipliers`, `target_days`, `history_days`.
`demo_config()` es un fixture, no una calibración para M5.
Los null de `experiment_plan.json` son decisiones pendientes, no defaults utilizables.

## Entorno y recompensa

Un SKU, tres acciones simultáneas de precio × cobertura objetivo. Antes de decidir
llegan pedidos previos y reposición externa limitada por capacidad. El reparto regional
es proporcional con restos enteros y desempate CA_1→CA_2→CA_3 (asimetría conocida).
Los envíos llegan al día siguiente y no satisfacen la demanda actual.
El inventario se conserva; no hay compras con entrega instantánea.

Demanda de referencia: ventas históricas × multiplicador^(-elasticidad),
redondeada al entero más cercano con mitades hacia arriba. Es una aproximación de
simulación, no demanda causal ni no censurada. La previsión operativa usa ventas
pasadas y, durante el episodio, ventas realizadas por la política. Puede subestimar
demanda bajo escasez. Los agentes no observan demanda futura.

Recompensa base = ingresos − mantenimiento de stock final − penalización de demanda
perdida. Se etiqueta **proxy de beneficio operativo simulado**. No incluye costos
de compra desconocidos. Los episodios son horizontes finitos, sin rescate terminal,
con tiempo restante observable. Entrenar con ventanas cortas y evaluar un año introduce
un cambio en el horizonte que debe estudiarse en el piloto.

H0 conserva precios históricos y aplica base-stock; es una referencia construida,
no la reposición observada de Walmart. La reducción de quiebres es simulada frente a H0/S0.
La reducción porcentual con baseline de cero quiebres queda indefinida.

## Familias, shaping y Dyna

- **T0:** tres tablas Q independientes con recompensa regional y estado agregado.
- **DDQN:** tres MLP independientes, replay real conjunto, Huber, Adam y red objetivo.
  La selección de la acción siguiente usa la red online; su valoración usa la objetivo.
- **PPO:** tres actores/críticos independientes, distribución categórica local, GAE,
  objetivo recortado, entropía y minibatches. Los cortes de rollout hacen bootstrap;
  los terminales reales no. Es PPO independiente, no un crítico centralizado.
- **PBRS:** `weight * (gamma * Phi(next) - Phi(current))`, potencial acotado basado
  en dispersión de cobertura y déficit, y Phi terminal = 0. Usa el mismo gamma del agente.
  El test comprueba la identidad telescópica. La invariancia del objetivo no garantiza
  igual convergencia con aproximación, observación parcial y presupuesto finito.
- **Naive:** bono por cobertura, diseñado para contrastar un cambio de objetivo.
  Hay prueba del incentivo a acumular stock; la existencia de reward hacking aprendido
  se debe demostrar con resultados, no se presupone.
- **Dyna:** MLP compartida condicionada por tienda y acción, entrenada sobre log(1+demanda)
  de transiciones reales de train. La demanda perdida es un diagnóstico privilegiado
  del simulador disponible después de actuar; no equivale a supervisión observada en M5.
  Inventario, racionamiento y recompensas se calculan con reglas exactas.
  Los calendarios y precios de referencia se tratan como programas exógenos conocidos.
  No se consulta venta histórica futura al generar demanda sintética.
  Rollout de **un paso conjunto**, una actualización sintética de tamaño 1 por paso
  real tras calentamiento por defecto. La predicción se limita a 10×capacidad local.
  Datos sintéticos no entran al replay real ni al ajuste del modelo. Sus pasos,
  pérdidas y costo adicional se registran por separado.

El estado local es parcialmente observable; no se presupone suficiente ni óptimo.
Las tres tiendas comparten recompensa en variantes cooperativas, pero no parámetros
de política. DDQN y PPO reciben el mismo encoder; T0 emplea su discretizador.
La escala positiva de recompensa es común (100 por defecto) y las métricas usan
siempre recompensa base sin escalar ni shaping. Los hiperparámetros iniciales de
`DeepConfig` son propuestas de piloto, no valores ajustados ni resultados.

Referencias: [Double DQN](https://arxiv.org/abs/1509.06461),
[PPO](https://arxiv.org/abs/1707.06347).

## Entrenamiento, checkpoints y evaluación

```bash
python -m src.retail.train_deep --demo --algorithm ddqn --seed 3 --episodes 2 --output notebooks/artifacts/p2_via_b/local_demo
python -m src.retail.train_deep --demo --algorithm ddqn --seed 3 --episodes 4 --output notebooks/artifacts/p2_via_b/local_demo --resume notebooks/artifacts/p2_via_b/local_demo/checkpoint.pt
```

Para M5 sustituir `--demo` por `--data-dir DIR --config assumptions.json --item-id ITEM_ID`.
`--settings settings.json` acepta campos de `DeepConfig`; `--device cuda` usa GPU.
Una corrida nueva rechaza una carpeta existente. La reanudación profunda guarda al
final de cada episodio: redes, optimizadores, replay, modelo Dyna, RNG y progreso.
Requiere el mismo código, datos, configuración, dispositivo y versión de PyTorch.
Se conserva el último episodio completo; no se promete recuperar la mitad de uno.
T0 guarda tablas para evaluación; su runner todavía no reanuda entrenamiento.

```bash
python -m src.retail.evaluate --checkpoint RUN/checkpoint.pt --data-dir DIR --split test --output OUT
python -m src.retail.evaluate_tabular --run T0_RUN --data-dir DIR --split test --output OUT_T0
python -m src.retail.rollout --data-dir DIR --config assumptions.json --item-id ITEM_ID --split test --policy h0 --output OUT_H0
```

Los evaluadores no actualizan políticas. Validar y elegir configuraciones antes de
consultar test. Aún no se ha elegido un ganador ni cumplido la meta de reducción del 30%.

## Matriz y análisis

```bash
python -m src.retail.suite --data-dir DIR --config assumptions.json --item-id ITEM_ID --episodes 100 --device cuda --output SUITE
```

Sin `--execute` solo prepara configuraciones y comandos. Con `--execute` ejecuta
secuencialmente 25 trabajos: H0 una vez y T0/V0/P0/S0/V1/V2/V3/N0 con semillas 0,1,2.
Todos los entrenables usan episodios×horizonte pasos regionales reales.
Las ventanas de entrenamiento se emparejan por semilla; Dyna añade cómputo, que se
informa aparte. El presupuesto final debe fijarse con pilotos y conservarse entre variantes.
No hay test en la matriz de entrenamiento.

```bash
python -m src.retail.analysis --suite SUITE --output ANALYSIS --plot
```

Genera resumen por familia (media±desviación muestral), curvas CSV de retorno/quiebres,
pérdidas CSV y figura PNG opcional. Rechaza mezclas de supuestos, SKU, datos, ventana
de evaluación, semilla o presupuesto incompatible. Señala corridas faltantes y no
presenta comparaciones incompletas como confirmatorias.

La eficiencia al 80% usa `baseline + 0.8*(reference-baseline)`, con baseline y referencia
fijados previamente mediante `--baseline-return` y `--reference-return`. Si no se
alcanza, se registra como censurado; no se sustituye por cero. La dispersión entre
semillas es descriptiva y no constituye por sí sola una prueba de significancia.

## Trabajo pendiente antes de resultados

1. Ejecutar todos los tests numéricos y el smoke en PyTorch/CUDA y corregir sus fallos.
2. Obtener/auditar M5 y congelar SKU, supuestos, presupuesto y referencia de eficiencia.
3. Piloto remoto, entrenamiento multisemilla y evaluación final separada.
4. Agregar resultados, discutir hipótesis, elaborar informe y presentación.

El Pod no ha sido creado ni conectado. La implementación publicada sigue en un PR borrador.
