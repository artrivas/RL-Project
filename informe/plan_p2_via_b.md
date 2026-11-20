> **Estado de cierre (16-11-2026):** el usuario compartió validación completa de 25 corridas y manifiestos con 109500 pasos por política. El cierre congelado en test está implementado en `scripts/launch_m5_final.sh`; aún debe ejecutarse y revisarse su paquete. Véase [M5_FINAL.md](../docker/M5_FINAL.md).
>
> **Enmienda:** se cierra descriptivamente con semillas de desarrollo 3/4/5; no se ejecuta la campaña confirmatoria 0/1/2 del diseño inferior. La elasticidad sigue siendo un supuesto.
>
> **Archivo histórico:** las menciones siguientes a componentes pendientes y ausencia de resultados describen la planificación inicial, no el estado actual. Se conservan para trazabilidad.

> Segundo incremento: DDQN/PPO, PBRS, bono ingenuo, Dyna de un paso, checkpoints profundos y análisis implementados. La validación numérica con PyTorch/CUDA queda pendiente; consultar src/retail/README.md. Los supuestos de M5 siguen sin calibrarse.

> Actualización de implementación (16-11-2026): entorno, carga M5, H0 y T0 disponibles en src/retail. El texto inferior conserva el diseño original; las referencias a módulos pendientes se deben contrastar con src/retail/README.md. No hay resultados M5. Rama basada en main 79c1b32; los archivos adicionales de dimael mencionados abajo no están integrados.

# P1 → P2: Vía B — Walmart M5, tres tiendas y un inventario regional

**Decisión del equipo:** Vía B. **Preparación:** 15 de noviembre de 2026 (Perú).
**Estado:** carga M5, entorno retail y baselines H0/T0 implementados. Falta la comparación profunda, PBRS/Dyna y la ejecución M5; no hay resultados reales. Ver src/retail/README.md.
**Fuente de requisitos:** `enunciado_proyecto_p2_alumno.pdf`, pp. 1–5. Sustentación: **20 de noviembre de 2026**, en horario de clase, con entrega previa en Canvas; verificar allí la hora límite.

Este documento sustituye las referencias antiguas a P2 como continuación obligatoria de RoboCup. Conservamos P1 como evidencia y reutilizamos sus métodos; las políticas de fútbol no se transfieren a precios e inventario. Las decisiones de implementación de abajo son propuestas de trabajo, no requisitos adicionales del docente ni resultados ya obtenidos.

## 1. Punto de partida y transición

La revisión local encuentra cuatro tareas P1 implementadas: persecución, tiro, conducción y pase 2v1; Q-Learning, SARSA y dos variantes de MC; evaluación por semillas, artefactos y verificación en `rcssserver`. Referencia local al preparar esta transición: rama `dimael`, ref `b2fe002f64b65b0cffc2290eeb050a493087dd2c`. No se pudo ejecutar Git en esta sesión, por lo que esta referencia no certifica un árbol limpio.

| Componente P1 | Uso en P2 |
|---|---|
| `src/agents.py` | Reutilizar Q-Learning tabular con **tablas nuevas** y estados/acciones retail. |
| `src/task_discretizer.py`, `src/exploration.py` | Reutilizar discretización y exploración, verificando imports y semántica de terminales. |
| `src/train.py`, `src/task_train.py` | Referencia del contrato de semillas, evaluación y artefactos; crear un runner retail separado. |
| `src/task_analysis.py` | Reutilizar ideas de agregación; añadir media ±1σ y métricas económicas. No copiar el criterio de CIs no solapados como prueba universal. |
| `src/SPEC.md`, entornos y clientes RoboCup | Evidencia P1; conservar su contrato y sus resultados. |
| `notebooks/01–13`, `informe/informe_p1.tex`, `informe_p1.pdf` | Archivo reproducible de P1; no reescribir como si fueran experimentos comerciales. |
| Docker P1 | Mantener su entrada actual; preparar después un Compose retail independiente sin servidor de fútbol. |

El baseline comparable será **Q-Learning de P1 reentrenado en el mismo entorno retail**. El informe explica la continuidad metodológica; no compara goles/capturas con dólares ni reutiliza las tablas de fútbol. Confirmar con el docente que esta lectura satisface “baseline tabular de P1” al cambiar de dominio.

## 2. Requisitos y evidencia de aceptación

| Requisito del enunciado | Implementación prevista | Evidencia requerida |
|---|---|---|
| Una familia de valor y una de política/Actor-Crítico (p. 2) | DDQN y PPO con GAE; tabular como baseline | Misma información disponible, rejilla de acciones y presupuesto de interacciones; curvas y pérdidas. No es obligatorio implementar todos los algoritmos enumerados. |
| PBRS obligatorio (p. 2) | Potencial de cobertura y balance; término `γΦ(s') − Φ(s)` | Demostración, identidad telescópica y comparación con shaping ingenuo. |
| Model-Based **o** imitación (p. 2) | Modelo híbrido de demanda + Dyna sobre DDQN | Rollouts sintéticos que realmente actualizan al agente, errores del modelo y eficiencia muestral. |
| Escalera y ≥3 semillas (pp. 2–3) | Matriz de §7; semillas finales 0, 1, 2 | Retorno, pérdidas, media ±1σ, retorno final, pasos al umbral y estabilidad. |
| CA_1, CA_2, CA_3; tres agentes (p. 3) | Decisiones locales simultáneas de precio y pedido; almacén regional finito | Conservación del inventario y competencia efectiva por suministro. |
| Último año fuera de muestra (p. 4) | Últimos 365 días de ventas observadas, bloqueados | Manifiesto temporal; ningún ajuste o selección con test. |
| ≥30% menos quiebres y superar beneficio histórico (p. 4) | Comparación con agentes egoístas y referencia histórica reconstruida | Métricas sin shaping; limitación de identificabilidad explícita y consulta al docente. No declarar cumplimiento histórico literal con proxies. |
| Reproducibilidad (p. 4) | Compose/script retail con un comando | Datos verificados, dependencias fijadas, entrenamiento → evaluación → figuras. Pendiente de implementar. |
| Informe y sustentación (pp. 4–5) | IEEE, LaTeX, máximo 8 páginas; 20 minutos y demo | Siete secciones exigidas, limitaciones, resultados y demostración. |

## 3. Datos, alcance y límites de lo observable

Usar `sales_train_evaluation.csv`, `sell_prices.csv` y `calendar.csv`; filtrar exactamente CA_1/CA_2/CA_3. Las ventas son diarias y los precios son semanales por artículo/tienda. Unir ventas a calendario por `d`, y a precios por `(store_id, item_id, wm_yr_wk)`; validar unicidad y no multiplicar filas.

La documentación oficial enumera ventas, calendario y precios, **no inventarios, pedidos, demanda perdida ni costes operativos**. En consecuencia, inferimos que M5 por sí solo no identifica quiebres históricos ni beneficio neto histórico de Walmart. Un cero de ventas no demuestra desabastecimiento. Fuente: [M5 Competitors’ Guide, pp. 4–5](https://storage.googleapis.com/kaggle-forum-message-attachments/772349/15032/M5-Competitors-Guide-Final-10-March-2020.pdf) y [archivos oficiales](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

Reglas para el proyecto:

- Etiquetar por separado **observado**, **estimado** y **supuesto de simulación**. La reducción de quiebres se medirá en el simulador; no representa evidencia causal de despliegue en Walmart.
- La referencia `historical_price_base_stock_proxy` reproduce precios históricos (`multiplicador = 1`) con una regla de pedido y costes declarados. No llamarla “política histórica observada”: M5 no contiene sus pedidos.
- El ingreso observado `Σ precio × ventas` puede describirse, pero no es beneficio neto ni un benchmark equivalente a un entorno con restricciones inventadas.
- Estimar elasticidad sólo con entrenamiento, controlando calendario, estacionalidad y tienda. Es una asociación observacional; si faltan cambios de precio, registrar falta de identificación y analizar un rango de supuestos, sin inventar una estimación.
- No imputar precios con información futura ni interpretar automáticamente los precios ausentes como cero. Definir elegibilidad y disponibilidad antes del entrenamiento.

**Alcance inicial propuesto:** un mismo SKU presente en las tres tiendas, elegido por cobertura/precio disponible/actividad en entrenamiento, no por resultados de test. Esto evita mezclar productos no intercambiables en un único inventario. Después repetir en SKUs adicionales si el tiempo lo permite. Consultar si el docente exige un conjunto mayor; no presentar un piloto de un SKU como evaluación de todo Walmart.

**Split propuesto, para el archivo con d_1…d_1941:**

| Uso | Días inclusivos | Fechas | Regla |
|---|---|---|---|
| Entrenamiento | d_1–d_1211 | 2011-01-29–2014-05-23 | Ajustar demanda, elasticidad, normalización y políticas. |
| Validación | d_1212–d_1576 | 2014-05-24–2015-05-23 | Seleccionar hiperparámetros, potencial, presupuesto y checkpoints. |
| Test bloqueado | d_1577–d_1941 | 2015-05-24–2016-05-22 | Un recorrido anual por política/semilla congelada; ninguna selección posterior. |

Verificar fechas contra el CSV antes de congelar el manifiesto; no usar días futuros de `calendar.csv` sin ventas como test. No sustituir silenciosamente el archivo de 1,941 días por el de 1,913. “Último año” se interpreta aquí como los últimos 365 días disponibles, pendiente de aclaración si se exige año calendario.

Las observaciones del día t contienen sólo información disponible antes de actuar: calendario conocido, inventario, llegadas programadas y ventas/precios pasados. Las ventas históricas de t pueden ser una entrada **oculta del simulador**, nunca una característica de la política ni del modelo predictivo antes de actuar. Los rezagos durante un rollout proceden de la historia generada en ese rollout. Los parámetros quedan congelados en test; actualizar el estado con observaciones ya transcurridas no equivale a reentrenar.

## 4. Contrato del entorno retail propuesto

**Sistema:** juego cooperativo de tres agentes con transición conjunta; un paso = un día. El estado del simulador incluye fecha/posición de episodio, inventarios de tiendas y centro, pedidos en tránsito, historial necesario y contexto exógeno. La observación local es una proyección de ese estado, por lo que no se asumirá Markovianidad perfecta de un vector pequeño de características.

**Información:** cada agente ve su inventario, cobertura prevista, historial, calendario y un resumen público del centro y cobertura regional. Las observaciones son numéricas continuas; los tabulares las discretizan. No dar futuro o información privilegiada sólo a uno de los algoritmos.

**Acciones locales:** `(m_i,t, u_i,t)`, multiplicador adimensional de precio y solicitud de unidades. Punto de partida: `m ∈ {0.90, 0.95, 1.00, 1.05, 1.10}` y pedidos objetivo de `{0, 3, 7}` días de cobertura, **15 acciones por tienda**. El pedido se traduce a unidades con el pronóstico disponible, descontando inventario en posición (incluido tránsito). DDQN y PPO usan esta misma rejilla inicialmente; no confundir estado continuo con obligación de acción continua. Rango y tamaños son supuestos piloto, pendientes de calibración en train/validación.

**Orden diario fijado:**

1. Recibir entregas en tránsito y suministro externo al centro, respetando capacidades.
2. Construir observaciones sin demanda del día y recoger simultáneamente las tres acciones.
3. Racionar solicitudes contra stock disponible del centro: proporcional si la suma excede el stock, con desempate determinista para unidades enteras. Descontar despachos del centro y ponerlos en tránsito (lead time inicial: un día).
4. Realizar demanda y ventas de cada tienda; los despachos de hoy no pueden venderse antes de llegar.
5. Calcular inventario final, demanda perdida, recompensa y observación siguiente.

El centro recibe suministro externo limitado y tiene almacenamiento finito; fijar estas dos cantidades por separado. Calibrar sólo con entrenamiento para que haya escasez verificable sin un escenario trivialmente imposible. Prohibir stock negativo, ventas superiores a existencias y creación/destrucción silenciosa de unidades.

**Demanda de referencia:** punto de partida semisintético `d_i,t(m) = y_i,t · m^(−ε_i)` (ε_i ≥ 0), con una regla registrada de redondeo/ruido. `y` son ventas históricas, utilizadas como proxy imperfecto de demanda de referencia. La regla reproduce `y` a precio histórico en el caso determinista; no recupera demanda censurada ni efectos causales. No añadir sustitución entre tiendas sin un modelo explícito: la interacción exigida se produce por el inventario compartido.

**Recompensa base, separada del shaping:** sea `P_i,t = precio_histórico_i,t × m_i,t`, `A_i,t` el stock disponible para venta, `q_i,t = min(d_i,t, A_i,t)`, `L_i,t = max(0, d_i,t − A_i,t)` y `I_i,t+1 = A_i,t − q_i,t`:

`R_base,t = Σ_i [P_i,t q_i,t − c_almacén I_i,t+1 − c_quiebre L_i,t]`.

Esto operacionaliza el objetivo del enunciado y corrige la diferencia de unidades entre multiplicador y precio monetario. Es un **proxy operativo simulado**; no incluye costes de compra reales desconocidos. Registrar los coeficientes y hacer sensibilidad. Si se incorpora coste de compra, mantener también el objetivo del enunciado como métrica separada y documentar el cambio. Los tres agentes cooperativos reciben la recompensa regional; el control egoísta recibe sólo la contribución local bajo iguales restricciones.

## 5. PBRS y experimento de reward hacking

Con pronóstico causal `μ̂_i(s)` y cobertura limitada `z_i(s) = clip(I_i / max(μ̂_i(s), δ), 0, z_max)`, proponer:

`Φ(s) = −λ_balance Var_i[z_i(s)] − λ_buffer Σ_i max(0, z_objetivo − z_i(s))²`.

La segunda parte impide que “las tres tiendas vacías” sea un buen estado sólo por estar balanceadas. Congelar pronosticador, escalas y λ durante cada comparación; Φ depende del estado, no de recompensas futuras ni de test. Usar `F_t = γΦ(s_t+1) − Φ(s_t)` y `R_train = R_base + F_t`. Reportar beneficio y quiebres siempre sin F.

Para un horizonte H, la suma descontada telescopa:

`Σ_t=0…H−1 γ^t F_t = −Φ(s_0) + γ^H Φ(s_H)`.

Con estado aumentado por tiempo y **Φ(s_H)=0 en terminales verdaderos**, el segundo término desaparece y `Q'π(s,a)=Qπ(s,a)−Φ(s)` para toda política. Como la resta no depende de a, preserva las acciones óptimas y `Q'* = Q*−Φ`. Para horizonte infinito, basta potencial acotado y γ<1. Un corte de rollout para PPO no es un terminal: conservar bootstrap y potencial; un fin de episodio definido por el problema sí lo es. Compartir exactamente γ entre agente, shaping y evaluación descontada. La garantía es sobre el objetivo descontado definido, no sobre cualquier indicador económico alternativo, ni garantiza convergencia de learners independientes.

**Prueba requerida:** enfrentar `+β × cobertura balanceada` por día (bono ingenuo) a PBRS en un caso pequeño enumerable donde acaparar/no vender o elevar precios preserva el bono y empeora el retorno base. Comprobar el orden de políticas por retorno base, ingenuo y PBRS, incluyendo la corrección terminal. Añadir el mismo control ingenuo a la matriz retail. No asegurar que PBRS impide todo reward hacking: preserva el objetivo base bajo sus supuestos; no corrige una recompensa base mal especificada ni un simulador explotable.

## 6. World model: híbrido + Dyna

La propuesta prioriza **Dyna sobre DDQN**: aprender demanda/reacción al precio y mantener inventarios, pedidos, asignación y costes como reglas exactas. Un predictor de ventas sin rollouts que actualicen la política no satisface por sí solo el módulo de aceleración.

- Modelo entrenado sólo con información de train y transiciones del entorno de entrenamiento. Guardar procedencia; las transiciones sintéticas no se reciclan como evidencia real para ajustar el propio modelo.
- Muestrear un estado conjunto real de replay, generar las tres acciones, predecir demanda y aplicar una sola transición regional coherente. Recalcular recompensa y PBRS con esa transición; nunca inventar asignaciones independientes que gasten el mismo stock tres veces.
- Empezar con rollouts cortos (1–3 días); horizonte, razón sintético/real y control de incertidumbre se seleccionan en validación. Registrar pérdidas, error multistep y discrepancias de beneficio/quiebres.
- Mantener separado el simulador de evaluación basado en las series retenidas. No evaluar a un agente sólo dentro del mismo modelo aprendido que usa para imaginar.
- Entrenar tres DDQN locales con recompensa compartida y actualizaciones sincronizadas; documentar no estacionariedad multiagente. PPO usa tres actores locales con recompensa compartida, clipping ε=0.2 y GAE como comparación de familia. Son diseños por implementar, no wrappers multiagente ya disponibles en SB3.
- No introducir replay arbitrario en PPO y llamarlo on-policy. La aceleración principal se mide en DDQN; una variante PPO con modelo requeriría otro diseño explícito.
- Dreamer/RSSM queda como extensión posterior. Para este plazo, Dyna permite estudiar modelos del mundo sin añadir aprendizaje latente innecesario a datos tabulares.

## 7. Matriz, métricas y artefactos

| ID | Condición | Recompensa de entrenamiento | Propósito |
|---|---|---|---|
| H0 | Precio histórico + regla base-stock | No aprende | Referencia histórica reconstruida, etiquetada como proxy. |
| T0 | Q-Learning tabular cooperativo | Regional base | Continuidad metodológica con P1. |
| V0 | DDQN cooperativo | Regional base | Familia de valor. |
| P0 | PPO + GAE cooperativo | Regional base | Familia de política/Actor-Crítico. |
| S0 | DDQN egoísta | Local base | Control de coordinación; resto comparable con V0. |
| V1 | DDQN + PBRS | Regional + potencial | Efecto de shaping. |
| V2 | DDQN + Dyna | Regional base | Efecto del modelo por separado. |
| V3 | DDQN + PBRS + Dyna | Regional + potencial | Agente final propuesto. |
| N0 | DDQN + shaping ingenuo | Regional + bono ingenuo | Contraste de reward hacking. |

Ocho condiciones entrenables × tres semillas = **24 corridas finales**, más H0 y pilotos. Semillas 3–5 para desarrollo; 0–2 para reporte. Los pilotos de una semilla sólo verifican ejecución. Ajustar presupuesto a partir de tiempos medidos, sin eliminar requisitos para esconder resultados negativos.

Compartir fechas, estados iniciales, suministro y perturbaciones aleatorias indexadas por día/tienda, no por el número de llamadas de una política. Separar RNG de entrenamiento, entorno, evaluación y modelo. Guardar fallos, no descartar silenciosamente semillas.

**Métricas predefinidas:**

- Curvas de retorno descontado `G0` y retorno base no descontado, junto con pérdidas propias de cada algoritmo; agregar por semilla en el mismo eje de interacciones reales con el simulador. Graficar media ±1σ (desviación muestral, ddof=1); nunca comparar escalas de pérdidas distintas como calidad.
- Beneficio operativo base total, ingreso, costes supuestos, inventario, precio, demanda y ventas por tienda; retorno final y dispersión entre semillas.
- Quiebres primarios: proporción de tienda-días elegibles con demanda perdida positiva. Secundarios: unidades perdidas y fill rate; registrar demanda total porque subir precios puede reducir quiebres simplemente suprimiendo demanda.
- Reducción frente a **cada** referencia: `1 − stockout_rate_agente / stockout_rate_referencia`. Umbral ≥0.30; si la referencia tiene cero quiebres, resultado porcentual indefinido, no éxito automático. Comparar además beneficio base con H0. Sin datos históricos adicionales, estos resultados no satisfacen literalmente una afirmación sobre el beneficio neto real de Walmart.
- Eficiencia: pasos reales hasta `0.8 × G_ref` si `G_ref > 0`, con `G_ref` empírico fijado en validación antes del reporte final y claramente distinto de un óptimo demostrado. Si es no positivo, reportar además un umbral de mejora normalizada `G_H0 + 0.8(G_ref−G_H0)` y explicar por qué la lectura literal es problemática; no cambiar de definición después de ver test. “No alcanzado” permanece censurado al presupuesto. Estas curvas se evalúan en validación, no consultando test repetidamente.
- Estabilidad interepisódica: desviación de retornos base de una cola de episodios de evaluación de igual longitud, definida antes de correr; separarla de variación entre semillas y de la estacionalidad del año de test.
- Contabilizar por separado pasos reales **del simulador**, pasos imaginados, actualizaciones, tiempo y coste del ajuste del modelo. “Real” aquí no significa interacción con tiendas físicas.

Artefactos P2 previstos en `notebooks/artifacts/p2_via_b/<condicion>/seed_<k>/`: configuración resuelta, manifiesto de datos/split, hashes de código y entradas, checkpoints, historial, evaluación por día/tienda y resumen. No mezclar esquemas P1/P2 ni sobrescribir corridas. Fijar presupuesto, hiperparámetros, manejo de terminales, costes, SKU y reglas del simulador antes de abrir test.

## 8. Implementación por hitos y calendario objetivo

| Hito | Objetivo | Condición de salida |
|---|---|---|
| M0 — 15/16 nov. | Resolver datos y contrato | CSV accesibles, SKU y split auditados; dudas al docente preparadas; supuestos explícitos. |
| M1 — 16 nov. | Entorno y H0/tabular | Tests de conservación, lead time, racionamiento simultáneo, determinismo, no leakage y límites; rollout trazable de tres tiendas. |
| M2 — 17 nov. | DDQN y PPO | Smoke de entrenamiento/evaluación; observaciones y acciones equivalentes; medición de coste. |
| M3 — 18 nov. | PBRS, hacking y Dyna | Identidad telescópica; ejemplo de hacking; rollouts coherentes y error del modelo. |
| M4 — 19 nov. | Matriz final y reporte | ≥3 semillas, test congelado, curvas, tabla de requisitos y discusión honesta de fallos. |
| M5 — antes de clase, 20 nov. | Entrega y demo | Comando reproducible en entorno limpio, IEEE ≤8 páginas, vídeo/demo y sustentación de 20 minutos. |

Fechas objetivo, no garantía de tiempo de cómputo. Mantener el alcance pequeño y medir costes en M2; no se promete alcanzar el 30% antes de ejecutar los experimentos.

**Próximo paso concreto:** auditoría de datos M5 y entorno determinista de tres tiendas con H0. Es la dependencia de todos los learners; empezar por redes antes de validar el entorno haría inconcluyentes las comparaciones.

**Consultas puntuales pendientes al docente:** (1) referencia histórica reconstruida ante ausencia de inventarios/costes; (2) baseline P1 reentrenado en retail; (3) alcance de SKU(s); (4) últimos 365 días frente a año calendario. La preparación e implementación verificable pueden avanzar bajo estos supuestos, pero no afirmar aprobación ni cumplimiento de los puntos ambiguos.

## 9. Estado al cerrar esta preparación

- [x] Leer el nuevo enunciado completo y registrar Vía B.
- [x] Revisar la estructura y contratos P1; separar reutilización de cambio de dominio.
- [x] Definir propuesta metodológica, ablations, límites de datos y hitos.
- [x] Añadir plan estructurado en `src/retail/experiment_plan.json` y actualizar entradas de documentación.
- [x] Preparar Docker/Compose P2, script de setup y diagnóstico de dependencias/cabeceras; 9 tests del diagnóstico pasaron en Python local.
- [ ] Ejecutar setup en Ubuntu: construcción, tests P1, smoke RL y Jupyter pendientes por bloqueo de acceso a WSL en la sesión de preparación.
- [ ] Auditar CSV, concretar costes/capacidades/elasticidad/SKU y congelar configuración.
- [ ] Implementar entorno, agentes, PBRS, modelo y runner retail.
- [ ] Ejecutar experimentos, verificar Docker retail, producir informe y demo.

No hay métricas P2 obtenidas en esta preparación. Los checks anteriores describen trabajo de transición, no la entrega completa del proyecto.
