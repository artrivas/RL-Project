# Proyecto RL — Tareas acotadas de RoboCup 2D con métodos tabulares

**Curso:** DS5345 · Aprendizaje por Refuerzo (UTEC, 2026-II) · **Docente:** Percy W. Lovon Ramos
**Entrega:** P1, baseline tabular. **Tareas:** las cuatro del catálogo — persecución e intercepción (*Ball
Pursuit*), tiro a puerta, conducción y cooperación 2v1 — con la cinemática del *starter kit* (desplazamientos
exactos, sin física, observación completa). **Algoritmos:** Q-Learning, SARSA y Monte Carlo first-visit (promedios
y α constante), cada uno con ε constante vs. ε decreciente geométrico, 5 semillas. `rcssserver` se usa para
verificar la ejecución de cada política con el cliente.

La persecución fue la primera tarea (notebooks 01–07, también con la física real del servidor). Las otras tres y la
comparación de métodos entre tareas están en los notebooks 10–13; su plan y sus decisiones, en
`informe/plan_otras_tareas.md`.

---

## 1. Puesta en marcha

Requisito: Docker con Compose v2.20 o superior (Docker Desktop en macOS/Windows, o Docker Engine en Linux/WSL2).

```bash
docker compose up -d
```

Levanta dos contenedores, definidos en `docker/docker-compose.yml`:

| contenedor | rol |
|---|---|
| `robocup_server` | `rcssserver`, controlado por entrenador (`coach=true`, `coach_w_referee=false`, `auto_mode=false`). Jugador en UDP 6000, entrenador en 6001. |
| `robocup_rl_agent` | Python 3.10, PyTorch, Gymnasium y Jupyter Lab en **http://localhost:8888** (sin token). Monta `src/` y `notebooks/`. |

El contenedor del agente corre con el UID/GID del usuario (1000 por defecto), así que los archivos
generados quedan editables. Si tu usuario tiene otro UID, usa
`UID=$(id -u) GID=$(id -g) docker compose up -d`.

**Verificar la conexión con el servidor** (handshake, reloj, reset, signo del giro, dash vs. modelo):

```bash
docker compose exec rl-agent python -m src.live_check
```

Todas las líneas deben decir `PASS`. El script también guarda los parámetros físicos reales del servidor
en `notebooks/artifacts/server_params.json`, y el simulador los usa.

---

## 2. Estructura

```text
├── docker/                   # Dockerfile, docker-compose.yml, requirements.txt
├── docker-compose.yml        # punto de entrada en la raíz (incluye docker/docker-compose.yml)
├── informe/informe_p1.tex    # fuente LaTeX del informe (formato de plantilla_informe_latex/)
├── informe_p1.pdf            # informe compilado
├── src/
│   ├── SPEC.md               # contrato observación/acción, MDP, decisiones y registro de evidencia
│   ├── client.py             # cliente UDP del jugador (cliente)
│   ├── trainer.py            # entrenador / coach offline: resets y verdad de terreno
│   ├── perception.py         # estimador del balón (solo mensajes del jugador)
│   ├── discretizer.py        # discretización del estado (discretizador)
│   ├── agents.py             # Q-Learning tabular (agente)
│   ├── exploration.py        # ε constante y ε decreciente geométrico
│   ├── env_base.py           # lógica de episodio común: recompensa, captura, truncamiento
│   ├── kit_env.py            # entorno cinemático del starter kit (entorno principal de P1)
│   ├── kit_optimal.py        # óptimo exacto en el kit (búsqueda A*): máximo alcanzable
│   ├── sim_env.py            # simulador rápido con la física de rcssserver
│   ├── live_env.py           # mismo entorno sobre el servidor real
│   ├── sampler.py            # distribución de inicios compartida
│   ├── controllers.py        # controlador heurístico de referencia y rollout
│   ├── train.py              # entrenamiento, evaluación y guardado de corridas
│   ├── live_eval.py          # evaluación en vivo y estabilidad de largo plazo
│   ├── sim_vs_live.py        # secuencias fijas: simulador vs. servidor
│   ├── live_snapshots.py     # políticas guardadas durante el entrenamiento, ejecutadas en vivo
│   ├── sensitivity.py        # sensibilidad de una política fija a los supuestos de evaluación
│   ├── feasibility.py        # estimaciones de factibilidad del presupuesto de 40 pasos
│   ├── live_check.py         # verificación rápida de la conexión
│   ├── task_discretizer.py   # discretizador genérico por producto de variables (tareas nuevas)
│   ├── task_train.py         # entrenamiento genérico: registro de tareas, matriz método × exploración, barrido de α, diagnósticos
│   ├── task_analysis.py      # resúmenes con IC bootstrap sobre semillas (JSON)
│   ├── shooting_env.py       # tiro a puerta
│   ├── shooting_optimal.py   # tiro: probabilidades exactas, óptimo verdadero y de la representación, calibración
│   ├── dribbling_env.py      # conducción (y controlador guionado: cota constructiva)
│   ├── passing_env.py        # cooperación 2v1 (y heurística de referencia)
│   ├── live_tasks.py         # verificación en rcssserver de tiro, conducción y 2v1
│   ├── plots.py              # figuras de los notebooks
│   └── tests/                # pruebas offline (incluye un servidor falso de punta a punta)
└── notebooks/
    ├── 01_mdp_formulation.ipynb    # tupla ⟨S, A, P, R, γ⟩, discretización, P̂ empírica, γ, presupuesto temporal
    ├── 02_train_baseline.ipynb     # selección del estado, ablación, curvas, π̂(s), V̂(s), trayectorias durante el aprendizaje
    ├── 03_live_validation.ipynb    # conexión, estabilidad, simulador vs. servidor, política en vivo, aprendizaje en vivo
    ├── 04_algorithm_selection.ipynb  # interno: Q-Learning vs. SARSA vs. MC y representaciones más finas
    ├── 05_sensitivity_analysis.ipynb # misma política bajo otros supuestos de evaluación (d₀, ciclos, información)
    ├── 06_alternative_interpretations.ipynb # lecturas alternativas del criterio, medidas en el servidor
    ├── 07_starter_kit_baseline.ipynb # RESULTADO PRINCIPAL: entorno del kit, ablación y transferencia
    ├── 08_formulacion_otras_tareas.ipynb # formulación inicial de las otras tres tareas (la implementada está en 10–12)
    ├── 10_tiro_a_puerta.ipynb      # tiro: calibración, óptimo exacto, ablación, métodos, diagnósticos, servidor
    ├── 11_conduccion.ipynb         # conducción: cota guionada, ablación, oscilación, métodos, diagnósticos, servidor
    ├── 12_pase_2v1.ipynb           # 2v1: referencias, ablación, métodos, diagnósticos, servidor
    ├── 13_comparacion_algoritmos.ipynb # los cuatro métodos en las cuatro tareas
    └── artifacts/                  # corridas guardadas (Q-tables, historiales, configs), figuras, JSON
```

---

## 3. Reproducir los experimentos

Todos los comandos corren dentro del contenedor (`docker compose exec rl-agent ...`) desde `/workspace`.

| paso | comando | duración aprox. |
|---|---|---|
| Pruebas offline | `python -m pytest -q src/tests` | 5 s |
| Comparación de representaciones de estado | `python -m src.train --preset discretization --out notebooks/artifacts/runs_discretization` | 3 min |
| Ídem con α = 0.03 (candidatos B y C) | `python -m src.train --preset discretization --alpha 0.03 --only disc_B_front17.5 disc_C_front17.5_speed --out notebooks/artifacts/runs_discretization` | 2 min |
| **Ablación de exploración** (5 semillas × 20k episodios) | `python -m src.train --preset ablation --out notebooks/artifacts/runs` | 2 min |
| Instantáneas de aprendizaje (corrida en vivo, semilla 4) | `python -m src.train --preset ablation --only qlearning_eps_decay_1.0_to_0.1 --seeds 4 --snapshots 0 500 1000 2000 5000 10000 20000 --out notebooks/artifacts/runs` | 1 min |
| Instantáneas ejecutadas en el servidor | `python -m src.live_snapshots --run notebooks/artifacts/runs/qlearning_eps_decay_1.0_to_0.1/seed_4` | 2 min |
| Representaciones más finas (comparación interna) | `python -m src.train --preset refinement --out notebooks/artifacts/runs_refinement` | 6 min |
| Análisis de sensibilidad (misma política, otros supuestos) | `python -m src.sensitivity --n 5000` | 1 min |
| Lectura alternativa: 1 paso = k ciclos (entrenamiento) | `python -m src.train --preset macro --out notebooks/artifacts/runs_macro` | 3 min |
| Lecturas alternativas en vivo | `python -m src.live_eval --policy <corrida> --episodes 200 [--distance-dist near]` | 12–20 min c/u |
| Compilar el informe | `cd informe && tectonic informe_p1.tex` (o pdfLaTeX/Overleaf) | 1 min |
| Kit: comparación inicial de discretizaciones (20 estados) | `python -m src.train --preset kit_discretization --out notebooks/artifacts/runs_kit_discretization` | 1 min |
| Kit: representaciones más finas | `python -m src.train --preset kit_refinement --out notebooks/artifacts/runs_kit_refinement`, luego `--preset kit_refinement2 --episodes 80000` (misma carpeta) | 5 min |
| Kit: óptimo exacto (A*) | `python -m src.kit_optimal --n 1000` | 1 min |
| **Kit: ablación final (R3)** | `python -m src.train --preset kit_ablation --episodes 80000 --snapshots 0 1000 2000 5000 10000 20000 40000 80000 --out notebooks/artifacts/runs_kit_final` | 1 min |
| Kit: selección de algoritmo (R3) | `python -m src.train --preset kit_algorithms --episodes 80000 --out notebooks/artifacts/runs_kit_final_algorithms` | 4 min |
| Kit: transferencia a rcssserver | `python -m src.live_eval --policy notebooks/artifacts/runs_kit_final/kit_qlearning_eps_decay_1.0_to_0.1/seed_1 --episodes 200 [--distance-dist kit]` | 12 min c/u |
| Comparación interna de algoritmos | `python -m src.train --preset algorithms --out notebooks/artifacts/runs_algorithms` | 3 min |
| Estimaciones de factibilidad (controlador / cota demostrable sin ruido) | `python -m src.feasibility --controller --params notebooks/artifacts/server_params.json` y `... --ceiling` | 30 s |
| Estabilidad y controlador en vivo (500 episodios) | `python -m src.live_eval --policy greedy --episodes 500` | 30 min |
| Política aprendida en vivo (200 episodios) | `python -m src.live_eval --policy notebooks/artifacts/runs/qlearning_eps_decay_1.0_to_0.1/seed_4 --episodes 200` | 12 min |
| Secuencias fijas: simulador vs. servidor | `python -m src.sim_vs_live --repeats 5` | 1 min |
| Notebooks | abrir en Jupyter Lab y ejecutar; o bien `jupyter nbconvert --execute --inplace --to notebook notebooks/0*.ipynb` | 1 min |

**Tareas nuevas** (`src/task_train.py`; α se elige en semillas 5–9 y se reporta en 0–4). Usa como mucho
~8 procesos en total: en una laptop de 8 núcleos, lanzar más solo hace cada corrida más lenta.

| paso | comando | duración aprox. |
|---|---|---|
| Tiro: calibración y óptimo exacto | `python -m src.shooting_optimal --calibrate notebooks/artifacts/shooting_calibration.json` y `python -m src.shooting_optimal --p-keeper-out 0.02 --out notebooks/artifacts/shooting_optimal.json` | 3 min |
| Tiro: barrido de α (solo ε decreciente) | `python -m src.task_train --task shooting --preset alpha_sweep --decay-only-sweep --episodes 50000 --eval-every 2000 --out notebooks/artifacts/runs_shooting_sweep`; luego `pick_alphas` → `shooting_alphas.json` (ver notebook 10) | 1 min |
| Tiro: matriz y diagnósticos | `python -m src.task_train --task shooting --preset matrix --alphas notebooks/artifacts/shooting_alphas.json --episodes 50000 --eval-every 2000 --snapshots 0 500 2000 10000 50000 --out notebooks/artifacts/runs_shooting`; `--preset diagnostics --out notebooks/artifacts/runs_shooting_diag` | 2 min |
| Conducción: barrido, matriz, diagnósticos | `--task dribbling --preset alpha_sweep --episodes 60000 --eval-every 2000 --out notebooks/artifacts/runs_dribbling_sweep`; `--preset matrix --alphas notebooks/artifacts/dribbling_alphas.json --episodes 60000 --eval-every 2000 --snapshots 0 2000 10000 30000 60000 --out notebooks/artifacts/runs_dribbling`; `--preset diagnostics --alphas ... --episodes 60000 --out notebooks/artifacts/runs_dribbling_diag` | ≈ 25 min con 5 procesos |
| 2v1: barrido, matriz, diagnósticos | igual que conducción con `--task passing` y carpetas `runs_passing*` | ≈ 30 min con 5 procesos |
| Resúmenes con IC | `python -m src.task_analysis --runs notebooks/artifacts/runs_<tarea> notebooks/artifacts/runs_<tarea>_diag --out notebooks/artifacts/<tarea>_summary.json` (el tiro añade `--optimal notebooks/artifacts/shooting_optimal.json`) | segundos |
| Verificación en el servidor | `docker compose restart rcssserver`, luego `python -m src.live_tasks shooting --episodes 20 --out notebooks/artifacts/live_shooting.json` (ídem `dribbling`/`passing` con `--episodes 10`), **reiniciando el servidor antes de cada una** | 1 min c/u |
| Notebooks 10–13 | `jupyter nbconvert --execute --inplace --to notebook notebooks/1[0-3]_*.ipynb` (leen los artefactos) | 1 min |

`python -m src.task_train` acepta `--task {pursuit_kit,shooting,dribbling,passing}`; `pursuit_kit` reproduce
exactamente las corridas de `train.py` en el entorno del kit (prueba de regresión).

Cada corrida de entrenamiento guarda en `notebooks/artifacts/runs/<condición>/seed_<k>/`:
- `q.npy`;
- `history.npz`, con retorno, pasos, captura y ε por episodio;
- `eval.json`, con las evaluaciones greedy periódicas;
- `config.json`, con bins, acciones, hiperparámetros, semilla, parámetros físicos, huella SHA-256 del código
  de `src/` (`source_sha256`) y commit de git si hay repositorio (dentro del contenedor es `null`, porque
  `.git` no se monta). Las corridas incluidas en `notebooks/artifacts/` se generaron antes de registrar la
  huella; con las mismas semillas se reproducen exactamente (verificado para Q-Learning). La huella cubre todos
  los `src/*.py`, así que agregar un módulo cambia la huella de corridas nuevas aunque el código que usan no cambie.

Los notebooks cargan esos artefactos; `RETRAIN = True` / `RUN_LIVE_EVAL = True` los regeneran.

---

## 4. Resultados principales

Criterio de éxito: captura (d ≤ 0.8 m) en menos de 40 pasos. Todo con 5 semillas.

**Entorno del starter kit** (notebook 07). Q-Learning, discretización R3 (99 estados), α = 0.1, 80 000 episodios,
ε decreciente, evaluación greedy:

| inicios | captura < 40 pasos | pasos | máximo alcanzable (óptimo exacto) |
|---|---|---|---|
| reset del kit (d₀ ≈ 11–19 m) | **100 %** (5/5 semillas) | 19.1 | 100 % (18.6 pasos) |
| d₀ ∈ [5, 40] m (rango del enunciado) | **85.7 % ± 0.3** | 26.3 | **87.9 %** |

- **Óptimo exacto** (`python -m src.kit_optimal`): como el kit es determinista, la búsqueda A* da el mínimo de pasos
  para cada inicio. Con d₀ uniforme en [5, 40] m, **ninguna política supera el 87.9 %**; nuestra política alcanza
  el 97.5 % de ese máximo.
- **Ablación:** con ε decreciente se llega al 80 % en 12 000–16 000 episodios, frente a 16 000–32 000 con
  ε = 0.1 constante, y el resultado final es algo más alto y estable (85.7 % ± 0.3 frente a 84.3 % ± 1.3).
- **Discretización:** la del kit da 56.6 %, nuestra versión de 20 estados 61.0 % y R3 85.7 %. No aislamos qué
  componente de R3 explica la mejora; su política avanza en una ventana asimétrica [−35°, +17.5°] que las rejillas
  más gruesas no pueden representar.
- **Fidelidad al kit:** con la configuración del notebook del kit (MC, 3 500 episodios) el éxito varía mucho entre
  semillas: 89.2 % ± 17.2.

**Extensión: física del servidor** (notebooks 01–05). Una política entrenada con la física de `rcssserver`,
solo con la percepción del jugador:

| política | servidor < 40 | simulador < 40 | servidor ≤ 40 |
|---|---|---|---|
| Q-Learning, ε decreciente (200 episodios) | **73.5 % (± 3.1)** | 71.5 % | 76.5 % |
| Controlador heurístico (500 episodios) | 67.2 % (± 2.1) | 67.6 % | 69.8 % |

**Lecturas alternativas del enunciado** (servidor, 200 episodios; notebook 06). El enunciado no fija la
duración del paso ni la forma de la distribución de d₀. Estas cifras dependen de esa lectura; no son el
resultado principal:

| lectura | captura < 40 pasos en el servidor |
|---|---|
| d₀ concentrado en distancias cortas (triangular), misma política | 92.5 % (± 1.9) |
| 1 paso = 2 ciclos (política reentrenada) | 99.5 % (± 0.5) |
| 1 paso = 3 ciclos (política reentrenada) | 96.0 % (± 1.4) |

**Sobre el criterio de > 90 %:** con nuestra interpretación de un paso = un ciclo del servidor (100 ms) y
nuestra distribución de inicios, el presupuesto de < 40 pasos es muy restrictivo. La distribución es d₀ uniforme
en [5, 40] m: el enunciado fija el rango, no la distribución. La política aprendida logra 73.5 % en el servidor,
con los fallos concentrados en los inicios lejanos (≈ 98 % bajo 30 m, 0 % sobre 35 m).

Una estimación relajada de girar y luego avanzar alcanza ≈ 89 %, lo que sugiere un problema de factibilidad,
pero **no demuestra que superar 90 % sea imposible**. La cota demostrable sin ruido es ≈ 95 %, y con ruido no hay
imposibilidad estricta. Está pendiente aclarar con el docente la duración de un paso y la distribución de
inicios prevista. Detalles en `src/SPEC.md` §11 y `notebooks/01_mdp_formulation.ipynb` §7.

---

**Tareas nuevas y comparación de métodos** (notebooks 10–13; media de 5 semillas, IC 95 % bootstrap, Q-Learning
con ε decreciente salvo que se indique):

| tarea | resultado | referencia | criterio |
|---|---|---|---|
| Tiro, sin arquero | 100 % de goles | óptimo 100 % | > 75 % ✔ |
| Tiro, con arquero | 92.3 % [91.8, 92.6] | óptimo exacto 95.2 %; óptimo con 36 estados 93.3 % | > 50 % ✔ |
| Conducción (36 estados) | 97.7 % [96.3, 99.0] (media de las últimas 5 evaluaciones) | controlador guionado 100 % | > 80 % ✔ |
| Cooperación 2v1 (324 estados) | 64.8 % [61.3, 68.1] de episodios con posesión > 50 pasos y ≥ 3 pases | heurística 10.8 % | — (no hay cota) |

- **Ningún método gana en todas las tareas.** MC con promedios aprende antes en el tiro (episodios de un paso) y
  falla en la 2v1; Q-Learning es el mejor en la 2v1 y el más rápido en la conducción; en la persecución los tres
  empatan al final. MC con α constante falla en las tres tareas de episodios largos.
- **Hallazgos con diagnóstico:** en el tiro, la desventaja de ε constante es velocidad (Q₀ = 0 con α pequeño), no la
  solución final; en la conducción, la política greedy oscila por la agregación de estados (con 245 estados, 100 %
  estable); en la 2v1, SARSA no es más prudente que Q-Learning pese al riesgo de quite.
- **Servidor real** (verificación de ejecución, 0 ciclos perdidos): tiro 20/20 y 20/20 goles (error angular medido
  1.75° frente a 4° del modelo); conducción 10/10 (un `KICK 25` real rueda ≈ 13.6 m, no 2 m); 2v1 0/10 (un pase
  real tarda varios ciclos).

---

## 5. Problemas conocidos

- **`rcssserver rejected the player: ... 11 uniform numbers`.** El servidor no reutiliza los dorsales
  después de `(bye)`, así que cada sesión admite 11 conexiones de jugador (`live_check`, `live_eval`,
  notebooks…). Solución:

  ```bash
  docker compose restart rcssserver
  ```
- **`no trainer reply ... coach=true`.** El servidor no se levantó con la configuración de `docker/`.
  Revisa `docker compose ps` y vuelve a ejecutar `docker compose up -d`.
- **No ejecutes dos evaluaciones en vivo a la vez.** Ambas moverían jugadores en la misma cancha.
- **`trainer init rejected: (error already_have_offline_coach)`.** El entrenador offline no se libera con `(bye)`:
  cada sesión del servidor admite un solo entrenador. Reinicia `rcssserver` antes de cada corrida de
  `src.live_tasks`.
