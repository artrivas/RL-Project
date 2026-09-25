# Proyecto RL — Persecución e Intercepción de Balón (RoboCup 2D)

**Curso:** DS5345 · Aprendizaje por Refuerzo (UTEC, 2026-II) · **Docente:** Percy W. Lovon Ramos
**Entrega:** P1, baseline tabular. **Tarea:** *Ball Pursuit*. **Algoritmo:** Q-Learning tabular, con ablación
ε constante vs. ε decreciente geométrico.

El agente aprende a orientarse y acelerar hacia un balón situado a d₀ ∈ [5, 40] m y capturarlo
(d ≤ 0.8 m) con 4 macro-acciones: `DASH 100`, `DASH 50`, `TURN +35`, `TURN −35`. Se entrena en un
simulador rápido construido con los parámetros físicos reales del servidor, y se valida conectándolo
a `rcssserver`.

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
├── src/
│   ├── SPEC.md               # contrato observación/acción, MDP, decisiones y registro de evidencia
│   ├── client.py             # cliente UDP del jugador (cliente)
│   ├── trainer.py            # entrenador / coach offline: resets y verdad de terreno
│   ├── perception.py         # estimador del balón (solo mensajes del jugador)
│   ├── discretizer.py        # discretización del estado (discretizador)
│   ├── agents.py             # Q-Learning tabular (agente)
│   ├── exploration.py        # ε constante y ε decreciente geométrico
│   ├── env_base.py           # lógica de episodio común: recompensa, captura, truncamiento
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
│   ├── plots.py              # figuras de los notebooks
│   └── tests/                # pruebas offline (incluye un servidor falso de punta a punta)
└── notebooks/
    ├── 01_mdp_formulation.ipynb    # tupla ⟨S, A, P, R, γ⟩, discretización, P̂ empírica, γ, presupuesto temporal
    ├── 02_train_baseline.ipynb     # selección del estado, ablación, curvas, π̂(s), V̂(s), trayectorias durante el aprendizaje
    ├── 03_live_validation.ipynb    # conexión, estabilidad, simulador vs. servidor, política en vivo, aprendizaje en vivo
    ├── 04_algorithm_selection.ipynb  # interno: Q-Learning vs. SARSA vs. MC y representaciones más finas
    ├── 05_sensitivity_analysis.ipynb # misma política bajo otros supuestos de evaluación (d₀, ciclos, información)
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
| Comparación interna de algoritmos | `python -m src.train --preset algorithms --out notebooks/artifacts/runs_algorithms` | 3 min |
| Estimaciones de factibilidad (controlador / cota demostrable sin ruido) | `python -m src.feasibility --controller --params notebooks/artifacts/server_params.json` y `... --ceiling` | 30 s |
| Estabilidad y controlador en vivo (500 episodios) | `python -m src.live_eval --policy greedy --episodes 500` | 30 min |
| Política aprendida en vivo (200 episodios) | `python -m src.live_eval --policy notebooks/artifacts/runs/qlearning_eps_decay_1.0_to_0.1/seed_4 --episodes 200` | 12 min |
| Secuencias fijas: simulador vs. servidor | `python -m src.sim_vs_live --repeats 5` | 1 min |
| Notebooks | abrir en Jupyter Lab y ejecutar; o bien `jupyter nbconvert --execute --inplace --to notebook notebooks/0*.ipynb` | 1 min |

Cada corrida de entrenamiento guarda en `notebooks/artifacts/runs/<condición>/seed_<k>/`:
- `q.npy`;
- `history.npz`, con retorno, pasos, captura y ε por episodio;
- `eval.json`, con las evaluaciones greedy periódicas;
- `config.json`, con bins, acciones, hiperparámetros, semilla, parámetros físicos, huella SHA-256 del código
  de `src/` (`source_sha256`) y commit de git si hay repositorio (dentro del contenedor es `null`, porque
  `.git` no se monta). Las corridas incluidas en `notebooks/artifacts/` se generaron antes de registrar la
  huella; con las mismas semillas se reproducen exactamente (verificado para Q-Learning).

Los notebooks cargan esos artefactos; `RETRAIN = True` / `RUN_LIVE_EVAL = True` los regeneran.

---

## 4. Resultados principales

Criterio de éxito: captura en < 40 pasos (1 paso = 1 ciclo de servidor de 100 ms). También se reporta ≤ 40.

**Simulador.** Evaluación greedy sobre 500 inicios fijos; media ± desviación estándar en 5 semillas:

| condición | captura < 40 |
|---|---|
| Q-Learning, ε decreciente 1.0 → 0.1 | **73.8 % ± 0.6** |
| Q-Learning, ε constante 0.1 | 70.0 % ± 3.7 |
| Controlador heurístico (mismos inicios) | 71.6 % |

**Servidor real** (`rcssserver`). Mismos inicios en ambos entornos; error estándar entre paréntesis:

| política | servidor < 40 | simulador < 40 | servidor ≤ 40 |
|---|---|---|---|
| Q-Learning, ε decreciente (200 episodios) | **73.5 % (± 3.1)** | 71.5 % | 76.5 % |
| Controlador heurístico (500 episodios) | 67.2 % (± 2.1) | 67.6 % | 69.8 % |

- La política aprendida captura ≥ 95 % de los balones que parten a menos de 30 m.
- En 500 episodios consecutivos en vivo hubo 0 comandos perdidos, 0 errores de reset y 0 cambios de modo.
- Durante el entrenamiento, la política greedy pasa de 0–1/4 capturas (0–1 000 episodios) a 4/4 desde 5 000
  episodios sobre 4 inicios fijos, tanto en el servidor como en el simulador.
- En el simulador, los algoritmos (Q-Learning, SARSA, MC) y las representaciones probadas se estancan cerca de
  75 % (mejor evaluación individual 74–75 %), con distinto promedio y estabilidad. En vivo, los fallos se
  concentran a más de 30 m: 97.8 % de captura bajo 30 m, 33 % entre 30 y 35 m, 0 % sobre 35 m. Estos datos no
  determinan la causa (tiempo, percepción, acciones, representación o aprendizaje) ni establecen un techo
  universal.
- La física del simulador coincide con la del servidor: sesgo de posición ≤ 0.16 m, menor que el ruido
  propio del servidor.

**Sobre el criterio de > 90 %:** con nuestra interpretación de un paso = un ciclo del servidor (100 ms) y
nuestra distribución de inicios, el presupuesto de < 40 pasos es muy restrictivo. La distribución es d₀ uniforme
en [5, 40] m: el enunciado fija el rango, no la distribución. La política aprendida logra 73.5 % en el servidor,
con los fallos concentrados en los inicios lejanos (≈ 98 % bajo 30 m, 0 % sobre 35 m).

Una estimación relajada de girar y luego avanzar alcanza ≈ 89 %, lo que sugiere un problema de factibilidad,
pero **no demuestra que superar 90 % sea imposible**. La cota demostrable sin ruido es ≈ 95 %, y con ruido no hay
imposibilidad estricta. Está pendiente aclarar con el docente la duración de un paso y la distribución de
inicios prevista. Detalles en `src/SPEC.md` §11 y `notebooks/01_mdp_formulation.ipynb` §7.

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
