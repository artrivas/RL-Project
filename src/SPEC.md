# Ball Pursuit — Observation/Action Specification (P1)

Single contract for the simulator (`sim_env.py`), the live environment (`live_env.py`), training
code and the report. Status tags: **DECIDED** (implemented, change only deliberately),
**PROPOSED** (default until validated or discussed), **MEASURE** (value comes from the live server;
fill in from `00_environment_validation.ipynb`), **ASK** (pending instructor clarification).

## 1. Frames and sign conventions — DECIDED

- **Field frame:** the server's global frame as reported by the trainer (`look` / `see_global`).
  Origin at the centre spot, `+x` toward the right goal, pitch `x ∈ [-52.5, 52.5]`, `y ∈ [-34, 34]`.
  The server's monitor draws `+y` downward; our plots invert the y-axis to match the monitor.
- **Angles:** degrees, wrapped to `(-180, 180]` (`sampler.wrap_deg`). A positive angle rotates
  from `+x` toward `+y`, i.e. clockwise on the monitor. `atan2(dy, dx)` in this frame gives
  directions consistent with server headings.
- **Body heading** `φ_body`: global direction of the body (trainer `body`).
  **Neck angle** `ν`: head relative to body, `ν ∈ [minneckang, maxneckang]` (`head_angle` in
  `sense_body`, trainer `neck`). **Head direction** = `φ_body + ν`.
- **Vision bearings** (`see`) are relative to the head. Body-relative bearing:
  `θ_body = wrap(θ_seen + ν)`.
- **Turn sign:** `TURN +m` increases `φ_body`, so a ball at positive body-relative bearing moves
  toward `θ = 0` after `TURN +35`.
- **Acceptance test (both environments):** place the ball at body-relative `θ = +20°`, 5 m away,
  player at rest; one `TURN +35` must reduce `|θ|` (overshooting to about `-15°` is expected).

## 2. Actions — DECIDED

| id | command | notes |
|---|---|---|
| 0 | `(dash 100)` | acceleration `100 · dash_power_rate · effort` along the body |
| 1 | `(dash 50)` | half of the above |
| 2 | `(turn 35)` | actual rotation `35 · (1 + ε) / (1 + inertia_moment · speed)`, `ε ~ U[-player_rand, player_rand]` |
| 3 | `(turn -35)` | mirror of id 2 |

Consequence (measured in noiseless kinematics, default params): at rest, reachable headings are
quantized in 35° steps, so the residual heading error after turning in place can be up to 17.5°.
Finer alignment needs turns while moving, which rotate much less (≈ 11.7° at 0.4 m/cycle, ≈ 5.8° at
1.0 m/cycle). Turning effectiveness therefore depends on speed, which is not in the (d, θ) state.

## 3. Step and timing

- **DECIDED — step:** 1 decision step = 1 server cycle (100 ms, `simulator_step`). Exactly one body
  command per cycle. `turn_neck` / `change_view` may be sent in the same cycle as the body command.
  Multi-cycle macro-actions are not used; if ever adopted, they must not be used to satisfy the step
  budget, and results must report decision steps and server cycles.
- **DECIDED — clock:** the cycle counter is the `sense_body` time. A decision for cycle `t` is
  taken after `sense_body(t)` arrives, using every `see` with time `≤ t`. The body command is
  sent immediately. Commands arriving late are detected via the `sense_body` command counters
  (`dash`/`turn` count must increase by exactly one per cycle); a missed cycle is logged, never
  silently ignored.
- **MEASURED (2026-09-24, `live_check`, timing probes):** in `view_mode high normal`, `see` arrives in
  2 of every 3 cycles. Its phase within the cycle depends on the connection: 0 / 50 ms after
  `sense_body` in one session, about 41 / 91 ms in another.
- **DECIDED — decision deadline and late sightings:** after `sense_body(t)` the agent waits up to 60 ms
  for `see(t)`, then decides. A `see(t)` that arrives later is used at cycle `t+1` as a *late sighting*:
  the estimator applies it before the motion update of `t → t+1`. The simulator models the same
  deadline, with a random phase per episode. In a 500-episode live run, no command missed its cycle.

## 4. Observation given to the policy

- **DECIDED — no privileged information:** the policy input comes only from player messages
  (`see`, `sense_body`). Trainer ground truth is used for resets, rewards, metrics and trajectory
  plots only.
- **DECIDED — ball estimator** (`perception.py`): tracks the stationary ball in a
  player-anchored frame by integrating own motion (body heading from turns, displacement from
  `sense_body` speed/direction, including drift while turning). Each `see` containing the ball
  replaces the estimate. Output: body-relative `(d̂, θ̂)` plus a status:
  `SEEN` (in this cycle's `see`), `TRACKED` (estimated since the last sighting), `UNKNOWN`
  (never seen this episode).
- **DECIDED — `(B)` sightings:** the unnamed close-range ball (`(B)`, within `visible_distance`
  outside the view cone) counts as a sighting when it carries a distance.
- **DECIDED — unknown ball:** `UNKNOWN` maps to a dedicated non-terminal state, so the learned
  policy also covers the search (every search cycle counts toward the budget). The scripted
  baseline controller searches by `TURN +35` until the ball is seen.
- **DECIDED — speed feature:** because turn size depends on speed (§2), the state includes one
  speed bit from `sense_body` (edge 0.2 m/cycle; post-decay speed is ≈ 0 at rest and ≈ 0.4 at cruise).
  It mainly stabilizes the learned policy (see §5). The state is still an approximate, not fully Markov,
  representation.
- **Neck:** PROPOSED neck fixed at 0 for the baseline. Neck scanning is an optional experiment and,
  if adopted, must run identically in the simulator and live, with bearings converted per §1.

## 5. Discretization — DECIDED

- Distance bins are half-open intervals `[e_i, e_{i+1})`. Capture (`d ≤ 0.8`) is a terminal state,
  not a bin. Distances beyond the last edge fall into the last bin (unbounded above).
- Angle bins partition `(-180, 180]` after wrapping, with each boundary assigned to exactly one bin.
  The bins are symmetric around 0, and boundaries are assigned toward the frontal bin.
- **Chosen representation "C"** (`DiscretizerConfig` defaults): distance edges `3, 10, 20` m;
  angle edges `17.5, 90, 180`° (front `|θ| ≤ 17.5`, right/left `(17.5, 90]`, back right/left);
  speed edge `0.2` m/cycle. `|S| = 4 × 5 × 2 + 1 (UNKNOWN) = 41` non-terminal states,
  `|S| × |A| = 164` Q-values.
- **Why ±17.5°:** at rest, `TURN ±35` reaches only headings 35° apart, so the residual error after
  turning in place is up to 17.5°. A narrower front bin cannot always be entered by turning at rest,
  and the policy oscillates between turns.
- **Empirical comparison** (`python -m src.train --preset discretization`; Q-learning, ε 1.0 → 0.1,
  20k episodes, 5 seeds; mean of the last 5 greedy evaluations, capture < 40):

  | candidate | α | last-5 mean | worst of last 5 |
  |---|---|---|---|
  | A: front ±10, no speed | 0.1 | 57.4% | 16% |
  | B: front ±17.5, no speed | 0.1 | 69.7% | 19% |
  | C: B + speed bit | 0.1 | 71.4% | 57% |
  | D: C + front split at 5° | 0.1 | 62.3% | 13% |
  | B | 0.03 | 73.2% | 64% |
  | **C** | **0.03** | **73.9%** | **71%** |

  α = 0.1 lets the greedy policy collapse periodically under state aliasing. α = 0.03 is adopted.
- Known wrinkle: `DASH 50`'s post-decay cruise speed is exactly 0.2 m/cycle, the speed edge, so the speed bit
  flickers under sustained `DASH 50`. The learned policy almost never uses `DASH 50`. An edge at 0.15 or
  0.25 would avoid this, but it would invalidate the recorded runs, so it is documented rather than changed.
- The analytical justification goes in `01_mdp_formulation.ipynb`.

## 6. Episode, capture, termination — DECIDED

- **Capture:** true player-centre to ball-centre distance ≤ 0.8 m at the end of a cycle
  (simulator state / trainer `see_global` for that cycle). Capture is terminal.
- **Step cap:** `T_max = 40` decision steps. Reaching it without capture is truncation, not termination.
- **Success:** capture with its step index recorded, so both readings are reported:
  `< 40` (capture at step ≤ 39) and `≤ 40` (capture at step ≤ 40). **ASK** which one the rubric means.
- **TD bootstrap rule (DECIDED, `agents.QLearningAgent`):** terminal transitions do not bootstrap. For truncation, bootstrap
  from `s'`, because elapsed time is not part of the state, so the cap is not a property of the
  state (standard time-limit treatment). MC returns are simply cut at `T_max`.

## 7. Reward — DECIDED (prescribed by the assignment)

`r_t = (d_t − d_{t+1}) − 0.2 + 100 · 1{d_{t+1} ≤ 0.8}`, with `d` the true distance (simulator
state / trainer ground truth), identical in both environments.

With `Φ(s) = −d(s)`, potential-based shaping consistent with discounting would be
`γΦ(s') − Φ(s) = d_t − γ d_{t+1}`. The prescribed term is `d_t − d_{t+1}`, so for `γ < 1` it is not
exactly potential-based. We keep the prescribed reward, document the difference, and do not claim
policy invariance (the state is also only approximately Markov). γ is chosen and justified in
`01_mdp_formulation.ipynb` (used: γ = 0.99).

## 8. Reset distribution — DECIDED (`sampler.py`)

1. `d ~ U[5, 40]`, relative bearing `~ U[-180, 180)`, body heading `~ U[-180, 180)`.
   **ASK:** the brief fixes the range `d_b ∈ [5, 40]` m but not the distribution; uniform is our choice,
   and success rates depend on it (far starts are the hard ones).
2. Ball offset fixed by those. Player position uniform on the feasible rectangle (pitch shrunk by a
   1 m margin, intersected with a copy shifted by the ball offset). This region is never empty, so there is
   no rejection sampling, and the relative marginals stay exactly uniform.
3. Evaluation uses a fixed list `sampler.evaluation_starts(n, seed)` shared by every condition.

## 9. Live reset sequence — DECIDED (`live_env.py`; server config in `docker/docker-compose.yml`)

Server: `coach=true`, `coach_w_referee=false`, `auto_mode=false`, `synch_mode=false`. The play mode
stays `play_on` for the whole session; the trainer never leaves it between episodes.

1. Trainer: `(move (player UTEC_RL 1) x y heading 0 0)`, then `(move (ball) x y 0 0 0)`. Acknowledgements are checked.
2. Player: `turn_neck(-head_angle)` if `head_angle ≠ 0`; confirm `head_angle = 0` in `sense_body`.
3. Trainer: `(recover)`, acknowledgement checked.
4. Clear estimator and observation history. Let `t_r` be the last cycle in which a reset command was
   applied. Observations with time `≤ t_r` are discarded (a `see(t_r)` may predate the move).
5. Verify with `see_global(t_r + 1)` that positions, zero velocities and heading match the sampled
   start within tolerance. Then step 0 is cycle `t_r + 1`.

## 10. Physics parameters — DECIDED

Operational note: rcssserver never reuses uniform numbers after `(bye)`, so one server session accepts
only 11 player connections. After that, `init` fails with `no_more_player_or_goalie`, and the client
tells you to restart the server (`docker compose restart rcssserver`). Disconnected players are removed
from the pitch, so they never interfere with later episodes.


The simulator and feasibility estimates use the parameters parsed from the live server handshake
(`client.params`, snapshot saved as JSON with every experiment). `params.DEFAULT_PARAMS` is only a
fallback before a snapshot exists.

## 11. Evidence log

| Date | Item | Result | Source |
|---|---|---|---|
| 2026-09-24 | Relaxed-turn turn-then-dash estimate (any turn ≤ 35°, no noise, default params) | 88.9% `<40`, 91.7% `≤40` (N = 200k, SE ≈ 0.07 pp) | `python -m src.feasibility` |
| 2026-09-24 | Greedy ±35 controller, true (d, θ), no noise, default params | 74.0% `<40`, 76.4% `≤40` (N = 50k, SE ≈ 0.2 pp) | `python -m src.feasibility --controller --use-truth --no-noise --n 50000` |
| 2026-09-24 | Same, with motion + sensor noise | 76.3% `<40`, 78.8% `≤40` | `... --controller --use-truth --n 50000` |
| 2026-09-24 | Same, player sensing only (search + estimator), no noise | 69.2% `<40`, 71.6% `≤40` | `... --controller --no-noise --n 50000` |
| 2026-09-24 | Same, player sensing only, with noise | 71.0% `<40`, 73.5% `≤40` | `... --controller --n 50000` |

| 2026-09-24 | Greedy controller, live-server params, late sightings modelled | 70.5% `<40`, 73.0% `≤40` (N = 50k) | `... --controller --n 50000 --params notebooks/artifacts/server_params.json` |
| 2026-09-24 | Live server physics vs. simulator, fixed sequences (5 repeats; turn at rest 30) | position bias ≤ 0.16 m vs. live spread ≤ 0.31 m; turn at rest 35.03° ± 0.16 (n = 180); turn while moving within 0.6° of 35/(1+5v) | `python -m src.sim_vs_live`, notebook 03 §3 |
| 2026-09-25 | Live stability: 500 consecutive episodes (heuristic controller) | 0 missed commands, 0 reset errors, 0 mode changes, 0 server errors, 0 restarts; 32% of cycles use a late sighting | `python -m src.live_eval --policy greedy --episodes 500` |
| 2026-09-25 | Heuristic controller, same 500 starts: server vs. simulator | 67.2% vs. 67.6% `<40` (SE 2.1 pp); 30.2 vs. 30.0 mean steps | same |
| 2026-09-25 | Q-learning (C, α = 0.03, ε 1.0 → 0.1), 200 starts: server vs. simulator | **73.5%** vs. 71.5% `<40` (SE 3.1 pp); 76.5% vs. 74.5% `≤40` | `python -m src.live_eval --policy notebooks/artifacts/runs/qlearning_eps_decay_1.0_to_0.1/seed_4 --episodes 200` |
| 2026-09-25 | Q-learning live, by start distance | ≥ 95% for d₀ < 30 m; ≈ 33% for 30–35 m; 0% for 35–40 m | notebook 03 |

| 2026-09-25 | Internal algorithm selection (C, ε 1.0 → 0.1, 20k episodes, 5 seeds; last-5 greedy mean) | Q-learning 73.9% ± 0.7; SARSA 72.0% ± 1.3; MC sample-average 68.1% ± 7.3 (frozen after ≈ 4k episodes); MC constant α 48.2% ± 13.5 | `python -m src.train --preset algorithms`, notebook 04 |

| 2026-09-25 | **Provable** noiseless ceiling: turn until the ball is ahead (≤ 35°/cycle), then straight full dash | 94.9% `<40`, 97.6% `≤40` (N = 200k) — above 90%, so it does **not** show the criterion is unreachable. With noise a cycle can cover up to ≈ 1.155 m, so there is no strict impossibility | `python -m src.feasibility --ceiling` |
| 2026-09-25 | Far starts, measured | learned policy 0 / 26 captured at d₀ ≥ 35 m; heuristic 1 / 71 (35.24 m in 39 steps, live and simulated) | `live_eval_*.json` |

| 2026-09-25 | Finer representations (α 0.03, 40k episodes, 5 seeds): C control, D front ±5°, E front ±7° + 30 m edge, F speed-dependent front (±17.5° at rest, ±5.85° moving) | last-5: 72.9%, 63.2%, 57.3%, 60.6% (± 18.8); best single evaluation of any configuration ≤ 75.2% (also for Q-learning, SARSA and MC) | `python -m src.train --preset refinement`, notebook 04 |
| 2026-09-25 | Learned policy by start distance (live, 200 starts) | < 30 m: 135/138 (97.8%); 30–35 m: 12/36 (33.3%); 35–40 m: 0/26 → observed 147/200 = 73.5%. Reweighted to uniform d₀: 74.6% ± 1.4 (an estimate, not the observed total; the sample had 31% starts ≥ 30 m vs 28.6% expected). Under uniform d₀, exceeding 90% needs > 65% capture at 30–40 m; the data do not determine why far starts fail | notebook 04 |
| 2026-09-25 | Learning snapshots (Q after 0 … 20k episodes), 4 fixed starts | greedy captures 1, 0, 1, 3, 4, 4, 4 of 4 live; 0, 0, 1, 3, 4, 4, 4 in the simulator | `python -m src.live_snapshots`, notebooks 02 §5 and 03 §6 |

Except for the provable ceiling, no row is an upper bound over admissible policies. Together they show the 40-step budget is tight,
that the ±35 action set loses substantially to 35° turn quantization (relaxed 88.9% vs realizable
74.0%), and that initial acquisition costs about 5 pp. Noise slightly *helps* the greedy
controller (+2 pp), so favourable noise is a real effect, not only a theoretical caveat. The first five
rows use default params, and the live server's params match them (only `min_dash_power` differs, and it
is unused).
