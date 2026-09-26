# Plan: the other three catalog tasks + algorithm comparison (P1)

Status (2026-09-26): M0, M1 (shooting) and M2 (dribbling) done except their rcssserver checks (M5).
Branch `dimael`.
See "Progress log" at the end.

## 0. Scope

- Implement Goal Shooting, Dribbling and 2v1 Passing with kit-style kinematics (exact displacements,
  no physics, full observation), plus the Ball Pursuit task that is already done.
- Run **all tabular methods** (Q-learning, SARSA, first-visit MC with sample averages, first-visit MC with
  constant α) on **all four tasks**, each with ε constant vs ε decaying, 5 seeds.
- Present a cross-task comparison that explains *why* a method wins on a given task, only where a
  diagnostic experiment isolates the cause (section 5).
- Note: CLAUDE.md lists the algorithm comparison as a P2 rubric item and keeps it out of the P1 report.
  The user decided on 2026-09-25 to include it anyway. The required ε ablation stays the core result, and
  the comparison is an added section.

## 1. Final task specifications

Conventions: pitch 105 × 68 m, +x toward the rival goal (x = 52.5), angles in degrees, positive from +x
toward +y, TURN exactly ±35°. γ = 0.99 everywhere. Every env returns `info["success"]` at episode end.

### 1.1 Goal shooting (`src/shooting_env.py`)

- **Start:** d_g ~ U[11, 25] m (the ball's distance to the goal line), y ~ U[-10, 10] m; goal
  y ∈ [-7.01, 7.01].
  A keeper is present with p = 0.5 at y_k ~ U[-3, 3] on the goal line.
- **Actions (6):** shoot at one of 5 target points y* ∈ {-6, -3.5, 0, 3.5, 6} (left post, left, centre,
  right, right post), or CONDUCIR (+2 m along +x, keeping y; d_g floor at 11 m, where it becomes a no-op).
- **Transition** (changed from the report draft so that lateral position and posts matter):
  - Angular shot error ε_θ ~ N(0, σ_θ²) around the direction to y*, so the lateral error grows with the
    real distance to the target point.
  - **Goal:** crosses the line between the posts, and |y − y_k| ≥ R (the keeper's dive reach).
  - With a keeper, each CONDUCIR has probability p_out that the keeper comes out and blocks: −50,
    terminal. This is the cost that stops "always walk to 11 m" from being trivially optimal.
- **Provisional parameters:** σ_θ = 4°, R = 2.0 m, p_out = 0.1.
  - Fixed *before any training* by a calibration check on the exact optimum. It must pass three checks:
    (a) the optimal shot differs across keeper bins; (b) at least one state has CONDUCIR optimal;
    (c) the optimum with a keeper is < 100%.
  - The check's output is saved as JSON. It does **not** tune toward a target percentage.
- **State (36):**
  - d_g {11–15, 15–20, 20–25}
  - lateral {left y < −3.5, centre, right y > 3.5}
  - keeper {none, left y_k < −1, centre, right y_k > 1}
- **Reward:** +100 goal, −30 off target, −50 saved or blocked, −0.2 per CONDUCIR. T_max = 10 (a shot
  ends the episode).
- **Exact optimum (`src/shooting_optimal.py`):** P(goal | state, target) by numerical integration.
  - A small dynamic program over the CONDUCIR chain gives V*(s) for the true continuous start, averaged
    over the bin, and the best policy on our 36-state representation.
  - Both numbers are reported: the true optimum, and the optimum under our representation.
- **Criterion:** > 75% open goal and > 50% with keeper. Train one policy (keeper present with p = 0.5);
  evaluate each variant on its own fixed starts.

### 1.2 Dribbling (`src/dribbling_env.py`)

- **Start:** player at x0 ~ U[-40, -10], y0 ~ U[-20, 20], random heading; ball 0.5 m ahead along the
  heading.
- **Actions (4):** KICK 25 (ball +2 m along the heading, only if d_b ≤ 0.8, otherwise a no-op),
  DASH 80 (+0.8 m), TURN ±35. No player–ball collision.
- **State (36):**
  - d_b {≤ 0.8, (0.8, 2], (2, 4]}
  - θ_b {front ±17.5, left, right}
  - θ_g (bearing to the goal centre) {front ±17.5, left, right, behind |θ| > 90}
  - **d_g dropped.** It stays in [32, 92] m during an episode and never changes the optimal action.
    The report justifies this analytically.
- **Reward:** Δx of the ball − 0.1; −30 if d_b > 4 m or the ball or player leaves the pitch (terminal);
  +50 when the ball has advanced 30 m in x (terminal). T_max = 100.
- **Bound (constructive):** a scripted controller (turn until θ_g is front, then repeat kick → dash
  until d_b ≤ 0.8) succeeds from every start in ≤ ~53 steps, so 100% is attainable. We record the
  scripted controller's success and step counts as the reference.
- **Criterion:** 30 m advance with control in > 80% of episodes.

### 1.3 2v1 passing (`src/passing_env.py`)

- **Zone:** a 30 × 20 m rectangle centred at the origin. If the holder or the ball leaves it: −10,
  terminal.
- **Start:** holder uniform in the inner 24 × 14 m, random heading; teammate 6–10 m from the holder at a
  random angle (inside the zone); defender 4–8 m from the holder.
- **Scripted teammate:** moves ≤ 0.6 m/step toward a support point 8 m from the holder, on the side away
  from the defender (perpendicular to holder→defender), clipped to the zone.
- **Defender:** moves 0.6 m/step toward the holder, with heading noise N(0, 15²)°.
- **Actions (4):**
  - PASE: resolved within the step. Intercepted with p = 0.8 if the line is blocked (defender < 1.5 m
    from the segment), otherwise completed.
  - DRIBLE: holder and ball +1 m along the heading.
  - GIRAR: turn 35° in the direction that increases |θ_def|.
  - DESPEJE: terminal, reward 0.
- **Order within a step:** agent action → teammate move → defender move → tackle check (d_def < 1 m,
  p = 0.5).
- **After a completed pass:** the agent controls the receiver (heading = pass direction), and the
  previous holder becomes the scripted teammate.
- **State (324):**
  - d_comp {< 5, 5–10, ≥ 10}; θ_comp {front ±45, side, behind |θ| > 135}
  - d_def {< 2, 2–5, ≥ 5}; θ_def same bins as θ_comp
  - pass line blocked {yes, no}; zone {centre, touchline = within 3 m of an edge}
- **Reward:** +30 completed pass, −30 interception or tackle (terminal), −10 out (terminal). T_max = 60.
- **Known property (goes in the report):** an unblocked pass is risk-free and earns +30, so frequent
  passing is optimal. This is the brief's reward as given, not a bug.
- **Success:** reaches step 51 without interception, tackle, going out or DESPEJE, and ≥ 3 passes by
  the episode's end.
- **Reference:** a heuristic ("pass if the line is open, else dribble away from the defender"), labelled
  a heuristic, not a bound.

## 2. Shared infrastructure (milestone M0)

`train.py` and `discretizer.py` are left untouched, so the reported pipeline cannot break.

- **`src/task_discretizer.py`:** `ProductDiscretizer(features)`. Each feature has a name, an obs key,
  edges, labels, and a flag for signed-angle bins. It has the same `__call__(obs) -> int`, `n_states`,
  `state_to_label` and `decompose` API as `Discretizer`.
- **`src/task_train.py`:**
  - A `TaskSpec` registry (`pursuit_kit`, `shooting`, `dribbling`, `passing`). Each spec has an env
    factory, discretizer, action names, `eval_starts(n, seed, variant)`, trajectory keys, and
    `metrics(results)`.
  - Generic `run_episode` / `evaluate` / `train_one`. They save the same artifacts as `train.py`
    (`q.npy`, `history.npz` with online success, `eval.json`, `config.json` with `source_sha256`,
    snapshots) and reuse `agents.AGENTS` and `exploration`.
  - Presets: `matrix` (task × method × schedule), `alpha_sweep`, `diagnostics`.
- **Regression gate:** `pursuit_kit` run through `task_train` must reproduce the saved
  `runs_kit_final` Q-learning numbers exactly (same seeds, same RNG draws). If it can't, find out why
  before continuing.
- **`src/plots.py` (additions only):**
  - `plot_policy_grid(Q, disc, rows, cols, facet)`, a small-multiples policy/value view for any
    ProductDiscretizer
  - `plot_task_trajectory(record, task)`, which draws the zone or goal for the task
  - `plot_method_matrix(...)` for the comparison
- **Fingerprint:** new files change `source_fingerprint()` for future runs. Existing artifacts keep
  their recorded hash; the README explains this.
- **Tests (`src/tests/test_<task>_env.py`, `test_task_train.py`):** kinematics, rewards, terminals,
  success flag, discretizer bins at the edges, and a determinism test with a fixed seed. Run them with
  the memory cap:
  `ulimit -v 3000000; uv run --no-project -p 3.10 --with numpy --with pytest python -m pytest -q src/tests`.

## 3. Experiment protocol (identical for every task)

- **Methods:** `qlearning`, `sarsa`, `mc_first_visit` (sample average), `mc_first_visit_alpha`.
- **Schedules:** ε constant 0.1, and ε decaying 1.0 → 0.1 geometrically (the floor is reached at 60% of
  the budget).
- **Seeds:** training on 0–4. Greedy evaluation every `eval_every` episodes on fixed starts with common
  random numbers (`eval_seed = 12345`); 500 starts per variant.
- **α selection without touching the reported seeds:** sweep α ∈ {0.03, 0.1, 0.3} for each α-method and
  task on **seeds 5–9** with a short budget. Pick the best final greedy success, then run the matrix on
  seeds 0–4.
- **Budgets:** set by a Q-learning pilot (where its curve plateaus), then the same for every method on
  that task. Starting points: shooting 20k, dribbling 40k, 2v1 100k, pursuit 80k (existing).
- **Cost:** ~3 tasks × 4 methods × 2 schedules × 5 seeds = 120 runs, plus the α sweep and diagnostics.
  About 1 min per 80k-episode run on 16 cores gives well under an hour of compute. Use 8 workers
  (7 GB RAM).
- **Ball Pursuit:** existing results are reused. Missing cells are added: SARSA and both MC variants
  with constant ε (the decaying runs exist in `runs_kit_final_algorithms`).

## 4. Metrics for the comparison

For each cell (task × method × schedule), over 5 seeds (mean, min–max, and a bootstrap 95% CI over
seeds):

1. Final greedy success (the task criterion) and the gap to the exact optimum or bound where one exists
   (pursuit, shooting, dribbling).
2. Sample efficiency: area under the greedy success curve, and episodes to reach 90% of the final value.
3. Online (ε-greedy) success and return during training. This separates learning the greedy policy
   (off-policy) from performing well while exploring (on-policy).
4. Stability: spread across seeds.
5. Value accuracy (shooting only, where V* is exact): the mean of V̂(s) − V*(s) over states.

A difference is claimed only when the CIs don't overlap; otherwise we write "no detectable
difference". With 5 seeds, this is stated as a limitation.

## 5. Hypotheses to test (fixed before running) and the diagnostic that isolates each

Each task has a different structure, and theory predicts different winners. The report explains a
difference only when its diagnostic supports it. If the data contradicts a hypothesis, we report that.

| Task | Structure | Prediction (theory) | Diagnostic |
|---|---|---|---|
| Pursuit | deterministic, no risk, ~20 steps, dense reward | All methods tie; the ceiling is the representation, not the update rule (already seen: 85.3 / 85.7 / 86.0% vs 87.9% optimum). | Done. MC with constant α failed (34%, large spread); its cause is **not yet isolated**. The α sweep (is α = 0.1 too large for full-return targets?) comes before any explanation. |
| Shooting | 1–10 steps, stochastic outcome, terminal reward | Similar greedy success. Q-learning's max over noisy estimates biases V upward (maximization bias, Sutton & Barto §6.7); MC sample averages are unbiased. | Measure V̂ − V* per method against the exact V*; repeat with larger σ_θ (more outcome noise), where the bias should grow for Q-learning only. |
| Dribbling | deterministic, ~50 steps, dense reward, −30 cliff for losing the ball | TD propagates faster than MC over long episodes. Under constant ε, SARSA and MC learn the ε-soft policy and may play safer ("cliff-walking" effect), so their online success is higher and greedy success equal or lower. | Episodes to 90%, online vs greedy success by method; compare policies in the risky states (ball at (2, 4] m, misaligned). Repeat with a doubled loss radius (less risk): the SARSA/Q gap should shrink. |
| 2v1 | stochastic, high risk (−30), 324 states with heavy aliasing (not Markov) | SARSA safer online (on-policy under risk). MC robust to aliasing because it doesn't bootstrap from aliased states (Singh, Jaakkola & Jordan 1994). Q-learning is hurt by both. | (a) **Aliasing:** rerun with a finer discretization (more d/θ bins and a defender-closing bit); if MC's relative advantage shrinks, aliasing is supported. (b) **Risk:** tackle p ∈ {0.2, 0.5, 0.8}; the SARSA–Q online gap should grow with p. |

Also: does the ε schedule matter more for on-policy methods? With constant ε, SARSA and MC converge to
the best ε-soft policy rather than the greedy optimum, so their greedy-evaluation gap between the two
schedules should be larger than Q-learning's. The matrix answers this directly.

## 6. Milestones (in order; each gate must pass before the next)

| # | Work | Gate |
|---|---|---|
| M0 | `task_discretizer`, `task_train`, plot additions, tests | pursuit regression reproduces `runs_kit_final` exactly; all tests pass |
| M1 | Shooting: env + tests, `shooting_optimal`, calibration check, pilot, α sweep, matrix | calibration JSON passes (a)–(c); criterion checked against the optimum |
| M2 | Dribbling: env + tests, scripted bound, pilot, α sweep, matrix | scripted controller 100%; env tests pass |
| M3 | 2v1: env + tests, heuristic reference, pilot, α sweep, matrix | env tests (pass geometry, tackle, control switch) pass |
| M4 | Pursuit missing cells; diagnostics from section 5; notebook 13 | every claim in the comparison maps to a saved JSON |
| M5 | rcssserver execution checks (below) | a log per task, with no socket errors |
| M6 | Report, README, SPEC rows | 8 pages; every number traced to an artifact |

Priority rule: in each task, the required Q-learning ε ablation is reviewed first (figures inspected)
before the other methods' cells are analysed.

## 7. rcssserver execution checks (M5)

Execution and logging only; no performance claims beyond what is measured.

- **Shooting:** trainer places the ball and player; send `kick 100 <dir>` toward each target zone.
  Open goal, plus a keeper placed by the trainer as a second, idle client. Log the result from ball
  position and referee messages.
- **Dribbling:** run the greedy policy with the brief's literal `kick 25` / `dash 80` / `turn ±35`. Log
  the measured ball displacement per kick. The server gives up to ~11 m (0.675 m/cycle, decay 0.94),
  not the 2 m we model; this is reported as a transfer gap.
- **2v1:** 3 clients (2 attackers + scripted defender), within the 11-connection limit (restart
  `rcssserver` between sessions). A few episodes, logged.

## 8. Notebooks (Spanish)

- `10_tiro_a_puerta.ipynb`, `11_conduccion.ipynb`, `12_pase_2v1.ipynb`. Each has: MDP, calibration or
  bound, the ε ablation curves (G0, success, steps), policy/value view, 2D trajectory while learning,
  criterion check, and the live check.
- `13_comparacion_algoritmos.ipynb`: the task × method matrix, learning curves, the section 5
  diagnostics, and the conclusions per task.
- Executed inside `rl-agent` with `jupyter nbconvert --execute --inplace`.

## 9. Report page budget (8 pages max)

| Section | Pages |
|---|---|
| Intro (4 tasks) | 0.6 |
| MDP: pursuit (0.8) + table for the other 3 with justification (0.8) | 1.6 |
| Algorithms: 4 update rules; on/off-policy, bootstrapping, bias | 0.5 |
| Pursuit results (trimmed: macro-actions and sensitivity get one sentence each) | 1.2 |
| Other 3 tasks: one results table + one 3-row figure (ε curves, policy, trajectory) | 1.3 |
| Algorithm comparison: matrix table + one figure + explanations backed by diagnostics | 1.2 |
| rcssserver checks and transfer | 0.4 |
| Conclusions + references (add Singh et al. 1994, van Hasselt 2010) | 0.6 |
| Slack for floats | 0.6 |

## 10. Risks

- Some hypotheses may not show up (e.g., all methods tie everywhere). Then the finding is that "the
  update rule matters less than the representation in these bounded tasks", which is still a valid,
  reportable result.
- 2v1 learning may be slow with 324 states. Raise the budget first, and only then consider the finer
  representation (which is also the aliasing diagnostic).
- Page pressure: if over 8 pages, move per-task policy views to the notebooks and keep one figure per
  task.

## Progress log

### M0 — done
- `src/task_discretizer.py`, `src/task_train.py`, `src/task_analysis.py`, plot additions in `src/plots.py`.
- Regression gate passed: `pursuit_kit` through `task_train` reproduces all 20 greedy evaluations of
  `runs_kit_final` seed 0 bit-for-bit. `test_task_train.py` checks parity with `train.py`.
- Tests: 67 pass.

### M1 — shooting (all but the live check)
- Calibration: the provisional parameters failed check (b). The grid rule selected sigma 4 deg, reach 2.0 m,
  p_out 0.02 (`shooting_calibration.json`). CONDUCIR is optimal in only 1 of 36 states, by 0.25 points, so the
  task is effectively one decision. The user chose to keep this and state it in the report.
- Optimum (`shooting_optimal.json`): open goal 100%. With a keeper: true optimum 95.2%, representation
  optimum 93.3% (sampling std 1.1%).
- The alpha sweep (seeds 5–9, decaying epsilon) selected 0.03 for every constant-alpha method
  (`shooting_alphas.json`).
- Matrix (`runs_shooting`), 50k episodes, 5 seeds: every condition meets the criterion in every seed.
- Diagnostics (`runs_shooting_diag`, summary in `shooting_summary.json`):
  - Constant epsilon with alpha 0.03 trails decaying epsilon (Q-learning 87.5 vs 92.3% with keeper), but
    alpha 0.1, Q0 = 100 or 4x the budget each close the gap. This is slow correction from Q0 = 0, not a
    different final solution.
  - MC with sample averages is best under constant epsilon and learns sooner under decaying epsilon (AUC
    CIs separated); there is no detectable final difference under decaying epsilon.
  - Q-learning = SARSA by construction (shots are terminal).
- **Correction to section 5, row "Shooting":** the maximization-bias prediction was badly posed. Q-learning's
  max only enters the target after CONDUCIR (shots are terminal), so it has almost no room to act. The
  observed value bias is underestimation (Q0 = 0), and the larger-sigma diagnostic was not run because it
  would not isolate a Q-learning effect.
- Notebook: `notebooks/10_tiro_a_puerta.ipynb` (executed in `rl-agent`). Figures:
  `fig_shooting_{optimal_policy,ablation,training,learned_policy,learning_traj,methods}.png`.

### Protocol change proposed for M2/M3 (from the shooting confound)
Alpha was selected with decaying epsilon only and then reused for constant epsilon, which confounded the
epsilon ablation. For dribbling and 2v1, sweep alpha per (method, schedule) on seeds 5–9.

### M2 — dribbling (all but the live check)
- Constructive bound: the scripted controller succeeds on 500/500 evaluation starts and 20 000/20 000 random
  starts (mean 60.4 steps, max 85 < T_max 100). The plan's "<= ~53 steps" estimate was wrong: after each kick the
  chase takes 2–3 dashes.
- Per-schedule alpha sweep (seeds 5–9, 60k episodes, `dribbling_alphas.json`): Q-learning 0.1 (constant) /
  0.03 (decay); SARSA 0.03 / 0.03; MC constant alpha 0.1 / 0.03.
- Matrix (`runs_dribbling`, 60k, seeds 0–4) and diagnostics (`runs_dribbling_diag`), summarized in
  `dribbling_summary.json`. Notebook `11_conduccion.ipynb`, figures `fig_dribbling_*.png`.
- **Finding: the greedy policy oscillates** in a deterministic env. The dips are timeouts from TURN +/-35 two-cycles
  between aggregated states with near-tied values. With the 245-state representation, Q-learning and SARSA are
  at 100% in every one of the last 5 evaluations (36 states: 95.1% / 97.6%), so the oscillation is caused by
  aggregation. The pre-registered 36-state representation is too coarse for this task.
- Epsilon ablation (Q-learning, alpha tuned per schedule): decaying beats constant (last-5 mean 97.7 vs 89.7%,
  CIs separated). SARSA: no final difference, but decaying is much slower (AUC 76 vs 96%).
- **H2 (risk/cliff) not observed:** losses while exploring are <= 0.5%; failures are timeouts, and a loss radius
  of 8 m changes nothing.
- **H3 (MC robust to aliasing) contradicted:** sample-average MC suffers most from the coarse representation
  (64.5%, frozen policy) and gains most from the fine one (97.8%).
- MC with constant alpha fails everywhere (<= 53% even at alpha 0.003, and with 245 states). Cause not isolated.
- Compute note: runs are CPU-bound Python (8.8 us/step alone). Running 15 workers on this 8-core laptop made each
  about 6x slower; use about 8 workers in total for 2v1.
- Plot change: task curves now use min–max bands (mean ± std went outside [0, 1]); notebooks 10 and 11 re-run.
- Provenance note: the dribbling sweep runs were saved while `src/` gained additive code (the fine representation,
  diagnostics, plots), so their `source_sha256` is that of the later code; sweep behavior was not changed.
