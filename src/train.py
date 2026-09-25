"""Training and evaluation on the simulator, following the experiment contract of the plan.

* Several training seeds per condition, same episode budget for every condition.
* Periodic greedy evaluation (epsilon = 0, learning off) on one fixed list of
  starts shared by all conditions, with the simulator's noise also seeded
  identically (common random numbers), so curves are directly comparable.
* Every run saves ``q.npy``, ``history.npz``, ``eval.json`` and
  ``config.json`` (bins, actions, hyperparameters, seeds, physics params, git commit).

    python -m src.train --preset ablation --out notebooks/artifacts/runs
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents import AGENTS, GreedyQPolicy, TabularAgent
from src.discretizer import Discretizer, DiscretizerConfig
from src.env_base import ACTION_NAMES, ACTIONS, BallPursuitEnvBase, T_MAX
from src.exploration import decay_reaching, make_schedule
from src.params import DEFAULT_PARAMS, load_params
from src.sampler import StartConfig, evaluation_starts
from src.sim_env import SimBallPursuitEnv

DEFAULT_PARAMS_PATH = "notebooks/artifacts/server_params.json"


@dataclass
class ExperimentConfig:
    name: str
    algorithm: str = "qlearning"
    n_episodes: int = 20_000
    alpha: float = 0.03
    gamma: float = 0.99
    schedule: Dict[str, Any] = field(default_factory=lambda: {"kind": "constant", "eps": 0.1})
    discretizer: DiscretizerConfig = field(default_factory=DiscretizerConfig)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    t_max: int = T_MAX
    eval_every: int = 1000
    eval_episodes: int = 500
    eval_seed: int = 12345
    params_path: Optional[str] = DEFAULT_PARAMS_PATH
    # Episodes (completed so far) at which to save a Q-table copy and record that training
    # episode's trajectory, to show the policy while it learns. Does not affect results.
    snapshot_episodes: List[int] = field(default_factory=list)
    # Server cycles per decision step (alternative reading of "paso"; 1 = implemented baseline).
    cycles_per_step: int = 1
    # "server_sim": simulator with rcssserver physics and player sensing (src/sim_env.py);
    # "kit": the course starter kit's kinematic environment (src/kit_env.py).
    env: str = "server_sim"
    # Extra start laws evaluated once with the final Q-table (e.g. ["kit"] = starter-kit reset).
    extra_eval_laws: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["discretizer"] = self.discretizer.as_dict()
        return d


def load_run_params(path: Optional[str]) -> Dict[str, Any]:
    return load_params(path) if path and os.path.exists(path) else dict(DEFAULT_PARAMS)


TRAJECTORY_KEYS = ("step", "player", "body_dir", "ball", "true_d", "true_theta", "ball_status")


def run_episode(env: BallPursuitEnvBase, agent: TabularAgent, disc: Discretizer,
                epsilon: float, learn: bool = True, start: Optional[StartConfig] = None,
                record: bool = False) -> Dict[str, Any]:
    obs, info = env.reset(start)
    trajectory = [{k: info[k] for k in TRAJECTORY_KEYS}] if record else None
    actions = []
    s = disc(obs)
    a = agent.act(s, epsilon)
    ret, disc_ret, discount = 0.0, 0.0, 1.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(a)
        if record:
            trajectory.append({k: info[k] for k in TRAJECTORY_KEYS})
            actions.append(a)
        s2 = disc(obs)
        # Next action: needed to act, and by SARSA at truncation to bootstrap from s2.
        # Other agents skip the draw at the end so their random streams are unchanged.
        done = terminated or (truncated and not agent.needs_next_action)
        a2 = None if done else agent.act(s2, epsilon)
        if learn:
            agent.observe(s, a, r, s2, a2, terminated, truncated)
        ret += r
        disc_ret += discount * r
        discount *= agent.gamma
        s, a = s2, a2
    if learn:
        agent.end_episode()
    out = {"return": ret, "discounted_return": disc_ret, "steps": env.step_count,
           "cycles": env.cycle_count,
           "captured": terminated, "capture_step": env.step_count if terminated else None}
    if record:
        out["trajectory"], out["actions"] = trajectory, actions
    return out


def evaluate(env: SimBallPursuitEnv, agent: TabularAgent, disc: Discretizer,
             starts: List[StartConfig], noise_seed: int) -> Dict[str, Any]:
    """Greedy policy, no learning, fixed starts and fixed simulator noise stream."""
    env.rng = np.random.default_rng(noise_seed)
    q_before = agent.Q.copy()
    results = [run_episode(env, agent, disc, 0.0, learn=False, start=s) for s in starts]
    assert np.array_equal(q_before, agent.Q)
    steps = np.array([r["capture_step"] or 10**6 for r in results])
    return {"capture_rate_lt40": float(np.mean(steps <= 39)),
            "capture_rate_le40": float(np.mean(steps <= 40)),
            "mean_return": float(np.mean([r["return"] for r in results])),
            "mean_discounted_return": float(np.mean([r["discounted_return"] for r in results])),
            "mean_steps": float(np.mean([r["steps"] for r in results])),
            "mean_cycles": float(np.mean([r["cycles"] for r in results])),
            "mean_capture_cycles": (float(np.mean([r["cycles"] for r in results if r["captured"]]))
                                    if any(r["captured"] for r in results) else None)}


def source_fingerprint() -> str:
    """SHA-256 over the contents of ``src/*.py`` (sorted): identifies the code that produced
    a run even inside the container, where ``.git`` is not mounted."""
    import hashlib
    src = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(f for f in os.listdir(src) if f.endswith(".py")):
        h.update(name.encode())
        with open(os.path.join(src, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_env(cfg: ExperimentConfig, params: Dict[str, Any], seed: int):
    if cfg.env == "kit":
        from src.kit_env import KitBallPursuitEnv
        return KitBallPursuitEnv(t_max=cfg.t_max, seed=seed)
    if cfg.env == "server_sim":
        return SimBallPursuitEnv(params, t_max=cfg.t_max, seed=seed, cycles_per_step=cfg.cycles_per_step)
    raise ValueError(f"unknown env {cfg.env!r}")


def train_one(cfg: ExperimentConfig, seed: int, out_dir: Optional[str] = None) -> Dict[str, Any]:
    params = load_run_params(cfg.params_path)
    disc = Discretizer(cfg.discretizer)
    env = make_env(cfg, params, seed)
    eval_env = make_env(cfg, params, cfg.eval_seed)
    starts = evaluation_starts(cfg.eval_episodes, seed=cfg.eval_seed)
    agent = AGENTS[cfg.algorithm](disc.n_states, len(ACTIONS), cfg.alpha, cfg.gamma, seed=seed)
    schedule = make_schedule(cfg.schedule)

    hist = {k: np.zeros(cfg.n_episodes) for k in
            ("return", "discounted_return", "steps", "captured", "capture_step", "epsilon")}
    evals, t0 = [], time.time()
    snapshots, snapshot_q, train_trajs = set(cfg.snapshot_episodes), {}, []
    for ep in range(cfg.n_episodes):
        eps = schedule(ep)
        if ep in snapshots:
            snapshot_q[f"ep_{ep}"] = agent.Q.copy()
        r = run_episode(env, agent, disc, eps, record=ep in snapshots)
        if ep in snapshots:
            train_trajs.append({"episode": ep, "epsilon": eps, **r})
        hist["return"][ep], hist["discounted_return"][ep] = r["return"], r["discounted_return"]
        hist["steps"][ep], hist["captured"][ep] = r["steps"], r["captured"]
        hist["capture_step"][ep] = r["capture_step"] or np.nan
        hist["epsilon"][ep] = eps
        if (ep + 1) % cfg.eval_every == 0:
            evals.append({"episode": ep + 1, **evaluate(eval_env, agent, disc, starts, cfg.eval_seed)})

    result = {"name": cfg.name, "seed": seed, "evals": evals, "final_eval": evals[-1],
              "train_seconds": time.time() - t0}
    for law in cfg.extra_eval_laws:
        extra = evaluation_starts(cfg.eval_episodes, seed=cfg.eval_seed, law=law)
        result[f"final_eval_{law}"] = evaluate(eval_env, agent, disc, extra, cfg.eval_seed)
    if out_dir:
        run_dir = os.path.join(out_dir, cfg.name, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        np.save(os.path.join(run_dir, "q.npy"), agent.Q)
        if cfg.n_episodes in snapshots:
            snapshot_q[f"ep_{cfg.n_episodes}"] = agent.Q.copy()
        if snapshot_q:
            np.savez_compressed(os.path.join(run_dir, "q_snapshots.npz"), **snapshot_q)
            with open(os.path.join(run_dir, "train_trajectories.json"), "w") as f:
                json.dump(train_trajs, f)
        np.savez_compressed(os.path.join(run_dir, "history.npz"), **hist)
        with open(os.path.join(run_dir, "eval.json"), "w") as f:
            json.dump(result, f, indent=2)
        config = {**cfg.to_dict(), "seed": seed, "actions": ACTION_NAMES,
                  "n_states": disc.n_states, "params": params, "git_commit": git_commit(),
                  "source_sha256": source_fingerprint()}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    return result


def run_experiment(cfg: ExperimentConfig, out_dir: Optional[str], workers: int = 8
                   ) -> List[Dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=min(workers, len(cfg.seeds))) as pool:
        return list(pool.map(train_one, [cfg] * len(cfg.seeds), cfg.seeds,
                             [out_dir] * len(cfg.seeds)))


def ablation_configs(n_episodes: int = 20_000, **overrides) -> List[ExperimentConfig]:
    """The required exploration ablation: constant vs geometrically decaying epsilon.

    Decay reaches eps_min at 60% of the budget; the constant epsilon equals the
    decaying schedule's floor, so both end with the same exploration level.
    """
    base = ExperimentConfig(name="", n_episodes=n_episodes, **overrides)
    return [
        replace(base, name="qlearning_eps_const_0.1",
                schedule={"kind": "constant", "eps": 0.1}),
        replace(base, name="qlearning_eps_decay_1.0_to_0.1",
                schedule={"kind": "decay", "eps_start": 1.0, "eps_min": 0.1,
                          "decay": decay_reaching(1.0, 0.1, int(0.6 * n_episodes))}),
    ]


def discretization_configs(n_episodes: int = 20_000, **overrides) -> List[ExperimentConfig]:
    """Candidate state representations, compared under identical training conditions.

    * A: front |theta| <= 10, no speed (initial proposal);
    * B: front |theta| <= 17.5 = half the 35-deg turn quantum, so every heading can be
      brought into the front bin by turning at rest;
    * C: B + speed bit at 0.2 m/cycle (turn size 35/(1+5v): 35 deg at rest, ~12 deg at speed);
    * D: C with the front split at 5 deg for fine alignment while moving.
    """
    decay = {"kind": "decay", "eps_start": 1.0, "eps_min": 0.1,
             "decay": decay_reaching(1.0, 0.1, int(0.6 * n_episodes))}
    overrides.setdefault("alpha", 0.1)  # the comparison was run at alpha 0.1 (and 0.03 via --alpha)
    base = ExperimentConfig(name="", n_episodes=n_episodes, schedule=decay, **overrides)
    dist = (3.0, 10.0, 20.0)
    return [
        replace(base, name="disc_A_front10", discretizer=DiscretizerConfig(dist, (10.0, 90.0, 180.0))),
        replace(base, name="disc_B_front17.5", discretizer=DiscretizerConfig(dist, (17.5, 90.0, 180.0))),
        replace(base, name="disc_C_front17.5_speed",
                discretizer=DiscretizerConfig(dist, (17.5, 90.0, 180.0), (0.2,))),
        replace(base, name="disc_D_front5_17.5_speed",
                discretizer=DiscretizerConfig(dist, (5.0, 17.5, 90.0, 180.0), (0.2,))),
    ]


def algorithm_configs(n_episodes: int = 20_000, **overrides) -> List[ExperimentConfig]:
    """Internal algorithm selection (not a report deliverable): same state representation,
    decaying schedule, budget and seeds; only the update rule differs. MC is run with
    sample averages (``alpha`` unused) and with the same constant alpha as the TD agents."""
    decay = {"kind": "decay", "eps_start": 1.0, "eps_min": 0.1,
             "decay": decay_reaching(1.0, 0.1, int(0.6 * n_episodes))}
    base = ExperimentConfig(name="", n_episodes=n_episodes, schedule=decay, **overrides)
    return [replace(base, name=f"algo_{algo}", algorithm=algo)
            for algo in ("qlearning", "sarsa", "mc_first_visit", "mc_first_visit_alpha")]


def refinement_configs(n_episodes: int = 40_000, **overrides) -> List[ExperimentConfig]:
    """Is the ~74% plateau set by the state representation? Finer heading bins (for the
    small corrections made while moving) and an extra far-distance edge, with twice the
    budget; C at the same budget is the control."""
    decay = {"kind": "decay", "eps_start": 1.0, "eps_min": 0.1,
             "decay": decay_reaching(1.0, 0.1, int(0.6 * n_episodes))}
    base = ExperimentConfig(name="", n_episodes=n_episodes, schedule=decay,
                            eval_every=2000, **overrides)
    return [
        replace(base, name="ref_C_control", discretizer=DiscretizerConfig()),
        replace(base, name="ref_D_front5",
                discretizer=DiscretizerConfig((3.0, 10.0, 20.0), (5.0, 17.5, 90.0, 180.0), (0.2,))),
        replace(base, name="ref_E_front7_far30",
                discretizer=DiscretizerConfig((3.0, 10.0, 20.0, 30.0), (7.0, 17.5, 90.0, 180.0), (0.2,))),
        # Front width = half the turn quantum at each speed: +/-17.5 at rest (35 deg turns),
        # +/-5.85 while moving (35/(1+5*0.4) = 11.7 deg turns at cruise).
        replace(base, name="ref_F_speed_dependent_front",
                discretizer=DiscretizerConfig((3.0, 10.0, 20.0), (17.5, 35.0, 90.0, 180.0), (0.2,),
                                              angle_edges_moving=(5.85, 17.5, 90.0, 180.0))),
    ]


def macro_configs(n_episodes: int = 20_000, **overrides) -> List[ExperimentConfig]:
    """Alternative reading of "paso": one decision = k server cycles (command repeated).

    Same algorithm, representation, hyperparameters, schedule, seeds and evaluation
    starts as the reported baseline; only k changes. The budget stays at 40 decision
    steps (= 40k cycles). k = 1 reproduces the reported decaying-epsilon runs."""
    decay = {"kind": "decay", "eps_start": 1.0, "eps_min": 0.1,
             "decay": decay_reaching(1.0, 0.1, int(0.6 * n_episodes))}
    base = ExperimentConfig(name="", n_episodes=n_episodes, schedule=decay, **overrides)
    return [replace(base, name=f"macro_k{k}", cycles_per_step=k) for k in (1, 2, 3)]


# ---------------------------------------------------------------- starter kit
KIT_DISCRETIZATIONS = {
    # Chosen for the report after kit_refinement/kit_refinement2: 9 distance x 11 heading bins.
    "R3": DiscretizerConfig((3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
                            (2.5, 7.5, 17.5, 35.0, 90.0, 180.0), ()),
    # The starter kit's own scheme (agente_cero notebook): 0.8 / 3 / 8 m, 15 / 60 deg.
    "kitdisc": DiscretizerConfig((0.8, 3.0, 8.0), (15.0, 60.0, 180.0), ()),
    # Ours without the speed bit (the kit has no inertia): 3 / 10 / 20 m, +/-17.5 / 90 deg.
    "ourdisc": DiscretizerConfig((3.0, 10.0, 20.0), (17.5, 90.0, 180.0), ()),
}


def _kit_base(n_episodes: int, **overrides) -> ExperimentConfig:
    decay = {"kind": "decay", "eps_start": 1.0, "eps_min": 0.1,
             "decay": decay_reaching(1.0, 0.1, int(0.6 * n_episodes))}
    overrides.setdefault("schedule", decay)
    return ExperimentConfig(name="", n_episodes=n_episodes, env="kit", params_path=None,
                            extra_eval_laws=["kit"], **overrides)


def kit_discretization_configs(n_episodes: int = 20_000, **overrides) -> List[ExperimentConfig]:
    """Starter-kit environment: its discretization vs ours (no speed bit), Q-learning, two alphas."""
    base = _kit_base(n_episodes, **overrides)
    return [replace(base, name=f"kit_{dname}_alpha{alpha:g}", discretizer=dcfg, alpha=alpha)
            for dname, dcfg in KIT_DISCRETIZATIONS.items() for alpha in (0.03, 0.1)]


def kit_algorithm_configs(n_episodes: int = 80_000, discretizer: str = "R3", alpha: float = 0.1,
                          **overrides) -> List[ExperimentConfig]:
    """Starter-kit environment: internal algorithm selection with the chosen representation."""
    overrides.setdefault("eval_every", 4000)
    base = _kit_base(n_episodes, discretizer=KIT_DISCRETIZATIONS[discretizer], alpha=alpha, **overrides)
    return [replace(base, name=f"kit_algo_{algo}", algorithm=algo)
            for algo in ("qlearning", "sarsa", "mc_first_visit", "mc_first_visit_alpha")]


def kit_ablation_configs(n_episodes: int = 80_000, discretizer: str = "R3", alpha: float = 0.1,
                         algorithm: str = "qlearning", **overrides) -> List[ExperimentConfig]:
    """Starter-kit environment: the required exploration ablation (constant vs decaying epsilon)."""
    overrides.setdefault("eval_every", 4000)
    base = _kit_base(n_episodes, discretizer=KIT_DISCRETIZATIONS[discretizer], alpha=alpha,
                     algorithm=algorithm, **overrides)
    return [
        replace(base, name=f"kit_{algorithm}_eps_const_0.1", schedule={"kind": "constant", "eps": 0.1}),
        replace(base, name=f"kit_{algorithm}_eps_decay_1.0_to_0.1"),
    ]


def kit_refinement_configs(n_episodes: int = 40_000, **overrides) -> List[ExperimentConfig]:
    """Starter kit, full [5, 40] m range: finer representations to close the gap to the exact
    optimum (src/kit_optimal.py). Heading precision matters at long range: with exact 35-deg
    turns the capture tolerance asin(0.8/d) is only 1.5-2.3 deg at 20-30 m."""
    reps = {
        "R1": DiscretizerConfig((3.0, 10.0, 20.0), (5.0, 17.5, 90.0, 180.0), ()),
        "R2": DiscretizerConfig((3.0, 6.0, 10.0, 15.0, 20.0, 30.0), (5.0, 17.5, 90.0, 180.0), ()),
        "R3": DiscretizerConfig((3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
                                (2.5, 7.5, 17.5, 35.0, 90.0, 180.0), ()),
    }
    base = _kit_base(n_episodes, eval_every=2000, **overrides)
    return [replace(base, name=f"kit_ref_{r}_alpha{alpha:g}", discretizer=d, alpha=alpha)
            for r, d in reps.items() for alpha in (0.03, 0.1)]


KIT_R3 = DiscretizerConfig((3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
                           (2.5, 7.5, 17.5, 35.0, 90.0, 180.0), ())
KIT_R4 = DiscretizerConfig((2.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0),
                           (1.25, 2.5, 5.0, 7.5, 12.5, 17.5, 35.0, 90.0, 180.0), ())


def kit_refinement2_configs(n_episodes: int = 80_000, **overrides) -> List[ExperimentConfig]:
    """R3 with a longer budget and a finer R4, alpha 0.1."""
    base = _kit_base(n_episodes, eval_every=4000, alpha=0.1, **overrides)
    return [replace(base, name=f"kit_ref_R3_alpha0.1_{n_episodes // 1000}k", discretizer=KIT_R3),
            replace(base, name=f"kit_ref_R4_alpha0.1_{n_episodes // 1000}k", discretizer=KIT_R4)]


PRESETS = {"kit_refinement": kit_refinement_configs, "kit_refinement2": kit_refinement2_configs, "ablation": ablation_configs, "discretization": discretization_configs,
           "algorithms": algorithm_configs, "refinement": refinement_configs,
           "macro": macro_configs, "kit_discretization": kit_discretization_configs,
           "kit_algorithms": kit_algorithm_configs, "kit_ablation": kit_ablation_configs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="ablation")
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--out", default="notebooks/artifacts/runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None, help="override the learning rate")
    ap.add_argument("--only", nargs="*", help="run only these config names (before suffixes)")
    ap.add_argument("--seeds", nargs="*", type=int, help="override the training seeds")
    ap.add_argument("--snapshots", nargs="*", type=int, default=[],
                    help="episodes at which to save Q snapshots and training trajectories")
    args = ap.parse_args()
    for cfg in PRESETS[args.preset](args.episodes):
        if args.seeds:
            cfg = replace(cfg, seeds=args.seeds)
        if args.snapshots:
            cfg = replace(cfg, snapshot_episodes=args.snapshots)
        if args.only and cfg.name not in args.only:
            continue
        if args.alpha is not None:
            cfg = replace(cfg, alpha=args.alpha, name=f"{cfg.name}_alpha{args.alpha:g}")
        t0 = time.time()
        results = run_experiment(cfg, args.out, args.workers)
        final = [r["final_eval"]["capture_rate_lt40"] for r in results]
        last5 = [np.mean([e["capture_rate_lt40"] for e in r["evals"][-5:]]) for r in results]
        worst = [min(e["capture_rate_lt40"] for e in r["evals"][-5:]) for r in results]
        print(f"{cfg.name}: greedy capture <40 final = {np.mean(final):.3f} ± {np.std(final):.3f}, "
              f"mean of last 5 evals = {np.mean(last5):.3f} ± {np.std(last5):.3f}, "
              f"worst of last 5 = {np.min(worst):.3f} (seeds {cfg.seeds}, "
              f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
