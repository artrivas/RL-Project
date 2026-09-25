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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["discretizer"] = self.discretizer.as_dict()
        return d


def load_run_params(path: Optional[str]) -> Dict[str, Any]:
    return load_params(path) if path and os.path.exists(path) else dict(DEFAULT_PARAMS)


def run_episode(env: BallPursuitEnvBase, agent: TabularAgent, disc: Discretizer,
                epsilon: float, learn: bool = True, start: Optional[StartConfig] = None
                ) -> Dict[str, Any]:
    obs, _ = env.reset(start)
    s = disc(obs)
    a = agent.act(s, epsilon)
    ret, disc_ret, discount = 0.0, 0.0, 1.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, _ = env.step(a)
        s2 = disc(obs)
        a2 = None if terminated or truncated else agent.act(s2, epsilon)
        if learn:
            agent.observe(s, a, r, s2, a2, terminated, truncated)
        ret += r
        disc_ret += discount * r
        discount *= agent.gamma
        s, a = s2, a2
    if learn:
        agent.end_episode()
    return {"return": ret, "discounted_return": disc_ret, "steps": env.step_count,
            "captured": terminated, "capture_step": env.step_count if terminated else None}


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
            "mean_steps": float(np.mean([r["steps"] for r in results]))}


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train_one(cfg: ExperimentConfig, seed: int, out_dir: Optional[str] = None) -> Dict[str, Any]:
    params = load_run_params(cfg.params_path)
    disc = Discretizer(cfg.discretizer)
    env = SimBallPursuitEnv(params, t_max=cfg.t_max, seed=seed)
    eval_env = SimBallPursuitEnv(params, t_max=cfg.t_max, seed=cfg.eval_seed)
    starts = evaluation_starts(cfg.eval_episodes, seed=cfg.eval_seed)
    agent = AGENTS[cfg.algorithm](disc.n_states, len(ACTIONS), cfg.alpha, cfg.gamma, seed=seed)
    schedule = make_schedule(cfg.schedule)

    hist = {k: np.zeros(cfg.n_episodes) for k in
            ("return", "discounted_return", "steps", "captured", "capture_step", "epsilon")}
    evals, t0 = [], time.time()
    for ep in range(cfg.n_episodes):
        eps = schedule(ep)
        r = run_episode(env, agent, disc, eps)
        hist["return"][ep], hist["discounted_return"][ep] = r["return"], r["discounted_return"]
        hist["steps"][ep], hist["captured"][ep] = r["steps"], r["captured"]
        hist["capture_step"][ep] = r["capture_step"] or np.nan
        hist["epsilon"][ep] = eps
        if (ep + 1) % cfg.eval_every == 0:
            evals.append({"episode": ep + 1, **evaluate(eval_env, agent, disc, starts, cfg.eval_seed)})

    result = {"name": cfg.name, "seed": seed, "evals": evals, "final_eval": evals[-1],
              "train_seconds": time.time() - t0}
    if out_dir:
        run_dir = os.path.join(out_dir, cfg.name, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        np.save(os.path.join(run_dir, "q.npy"), agent.Q)
        np.savez_compressed(os.path.join(run_dir, "history.npz"), **hist)
        with open(os.path.join(run_dir, "eval.json"), "w") as f:
            json.dump(result, f, indent=2)
        config = {**cfg.to_dict(), "seed": seed, "actions": ACTION_NAMES,
                  "n_states": disc.n_states, "params": params, "git_commit": git_commit()}
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


PRESETS = {"ablation": ablation_configs, "discretization": discretization_configs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="ablation")
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--out", default="notebooks/artifacts/runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None, help="override the learning rate")
    ap.add_argument("--only", nargs="*", help="run only these config names (before suffixes)")
    args = ap.parse_args()
    for cfg in PRESETS[args.preset](args.episodes):
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
