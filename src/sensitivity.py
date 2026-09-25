"""Sensitivity of one fixed learned policy to evaluation assumptions (simulator only, no retraining).

Every variant evaluates the same Q-table (identified by the SHA-256 of ``q.npy``)
in the simulator with the live server's physics parameters. The variants change
one assumption at a time; they are *not* alternative readings of the brief unless
stated:

* ``distribution``: the shape of the d0 law over the stated range [5, 40] m
  (uniform in distance = our choice; uniform in area; triangular favouring near starts).
* ``subset``: d0 restricted to [5, 30] m. An easier subset, not the stated range.
* ``budget``: capture within fewer than 40 / 45 / 50 / 60 cycles, measured on the
  same episodes (run with a 60-cycle cap; the state has no clock, so the first
  40 cycles are identical to a 40-cycle run, which is checked).
* ``information``: player sensing (as implemented); ball position given to the
  estimator once at reset (then normal sightings and odometry); true (d, theta) at
  every step (continuous privileged access).

    python -m src.sensitivity --run notebooks/artifacts/runs/qlearning_eps_decay_1.0_to_0.1/seed_4
"""

import argparse
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents import GreedyQPolicy
from src.discretizer import Discretizer, DiscretizerConfig
from src.env_base import BallPursuitEnvBase
from src.params import load_params
from src.perception import SEEN, Observation
from src.sampler import DISTANCE_DISTRIBUTIONS, StartConfig, sample_start
from src.sim_env import SimBallPursuitEnv
from src.train import source_fingerprint

BUDGETS = {"<40": 39, "<=40": 40, "<45": 44, "<50": 49, "<60": 59}
INFORMATION = {
    "sensing": "player sensing only: vision cone, late sightings, odometry (as implemented)",
    "once": "true ball position given to the estimator once, at reset; afterwards normal sensing",
    "truth": "true (d, theta) replaces the observation at every step (continuous privileged access)",
}


def run_episode(env: BallPursuitEnvBase, policy, start: StartConfig, information: str) -> Optional[int]:
    """Return the capture step, or None. ``information`` is a key of ``INFORMATION``."""
    obs, info = env.reset(start)
    if information == "once":
        d, th = info["true_d"], math.radians(info["true_theta"])
        env.estimator.bx, env.estimator.by = d * math.cos(th), d * math.sin(th)
        env.estimator.cycles_since_seen = 0
        obs = env.estimator.observation(obs.speed, obs.step)
    terminated = truncated = False
    while not (terminated or truncated):
        if information == "truth":
            obs = Observation(info["true_d"], info["true_theta"], SEEN, obs.speed, obs.step)
        obs, _, terminated, truncated, info = env.step(policy(obs))
    return env.step_count if terminated else None


def rates(steps: List[Optional[int]], budgets: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    arr = np.array([s if s is not None else 10**6 for s in steps])
    out = {}
    for label, b in budgets.items():
        p = float(np.mean(arr <= b))
        out[label] = {"rate": p, "stderr": math.sqrt(p * (1 - p) / len(arr))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="notebooks/artifacts/runs/qlearning_eps_decay_1.0_to_0.1/seed_4")
    ap.add_argument("--params", default="notebooks/artifacts/server_params.json")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=11, help="seed of the start lists and simulator noise")
    ap.add_argument("--out", default="notebooks/artifacts/sensitivity.json")
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    q_path = os.path.join(args.run, "q.npy")
    with open(q_path, "rb") as f:
        q_sha = hashlib.sha256(f.read()).hexdigest()
    disc = Discretizer(DiscretizerConfig.from_dict(cfg["discretizer"]))
    policy = GreedyQPolicy(np.load(q_path), disc)
    params = load_params(args.params)

    def starts(dist: str = "uniform", d_max: float = 40.0) -> List[StartConfig]:
        rng = np.random.default_rng(args.seed)
        return [sample_start(rng, d_max=d_max, distance_dist=dist) for _ in range(args.n)]

    def evaluate(start_list, information="sensing", t_max=40):
        # Noise seeded per episode (seed, index): episodes do not depend on how long earlier
        # ones ran, and every variant sees the same noise stream for the same episode index.
        env = SimBallPursuitEnv(params, t_max=t_max, seed=args.seed)
        steps = []
        for i, s in enumerate(start_list):
            env.rng = np.random.default_rng((args.seed, i))
            steps.append(run_episode(env, policy, s, information))
        return steps

    uniform = starts()
    far_share = lambda sl: float(np.mean([s.distance >= 30 for s in sl]))
    base_steps = evaluate(uniform)
    results: List[Dict[str, Any]] = []

    def add(group, variant, definition, start_list, steps, budgets=None):
        results.append({"group": group, "variant": variant, "definition": definition,
                        "n": len(steps), "share_d0_ge_30": far_share(start_list),
                        "rates": rates(steps, budgets or {"<40": 39, "<=40": 40})})
        r = results[-1]["rates"]
        print(f"[{group}] {variant}: " + ", ".join(f"{k} {v['rate']:.1%}" for k, v in r.items()), flush=True)

    add("baseline", "as implemented", "d0 uniform on [5, 40] m, player sensing, fewer than 40 cycles",
        uniform, base_steps)
    for dist in ("area", "near"):
        sl = starts(dist)
        add("distribution", dist, DISTANCE_DISTRIBUTIONS[dist] + " on [5, 40] m", sl, evaluate(sl))
    sl = starts("uniform", d_max=30.0)
    add("subset", "d0 in [5, 30] m", "uniform on [5, 30] m: an easier subset, NOT the stated range",
        sl, evaluate(sl))
    long_steps = evaluate(uniform, t_max=60)
    consistent = all((a if a is not None and a <= 39 else None) == (b if b is not None and b <= 39 else None)
                     for a, b in zip(base_steps, long_steps))
    add("budget", "cycle budgets", "same starts, 60-cycle cap; capture within the listed number of cycles",
        uniform, long_steps, BUDGETS)
    for info_mode in ("once", "truth"):
        add("information", info_mode, INFORMATION[info_mode], uniform, evaluate(uniform, info_mode))

    out = {"method": __doc__.splitlines()[0], "policy_run": args.run, "q_sha256": q_sha,
           "policy_config": {k: cfg[k] for k in ("algorithm", "alpha", "gamma", "n_episodes",
                                                 "schedule", "discretizer", "seed")},
           "params_file": args.params, "n": args.n, "seed": args.seed,
           "source_sha256": source_fingerprint(),
           "budget_run_matches_40_cycle_run": bool(consistent), "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"first 40 cycles identical with a 60-cycle cap: {consistent}")
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
