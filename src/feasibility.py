"""Relaxed-turn, turn-then-dash timing estimate for Ball Pursuit.

This is an *estimate*, not an upper bound over all admissible policies. Model:

* The ball position is known from cycle 0 (no search), and there is no noise.
* RELAXED ACTION SET: the player turns in place by any angle up to 35 deg per
  cycle until its heading error is within asin(capture_radius / d). The task's
  action set only has TURN +/-35 exactly, so at rest reachable headings are
  quantized in 35 deg steps (residual error up to 17.5 deg) and fine alignment
  needs turns while moving, which rotate only 35 / (1 + inertia_moment * speed).
  This relaxation makes the estimate optimistic about turning; a realizable
  +/-35 controller in the same noiseless kinematics does markedly worse (see
  the baseline controller in the simulator).
* After turning, the player dashes at full power every cycle along a straight
  line: u = min(v + power*dash_power_rate*effort, player_speed_max); x += u;
  v = u * player_decay. Capture when the path reaches d - capture_radius.

Behaviour it does not model, which could beat it: coasting during turns,
dashing before the heading error is removed, favourable noise.

Starts are drawn by random sampling from ``src.sampler.sample_start`` (only the
relative distance/bearing matter). Reported with binomial standard errors.
"""

import argparse
import json
import math
from typing import Any, Dict, Optional

import numpy as np

from src.params import DEFAULT_PARAMS, load_params
from src.sampler import sample_start

CAPTURE_RADIUS = 0.8
TURN_MOMENT = 35.0


def dash_cycles(distance: float, params: Dict[str, Any], power: float = 100.0,
                max_cycles: int = 1000) -> int:
    """Cycles of full dashing from rest needed to travel ``distance`` metres."""
    accel = power * params["dash_power_rate"] * params.get("effort_max", 1.0)
    x = v = 0.0
    for n in range(1, max_cycles + 1):
        u = min(v + accel, params["player_speed_max"])
        x += u
        v = u * params["player_decay"]
        if x >= distance:
            return n
    return max_cycles


def turn_then_dash_cycles(distance: float, bearing: float, params: Dict[str, Any]) -> int:
    tolerance = math.degrees(math.asin(min(1.0, CAPTURE_RADIUS / distance)))
    turns = max(0, math.ceil((abs(bearing) - tolerance) / TURN_MOMENT))
    return turns + dash_cycles(distance - CAPTURE_RADIUS, params)


def estimate(n: int = 200_000, seed: int = 0, params: Optional[Dict[str, Any]] = None,
             budgets: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Success rates per budget label, e.g. ``{"<40": 39, "<=40": 40}`` (cycles)."""
    budgets = budgets or {"<40": 39, "<=40": 40}
    params = params or DEFAULT_PARAMS
    rng = np.random.default_rng(seed)
    starts = [sample_start(rng) for _ in range(n)]
    cycles = np.array([turn_then_dash_cycles(s.distance, s.bearing, params) for s in starts])
    distances = np.array([s.distance for s in starts])

    def rate(mask: np.ndarray, budget: int) -> Dict[str, float]:
        k = int(mask.sum())
        p = float(np.mean(cycles[mask] <= budget)) if k else float("nan")
        return {"rate": p, "stderr": math.sqrt(p * (1 - p) / k) if k else float("nan"), "n": k}

    everything = np.ones(n, dtype=bool)
    return {
        "method": "relaxed-turn turn-then-dash; random sampling of src.sampler.sample_start",
        "n": n, "seed": seed,
        "params_used": {k: params[k] for k in ("dash_power_rate", "player_decay",
                                               "player_speed_max", "effort_max")
                        if k in params},
        "capture_rate": {label: rate(everything, b) for label, b in budgets.items()},
        "capture_rate_lt40_by_min_distance": {
            f"d>={d}": rate(distances >= d, budgets.get("<40", 39)) for d in (30, 35, 38)},
        "dash_only_cycles_d40": dash_cycles(40.0 - CAPTURE_RADIUS, params),
    }


def controller_estimate(n: int = 20_000, seed: int = 0,
                        params: Optional[Dict[str, Any]] = None, use_truth: bool = False,
                        noise: bool = True) -> Dict[str, Any]:
    """Capture rates of the greedy +/-35 baseline controller in the simulator.

    ``use_truth=True`` feeds the controller true (d, theta) from cycle 0 (no
    search, no estimator); otherwise it sees only what the player sees.
    ``noise`` toggles both motion and sensor noise.
    """
    from src.controllers import GreedyPursuit, rollout
    from src.sim_env import SimBallPursuitEnv

    params = params or DEFAULT_PARAMS
    env = SimBallPursuitEnv(params, t_max=40, seed=seed, motion_noise=noise, sensor_noise=noise)
    policy = GreedyPursuit(params)
    rng = np.random.default_rng(seed)
    steps = np.array([r["capture_step"] or 10**6 for r in
                      (rollout(env, policy, sample_start(rng), use_truth=use_truth)
                       for _ in range(n))])
    out = {"method": "greedy +/-35 controller in SimBallPursuitEnv; random starts",
           "n": n, "seed": seed, "use_truth": use_truth, "noise": noise, "capture_rate": {}}
    for label, budget in {"<40": 39, "<=40": 40}.items():
        p = float(np.mean(steps <= budget))
        out["capture_rate"][label] = {"rate": p, "stderr": math.sqrt(p * (1 - p) / n)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", help="JSON params snapshot from the live server")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controller", action="store_true",
                    help="run the greedy +/-35 controller in the simulator instead")
    ap.add_argument("--use-truth", action="store_true", help="controller sees true (d, theta)")
    ap.add_argument("--no-noise", action="store_true", help="disable motion and sensor noise")
    args = ap.parse_args()
    params = load_params(args.params) if args.params else None
    if args.controller:
        result = controller_estimate(args.n, args.seed, params, args.use_truth, not args.no_noise)
    else:
        result = estimate(args.n, args.seed, params)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
