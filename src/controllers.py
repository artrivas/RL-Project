"""Scripted controllers and a rollout helper (sanity checks and live feasibility reference).

``GreedyPursuit`` is the baseline "search, turn toward the ball, dash"
controller, restricted to the task's four actions:

* ball UNKNOWN -> TURN +35 (search);
* otherwise, turn toward the ball when the straight line would miss the
  capture radius (or the ball is behind) *and* a turn at the current speed,
  ``35 / (1 + inertia_moment * speed)``, reduces |theta|; else DASH 100.
"""

import math
from typing import Any, Callable, Dict, List, Optional

from src.env_base import BallPursuitEnvBase, CAPTURE_RADIUS
from src.perception import SEEN, UNKNOWN, Observation
from src.sampler import StartConfig

DASH_100, DASH_50, TURN_POS, TURN_NEG = 0, 1, 2, 3

Policy = Callable[[Observation], int]


class GreedyPursuit:
    def __init__(self, params: Dict[str, Any]):
        self.inertia = params["inertia_moment"]

    def __call__(self, obs: Observation) -> int:
        if obs.status == UNKNOWN:
            return TURN_POS
        turn = 35.0 / (1.0 + self.inertia * obs.speed)
        misses = abs(obs.theta) > 90.0 or obs.d * math.sin(math.radians(abs(obs.theta))) > CAPTURE_RADIUS
        if misses and turn < 2.0 * abs(obs.theta):
            return TURN_POS if obs.theta > 0 else TURN_NEG
        return DASH_100


def rollout(env: BallPursuitEnvBase, policy: Policy, start: Optional[StartConfig] = None,
            use_truth: bool = False) -> Dict[str, Any]:
    """Run one episode. ``use_truth`` feeds the policy true (d, theta) — diagnostics only."""
    obs, info = env.reset(start)
    trajectory: List[Dict[str, Any]] = [info]
    actions: List[int] = []
    ret, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        if use_truth:
            obs = Observation(info["true_d"], info["true_theta"], SEEN, obs.speed, obs.step)
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        trajectory.append(info)
        ret += reward
    return {"return": ret, "steps": env.step_count, "captured": terminated,
            "capture_step": env.step_count if terminated else None,
            "start": trajectory[0]["start"],
            "actions": actions, "trajectory": trajectory}
