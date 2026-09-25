"""Exploration schedules for the required ablation: constant vs. geometrically decaying epsilon."""

import math
from typing import Callable, Dict

Schedule = Callable[[int], float]


def constant_epsilon(eps: float) -> Schedule:
    return lambda episode: eps


def decaying_epsilon(eps_start: float, eps_min: float, decay: float) -> Schedule:
    """epsilon_k = max(eps_min, eps_start * decay**k)."""
    return lambda episode: max(eps_min, eps_start * decay ** episode)


def decay_reaching(eps_start: float, eps_min: float, episodes: int) -> float:
    """Decay factor such that eps_start * decay**episodes == eps_min."""
    return math.exp(math.log(eps_min / eps_start) / episodes)


def make_schedule(spec: Dict) -> Schedule:
    """``{"kind": "constant", "eps": 0.1}`` or
    ``{"kind": "decay", "eps_start": 1.0, "eps_min": 0.05, "decay": 0.9998}``."""
    if spec["kind"] == "constant":
        return constant_epsilon(spec["eps"])
    if spec["kind"] == "decay":
        return decaying_epsilon(spec["eps_start"], spec["eps_min"], spec["decay"])
    raise ValueError(f"unknown schedule {spec}")
