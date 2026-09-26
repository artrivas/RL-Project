"""Goal Shooting (Definición y tiro a puerta) with kit-style simple kinematics.

The attacker stands on the ball in front of the rival goal (goal line x = 52.5, posts at
y = +/-7.01, +x toward the goal, +y to the attacker's right as on the rcssserver monitor).

* Start: distance to the goal line ``d ~ U[11, 25]`` m, lateral position ``y ~ U[-10, 10]`` m.
  A goalkeeper is present with probability ``p_keeper``, standing on the goal line at
  ``y_k ~ U[-3, 3]`` m (evaluation uses the "open" and "keeper" variants separately).
* Actions: shoot at one of five target points on the goal line (``TARGETS``: left post, left,
  centre, right, right post) or ``CONDUCIR`` (ball and player move ``CONDUCIR_STEP`` along +x;
  the distance never goes below ``D_MIN``, where CONDUCIR only costs time).
* Shot: the direction to the target gets an angular error ``eps ~ N(0, sigma_deg^2)``, so the
  lateral error at the goal line grows with the real distance to the target. The ball crosses
  the line at ``y_c = y + d * tan(phi + eps)``. Off target if ``|y_c| >= POST_Y``; saved if a
  keeper is present and ``|y_c - y_k| < keeper_reach``; otherwise goal. A shot ends the episode.
* With a keeper, each CONDUCIR ends the episode with probability ``p_keeper_out`` (the keeper
  comes out and blocks): this is the cost that keeps walking to ``D_MIN`` from being free.
* Reward: +100 goal, -30 off target, -50 saved or blocked, -0.2 per CONDUCIR.

Parameters are ``ShootingParams``; their values were fixed by the calibration check in
``src/shooting_optimal.py`` before any training: the provisional (4 deg, 2.0 m, p_out 0.1)
failed check (b) (CONDUCIR never optimal), and the pre-defined grid selected p_out = 0.02
(``notebooks/artifacts/shooting_calibration.json``). Full observability: the policy sees
``{"d_g", "y", "keeper"}`` (keeper = its y, or None).
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

GOAL_X = 52.5
POST_Y = 7.01
TARGETS = (-6.0, -3.5, 0.0, 3.5, 6.0)
CONDUCIR = len(TARGETS)                      # action id of CONDUCIR
ACTION_NAMES = ["TIRO y=-6", "TIRO y=-3.5", "TIRO y=0", "TIRO y=+3.5", "TIRO y=+6", "CONDUCIR"]
CONDUCIR_STEP = 2.0
D_MIN, D_MAX = 11.0, 25.0
Y_MAX = 10.0
KEEPER_Y_MAX = 3.0
R_GOAL, R_OFF, R_SAVED, R_CONDUCIR = 100.0, -30.0, -50.0, -0.2
T_MAX = 10

# outcome codes (history / metrics)
RUNNING, GOAL, OFF_TARGET, SAVED, BLOCKED = 0, 1, 2, 3, 4
OUTCOME_NAMES = {RUNNING: "sin tiro (truncado)", GOAL: "gol", OFF_TARGET: "afuera",
                 SAVED: "atajado", BLOCKED: "bloqueado al conducir"}


@dataclass(frozen=True)
class ShootingParams:
    sigma_deg: float = 4.0        # std of the angular shot error
    keeper_reach: float = 2.0     # m, keeper saves if |y_c - y_k| < reach
    p_keeper_out: float = 0.02    # per CONDUCIR, with a keeper present (calibrated; see below)
    p_keeper: float = 0.5         # probability that a training start has a keeper

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ShootStart:
    d: float                      # m, ball to goal line
    y: float                      # m, lateral position of the ball
    keeper_y: Optional[float]     # m, None = open goal

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sample_start(rng: np.random.Generator, keeper: Optional[bool] = None,
                 p_keeper: float = 0.5) -> ShootStart:
    """``keeper``: True / False forces the variant; None draws it with ``p_keeper``.
    Always consumes the same number of random draws."""
    d = float(rng.uniform(D_MIN, D_MAX))
    y = float(rng.uniform(-Y_MAX, Y_MAX))
    has = float(rng.random()) < p_keeper
    ky = float(rng.uniform(-KEEPER_Y_MAX, KEEPER_Y_MAX))
    if keeper is not None:
        has = keeper
    return ShootStart(d, y, ky if has else None)


def evaluation_starts(n: int, seed: int, variant: str = "keeper") -> list:
    """Fixed starts per variant: "open" (no keeper), "keeper" (always), "mixed" (training law)."""
    rng = np.random.default_rng(seed)
    force = {"open": False, "keeper": True, "mixed": None}[variant]
    return [sample_start(rng, force) for _ in range(n)]


def crossing_y(d: float, y: float, target: float, eps_deg: float) -> Optional[float]:
    """Lateral position where the shot crosses the goal line (None if it never reaches it)."""
    phi = math.atan2(target - y, d) + math.radians(eps_deg)
    if abs(phi) >= math.pi / 2:
        return None
    return y + d * math.tan(phi)


def shot_outcome(y_c: Optional[float], keeper_y: Optional[float], reach: float) -> int:
    if y_c is None or abs(y_c) >= POST_Y:
        return OFF_TARGET
    if keeper_y is not None and abs(y_c - keeper_y) < reach:
        return SAVED
    return GOAL


class ShootingEnv:
    def __init__(self, t_max: int = T_MAX, seed: Optional[int] = None,
                 params: Optional[ShootingParams] = None, **param_overrides):
        self.params = params or ShootingParams(**param_overrides)
        self.t_max = t_max
        self.rng = np.random.default_rng(seed)
        self.step_count = self.cycle_count = 0
        self._done = True

    def _observe(self):
        obs = {"d_g": self.d, "y": self.y, "keeper": self.keeper_y}
        info = {"step": self.step_count, "d_g": self.d, "y": self.y,
                "ball": (GOAL_X - self.d, self.y), "keeper": self.keeper_y,
                "outcome": self.outcome, "shot_end": self.shot_end,
                "n_conducir": self.n_conducir}
        return obs, info

    def reset(self, start: Optional[ShootStart] = None):
        start = start or sample_start(self.rng, None, self.params.p_keeper)
        self.d, self.y, self.keeper_y = start.d, start.y, start.keeper_y
        self.step_count = self.cycle_count = 0
        self.outcome, self.shot_end, self.n_conducir = RUNNING, None, 0
        self._done = False
        obs, info = self._observe()
        info["start"] = start
        return obs, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        p = self.params
        if action == CONDUCIR:
            reward = R_CONDUCIR
            self.n_conducir += 1
            if self.keeper_y is not None and float(self.rng.random()) < p.p_keeper_out:
                self.outcome = BLOCKED
                reward += R_SAVED
            else:
                self.d = max(D_MIN, self.d - CONDUCIR_STEP)
        else:
            eps = float(self.rng.normal(0.0, p.sigma_deg))
            y_c = crossing_y(self.d, self.y, TARGETS[action], eps)
            self.outcome = shot_outcome(y_c, self.keeper_y, p.keeper_reach)
            self.shot_end = (GOAL_X, y_c) if y_c is not None else None
            reward = {GOAL: R_GOAL, OFF_TARGET: R_OFF, SAVED: R_SAVED}[self.outcome]
        self.step_count += 1
        self.cycle_count += 1
        terminated = self.outcome != RUNNING
        truncated = (not terminated) and self.step_count >= self.t_max
        self._done = terminated or truncated
        obs, info = self._observe()
        info["success"] = self.outcome == GOAL
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------ representation
def default_discretizer():
    """36 states: d_g {11-15, 15-20, 20-25} x lateral {left, centre, right}
    x keeper {none, left, centre, right}."""
    from src.task_discretizer import Feature, ProductDiscretizer
    return ProductDiscretizer([
        Feature("d_g", "bins", (15.0, 20.0), ("11–15 m", "15–20 m", "20–25 m")),
        Feature("y", "bins", (-3.5, 3.5), ("izq.", "centro", "der.")),
        Feature("keeper", "bins", (-1.0, 1.0), ("izq.", "centro", "der."), none_label="sin arquero"),
    ])


def task_spec(TaskSpec):
    def summarize(out, info):
        return {"success": info["outcome"] == GOAL, "outcome": info["outcome"],
                "n_conducir": info["n_conducir"]}

    def extra(results):
        outcomes = np.array([r["outcome"] for r in results])
        return {"goal_rate": float(np.mean(outcomes == GOAL)),
                "off_target_rate": float(np.mean(outcomes == OFF_TARGET)),
                "saved_rate": float(np.mean(outcomes == SAVED)),
                "blocked_rate": float(np.mean(outcomes == BLOCKED)),
                "no_shot_rate": float(np.mean(outcomes == RUNNING)),
                "mean_conducir": float(np.mean([r["n_conducir"] for r in results]))}

    return TaskSpec(
        name="shooting", action_names=list(ACTION_NAMES),
        make_env=lambda seed, t_max=T_MAX, **kw: ShootingEnv(t_max=t_max, seed=seed, **kw),
        representations={"default": default_discretizer},
        eval_starts=lambda n, seed, variant: evaluation_starts(n, seed, variant),
        eval_variants=["open", "keeper"], summarize=summarize,
        history_keys=["outcome", "n_conducir"], extra_metrics=extra,
        trajectory_keys=["step", "ball", "keeper", "shot_end", "outcome"],
        default_t_max=T_MAX)
