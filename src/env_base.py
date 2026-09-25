"""Episode logic shared by the simulator and the live environment (``src/SPEC.md`` §2, §6, §7).

Backends implement three hooks:

* ``_backend_reset(start)`` -> ``(body, see, late_see, truth)`` for the first decision cycle;
* ``_backend_step(command)`` -> ``(body, see, late_see, truth)`` for the next cycle;
* ``params`` — physics parameters (parsed from the server, or a snapshot).

``see`` is this cycle's see received before the decision deadline; ``late_see``
is the previous cycle's see that missed its deadline (both may be None).
``truth`` is ``{"px", "py", "body", "bx", "by", "cycle"}`` (global frame, privileged):
it is used for rewards, metrics and trajectories only, never for the policy.

Macro-actions (alternative reading of "paso", ``cycles_per_step`` = k > 1): one
decision repeats its command for k consecutive server cycles (the server executes
one body command per cycle, so a held TURN keeps turning). The estimator updates
and capture is checked on every cycle, so an episode ends on the exact capture
cycle. The reward follows the brief per decision step: distance gained over the
macro-step, minus 0.2, plus 100 on capture. ``t_max`` counts decision steps;
``cycle_count`` counts server cycles. k = 1 is the implemented baseline and runs
exactly the same code path as before.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.perception import BallEstimator, Observation
from src.sampler import StartConfig, sample_start, wrap_deg

# id -> (command, argument); SPEC §2
ACTIONS: List[Tuple[str, float]] = [("dash", 100.0), ("dash", 50.0), ("turn", 35.0), ("turn", -35.0)]
ACTION_NAMES = ["DASH 100", "DASH 50", "TURN +35", "TURN -35"]
N_ACTIONS = len(ACTIONS)

CAPTURE_RADIUS = 0.8
STEP_PENALTY = 0.2
CAPTURE_BONUS = 100.0
T_MAX = 40


def true_polar(truth: Dict[str, float]) -> Tuple[float, float]:
    dx, dy = truth["bx"] - truth["px"], truth["by"] - truth["py"]
    return math.hypot(dx, dy), wrap_deg(math.degrees(math.atan2(dy, dx)) - truth["body"])


class BallPursuitEnvBase:
    params: Dict[str, Any]

    def __init__(self, t_max: int = T_MAX, seed: Optional[int] = None, cycles_per_step: int = 1):
        if cycles_per_step < 1:
            raise ValueError("cycles_per_step must be >= 1")
        self.t_max = t_max
        self.cycles_per_step = cycles_per_step
        self.cycle_count = 0
        self.rng = np.random.default_rng(seed)
        self.estimator = BallEstimator(self.params)
        self.step_count = 0
        self._last_command: Optional[tuple] = None
        self._d = float("nan")
        self._done = True

    # ---------------------------------------------------------------- hooks
    def _backend_reset(self, start: StartConfig):
        raise NotImplementedError

    def _backend_step(self, command: Tuple[str, float]):
        raise NotImplementedError

    # ---------------------------------------------------------------- API
    def reset(self, start: Optional[StartConfig] = None) -> Tuple[Observation, Dict[str, Any]]:
        start = start or sample_start(self.rng)
        self.estimator.reset()
        self.step_count = 0
        self.cycle_count = 0
        self._last_command = None
        body, see, _late, truth = self._backend_reset(start)  # pre-reset sightings are dropped
        self.estimator.update(body, see, None)
        self._d, theta = true_polar(truth)
        self._done = False
        return self._obs(body), self._info(truth, theta, start=start)

    def step(self, action: int) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        command = ACTIONS[action]
        for _ in range(self.cycles_per_step):
            body, see, late_see, truth = self._backend_step(command)
            self.cycle_count += 1
            self.estimator.update(body, see, command, late_see)
            d_new, theta = true_polar(truth)
            captured = d_new <= CAPTURE_RADIUS
            if captured:
                break
        self.step_count += 1
        self._last_command = command

        reward = (self._d - d_new) - STEP_PENALTY + (CAPTURE_BONUS if captured else 0.0)
        self._d = d_new
        terminated = captured
        truncated = (not captured) and self.step_count >= self.t_max
        self._done = terminated or truncated
        info = self._info(truth, theta, captured=captured)
        return self._obs(body), reward, terminated, truncated, info

    # ------------------------------------------------------------- helpers
    def _obs(self, body: Dict[str, Any]) -> Observation:
        return self.estimator.observation(float(body.get("speed") or 0.0), self.step_count)

    def _info(self, truth: Dict[str, float], theta: float, **extra) -> Dict[str, Any]:
        info = {"true_d": self._d, "true_theta": theta, "step": self.step_count,
                "cycle": truth.get("cycle"), "player": (truth["px"], truth["py"]),
                "body_dir": truth["body"], "ball": (truth["bx"], truth["by"]),
                "ball_status": self.estimator.status, "cycles": self.cycle_count,
                "cycles_since_seen": self.estimator.cycles_since_seen}
        info.update(extra)
        return info
