"""Ball Pursuit environment of the course starter kit (``workspace/agente_cero_mc_control_persecucion.ipynb``).

Kinematics exactly as in the starter kit's ``BallPursuitSimEnv``:

* ``DASH 100`` moves the player 1.0 m along its heading; ``DASH 50`` moves it 0.5 m;
* ``TURN +35`` / ``TURN -35`` rotate the heading by exactly +/-35 deg;
* no inertia, no noise; the ball is stationary;
* the agent observes the true (d, theta) every step (full observability).

Reward, capture radius, step cap and angle conventions are those of the rest of the
project (``src/env_base.py``): positive angles rotate from +x toward +y and
``TURN +35`` increases the heading, as in the starter kit.

Two start laws (``src/sampler.py``): ``sample_start`` (d0 over [5, 40] m, the range in
the brief) and ``kit_start`` (the starter kit's own reset: player near (-15, 0), ball
near the centre, so d0 is about 11-19 m).

The interface matches ``SimBallPursuitEnv`` (``reset(start)``, ``step(action)``,
``step_count``, ``cycle_count``, ``rng``), so training, evaluation and analysis code
work unchanged.
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.env_base import ACTIONS, CAPTURE_BONUS, CAPTURE_RADIUS, STEP_PENALTY, T_MAX, true_polar
from src.perception import SEEN, Observation
from src.sampler import StartConfig, sample_start, wrap_deg

KIT_DASH_STEP = {100.0: 1.0, 50.0: 0.5}   # metres per DASH, from the starter kit
KIT_TURN = 35.0                           # degrees per TURN, exact


class KitBallPursuitEnv:
    def __init__(self, t_max: int = T_MAX, seed: Optional[int] = None):
        self.t_max = t_max
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.cycle_count = 0
        self._done = True

    def _truth(self) -> Dict[str, float]:
        return {"px": self.px, "py": self.py, "body": self.body, "bx": self.bx, "by": self.by,
                "cycle": self.step_count}

    def _observe(self) -> Tuple[Observation, Dict[str, Any]]:
        truth = self._truth()
        d, theta = true_polar(truth)
        obs = Observation(d, theta, SEEN, 0.0, self.step_count)
        info = {"true_d": d, "true_theta": theta, "step": self.step_count, "cycle": self.step_count,
                "player": (self.px, self.py), "body_dir": self.body, "ball": (self.bx, self.by),
                "ball_status": SEEN, "cycles_since_seen": 0, "cycles": self.cycle_count}
        return obs, info

    def reset(self, start: Optional[StartConfig] = None):
        start = start or sample_start(self.rng)
        self.px, self.py, self.body = start.player_x, start.player_y, start.heading
        self.bx, self.by = start.ball_x, start.ball_y
        self.step_count = self.cycle_count = 0
        self._done = False
        obs, info = self._observe()
        self._d = info["true_d"]
        info["start"] = start
        return obs, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        kind, arg = ACTIONS[action]
        if kind == "dash":
            step = KIT_DASH_STEP[arg]
            self.px += step * math.cos(math.radians(self.body))
            self.py += step * math.sin(math.radians(self.body))
        else:
            self.body = wrap_deg(self.body + math.copysign(KIT_TURN, arg))
        self.step_count += 1
        self.cycle_count += 1
        obs, info = self._observe()
        d_new = info["true_d"]
        captured = d_new <= CAPTURE_RADIUS
        reward = (self._d - d_new) - STEP_PENALTY + (CAPTURE_BONUS if captured else 0.0)
        self._d = d_new
        truncated = (not captured) and self.step_count >= self.t_max
        self._done = captured or truncated
        info["captured"] = captured
        return obs, reward, captured, truncated, info
