"""Fast in-process Ball Pursuit simulator, built from rcssserver parameters.

Per-cycle model (rcssserver order), one body command per cycle:

* ``turn m``: body += m * (1 + U(-r, r)) / (1 + inertia_moment * |v|), no acceleration.
* ``dash p``: acceleration p * dash_power_rate * effort along the body direction.
* Movement: v += a; |v| capped at player_speed_max; v += U(-r*|v|, r*|v|) per axis;
  x += v; v *= player_decay.  (r = player_rand; noise switchable.)

Sensing reproduces what the player receives (SPEC §3-§4):

* ``sense_body`` every cycle: speed (0.01 m) and its direction relative to the
  head (1 deg), head angle.
* ``see`` at the ``send_step`` cadence (150 ms in normal/high view -> 2 of every
  3 cycles) with a random phase per episode. A see arriving later than
  ``see_deadline_ms`` into its cycle misses that cycle's decision and is
  delivered as ``late_see`` in the next cycle, as in the live environment
  (measured live: arrival 0/50 ms or 41/91 ms after sense_body, per session). The ball is named ``(b)`` inside
  the view cone (``visible_angle``), unnamed ``(B)`` within
  ``visible_distance`` outside it, absent otherwise. Distance is quantized as
  rcssserver does for movable objects; directions are rounded to 1 deg.

Simplifications, to be checked against the live server
(``notebooks/00_environment_validation.ipynb``): the ball is stationary and
player-ball collisions are ignored (collisions can only happen in the capture
cycle); effort stays at ``effort_max`` (stamina is recovered at every reset and
40 cycles of DASH 100 stay above the effort-decrease threshold with default
parameters); the neck stays at 0.
"""

import math
from typing import Any, Dict, Optional, Tuple

from src.env_base import BallPursuitEnvBase, T_MAX
from src.params import DEFAULT_PARAMS
from src.sampler import StartConfig, wrap_deg

_EPS = 1e-10


def _quantize(value: float, q: float) -> float:
    return round(value / q) * q


class SimBallPursuitEnv(BallPursuitEnvBase):
    def __init__(self, params: Optional[Dict[str, Any]] = None, t_max: int = T_MAX,
                 seed: Optional[int] = None, motion_noise: bool = True,
                 sensor_noise: bool = True, see_deadline_ms: float = 60.0,
                 see_phase_ms: Optional[float] = None):
        self.params = dict(params or DEFAULT_PARAMS)
        self.see_deadline_ms = see_deadline_ms
        self.see_phase_ms = see_phase_ms  # None: random phase per episode
        self.motion_noise = motion_noise
        self.sensor_noise = sensor_noise
        super().__init__(t_max=t_max, seed=seed)

    # ------------------------------------------------------------ physics
    def _backend_reset(self, start: StartConfig):
        self.px, self.py = start.player_x, start.player_y
        self.vx = self.vy = 0.0
        self.body = start.heading
        self.neck = 0.0
        self.bx, self.by = start.ball_x, start.ball_y
        self.cycle = 0
        self._see_phase_ms = (float(self.rng.uniform(0.0, self.params["send_step"]))
                              if self.see_phase_ms is None else self.see_phase_ms)
        self._late_see = None
        body, see, truth = self._sense()
        return body, see, None, truth

    def _backend_step(self, command: Tuple[str, float]):
        p, rand = self.params, self.params["player_rand"]
        speed = math.hypot(self.vx, self.vy)
        ax = ay = 0.0
        kind, arg = command
        if kind == "turn":
            moment = min(max(arg, p["minmoment"]), p["maxmoment"])
            if self.motion_noise:
                moment *= 1.0 + self.rng.uniform(-rand, rand)
            self.body = wrap_deg(self.body + moment / (1.0 + p["inertia_moment"] * speed))
        elif kind == "dash":
            power = min(max(arg, p["min_dash_power"]), p["max_dash_power"])
            accel = power * p["dash_power_rate"] * p.get("effort_max", 1.0)
            ax = accel * math.cos(math.radians(self.body))
            ay = accel * math.sin(math.radians(self.body))
        else:
            raise ValueError(f"unknown command {command}")

        self.vx += ax
        self.vy += ay
        sp = math.hypot(self.vx, self.vy)
        if sp > p["player_speed_max"]:
            self.vx *= p["player_speed_max"] / sp
            self.vy *= p["player_speed_max"] / sp
        if self.motion_noise:
            r = rand * math.hypot(self.vx, self.vy)
            self.vx += self.rng.uniform(-r, r)
            self.vy += self.rng.uniform(-r, r)
        self.px += self.vx
        self.py += self.vy
        self.vx *= p["player_decay"]
        self.vy *= p["player_decay"]
        self.cycle += 1
        late, self._late_see = self._late_see, None
        body, see, truth = self._sense()
        return body, see, late, truth

    # ------------------------------------------------------------ sensing
    def _see_offset_ms(self) -> Optional[float]:
        """Arrival offset of this cycle's see within the cycle, or None (phase + k * send_step)."""
        step, period = self.params["simulator_step"], self.params["send_step"]
        start = self.cycle * step
        k = math.ceil((start - self._see_phase_ms) / period)
        offset = self._see_phase_ms + k * period - start
        return offset if offset < step else None

    def _see_this_cycle(self) -> bool:
        return self._see_offset_ms() is not None

    def _sense(self):
        head_dir = self.body + self.neck
        speed = math.hypot(self.vx, self.vy)
        speed_dir = wrap_deg(math.degrees(math.atan2(self.vy, self.vx)) - head_dir) if speed > 0 else 0.0
        if self.sensor_noise:
            speed, speed_dir = _quantize(speed, 0.01), float(round(speed_dir))
        body = {"time": self.cycle, "speed": speed, "speed_dir": speed_dir,
                "head_angle": self.neck, "counts": {}}

        see = None
        offset = self._see_offset_ms()
        if offset is not None:
            see = {"time": self.cycle, "ball": None}
            dx, dy = self.bx - self.px, self.by - self.py
            d = math.hypot(dx, dy)
            direction = wrap_deg(math.degrees(math.atan2(dy, dx)) - head_dir)
            named = abs(direction) <= self.params["visible_angle"] / 2.0
            if named or d <= self.params["visible_distance"]:
                if self.sensor_noise:
                    d = _quantize(math.exp(_quantize(math.log(d + _EPS),
                                                     self.params["quantize_step"])), 0.1)
                    direction = float(round(direction))
                see["ball"] = {"named": named, "dist": d, "dir_head": direction,
                               "dist_chng": None, "dir_chng": None}

        if see is not None and offset > self.see_deadline_ms:
            self._late_see, see = see, None  # arrives after this cycle's decision
        truth = {"px": self.px, "py": self.py, "body": self.body,
                 "bx": self.bx, "by": self.by, "cycle": self.cycle}
        return body, see, truth
