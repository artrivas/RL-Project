"""Ball estimator: body-relative (d, theta) of a stationary ball from player messages only.

The estimate lives in the current body frame (x forward, y toward positive
bearings; see ``src/SPEC.md`` §1). Each cycle, after the server has executed
the previous command:

1. Rotate by the body turn of the last command. The turn is not observed, so
   it is predicted as ``moment / (1 + inertia_moment * speed_prev)``; dash
   commands do not rotate the body.
2. Subtract the player's displacement. ``sense_body`` reports the velocity
   *after* decay, so the displacement of the last step is ``speed / decay`` in
   direction ``speed_dir + head_angle`` relative to the new body direction.
3. A sighting in this cycle's ``see`` replaces the estimate.

A ``see`` can arrive after the decision deadline of its own cycle (its phase
within the 100 ms cycle depends on the connection; measured live: ~0/50 ms in
one session, ~41/91 ms in another). Such a *late* sighting of cycle t-1 is
applied at cycle t *before* step 1, so the regular motion update carries it
into the current body frame.

Odometry errors (turn noise, speed quantization) accumulate while the ball is
not seen; ``cycles_since_seen`` lets callers judge staleness.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.sampler import wrap_deg

SEEN, TRACKED, UNKNOWN = "SEEN", "TRACKED", "UNKNOWN"


@dataclass(frozen=True)
class Observation:
    """Policy input for one decision step (no privileged information)."""
    d: Optional[float]       # estimated distance [m], None while UNKNOWN
    theta: Optional[float]   # estimated body-relative bearing [deg], None while UNKNOWN
    status: str              # SEEN / TRACKED / UNKNOWN
    speed: float             # own speed from sense_body [m/cycle]
    step: int                # decision step index within the episode


class BallEstimator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.bx: Optional[float] = None  # ball in body frame [m]
        self.by: Optional[float] = None
        self.cycles_since_seen: Optional[int] = None
        self.seen_now = False
        self._prev_speed = 0.0
        self._prev_head = 0.0

    @property
    def status(self) -> str:
        if self.bx is None:
            return UNKNOWN
        return SEEN if self.seen_now else TRACKED

    def _apply_sighting(self, see: Optional[Dict[str, Any]], head: float) -> bool:
        ball = see["ball"] if see else None
        if ball is None or ball["dist"] is None:
            return False
        bearing = math.radians(ball["dir_head"] + head)
        self.bx, self.by = ball["dist"] * math.cos(bearing), ball["dist"] * math.sin(bearing)
        return True

    def update(self, body: Dict[str, Any], see: Optional[Dict[str, Any]],
               last_command: Optional[tuple], late_see: Optional[Dict[str, Any]] = None) -> None:
        """Advance one cycle.

        ``body`` is this cycle's parsed ``sense_body``; ``see`` is this cycle's
        parsed ``see`` received before the decision deadline, or None;
        ``late_see`` is the previous cycle's ``see`` that arrived after its
        deadline, or None; ``last_command`` is ``("turn", moment)``,
        ``("dash", power)`` or None (first cycle after a reset).
        """
        speed = float(body.get("speed") or 0.0)
        head = float(body.get("head_angle") or 0.0)
        if last_command is not None and self._apply_sighting(late_see, self._prev_head):
            self.cycles_since_seen = 0
        if self.bx is not None and last_command is not None:
            if last_command[0] == "turn":
                dphi = last_command[1] / (1.0 + self.params["inertia_moment"] * self._prev_speed)
                c, s = math.cos(math.radians(-dphi)), math.sin(math.radians(-dphi))
                self.bx, self.by = c * self.bx - s * self.by, s * self.bx + c * self.by
            step = speed / self.params["player_decay"]
            direction = math.radians(float(body.get("speed_dir") or 0.0) + head)
            self.bx -= step * math.cos(direction)
            self.by -= step * math.sin(direction)
        self._prev_speed = speed
        self._prev_head = head

        self.seen_now = self._apply_sighting(see, head)
        if self.seen_now:
            self.cycles_since_seen = 0
        elif self.cycles_since_seen is not None:
            self.cycles_since_seen += 1

    def observation(self, speed: float, step: int) -> Observation:
        if self.bx is None:
            return Observation(None, None, UNKNOWN, speed, step)
        d = math.hypot(self.bx, self.by)
        theta = wrap_deg(math.degrees(math.atan2(self.by, self.bx)))
        return Observation(d, theta, self.status, speed, step)
