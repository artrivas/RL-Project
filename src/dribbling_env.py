"""Ball Dribbling (Conducción y drible) with kit-style simple kinematics.

Pitch 105 x 68 m (x in [-52.5, 52.5], y in [-34, 34]), rival goal centre at (52.5, 0), angles in
degrees, positive from +x toward +y (``src/SPEC.md`` §1), ``TURN +35`` increases the heading.

* Start: player at ``x0 ~ U[-40, -10]``, ``y0 ~ U[-20, 20]``, heading ``U(-180, 180)``; ball
  ``BALL_START`` m ahead along the heading (the player starts in possession).
* Actions: ``KICK 25`` moves the ball ``KICK_STEP`` m along the player's heading, only if the ball
  is within ``POSSESSION`` m (otherwise it does nothing and costs a step); ``DASH 80`` moves the
  player ``DASH_STEP`` m along its heading; ``TURN +/-35`` rotates the heading exactly. No
  player-ball collision (the player can walk past the ball; a kick then sends the ball forward
  from where it is, as rcssserver allows kicks in any direction relative to the ball).
* Reward: ball advance in x (``Delta x_ball``) minus ``STEP_COST`` per step; ``-30`` and the episode
  ends if the ball gets farther than ``loss_radius`` from the player (lost control) or the ball or
  player leaves the pitch; ``+50`` and the episode ends when the ball has advanced ``GOAL_ADVANCE``
  m in x from its start (success).
* ``gamma = 0.99``, ``T_MAX = 100``. Deterministic; full observability. The policy sees
  ``{"d_b", "theta_b", "theta_g"}`` (ball distance and body-relative bearing, bearing to the
  goal centre).

Every start can reach the success condition within ``T_MAX`` (``scripted_action``: turn to the goal,
then kick / dash), so 100% success is attainable (constructive bound).
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.sampler import wrap_deg

HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
GOAL = (52.5, 0.0)
ACTION_NAMES = ["KICK 25", "DASH 80", "TURN +35", "TURN -35"]
KICK, DASH, TURN_POS, TURN_NEG = range(4)
KICK_STEP, DASH_STEP, TURN = 2.0, 0.8, 35.0
POSSESSION = 0.8
BALL_START = 0.5
GOAL_ADVANCE = 30.0
STEP_COST = 0.1
R_LOSS, R_SUCCESS = -30.0, 50.0
LOSS_RADIUS = 4.0
T_MAX = 100

RUNNING, SUCCESS, LOST, OUT = 0, 1, 2, 3
OUTCOME_NAMES = {RUNNING: "sin terminar (truncado)", SUCCESS: "avance de 30 m",
                 LOST: "pérdida (d_b > 4 m)", OUT: "fuera de la cancha"}


@dataclass(frozen=True)
class DribbleStart:
    x: float
    y: float
    heading: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def sample_start(rng: np.random.Generator) -> DribbleStart:
    return DribbleStart(float(rng.uniform(-40.0, -10.0)), float(rng.uniform(-20.0, 20.0)),
                        wrap_deg(float(rng.uniform(-180.0, 180.0))))


def evaluation_starts(n: int, seed: int, variant: str = "default") -> list:
    rng = np.random.default_rng(seed)
    return [sample_start(rng) for _ in range(n)]


def _bearing(fx, fy, tx, ty, heading) -> float:
    return wrap_deg(math.degrees(math.atan2(ty - fy, tx - fx)) - heading)


class DribblingEnv:
    def __init__(self, t_max: int = T_MAX, seed: Optional[int] = None,
                 loss_radius: float = LOSS_RADIUS):
        self.t_max = t_max
        self.loss_radius = loss_radius
        self.rng = np.random.default_rng(seed)
        self.step_count = self.cycle_count = 0
        self._done = True

    def _observe(self):
        d_b = math.hypot(self.bx - self.px, self.by - self.py)
        theta_b = _bearing(self.px, self.py, self.bx, self.by, self.body)
        theta_g = _bearing(self.px, self.py, GOAL[0], GOAL[1], self.body)
        obs = {"d_b": d_b, "theta_b": theta_b, "theta_g": theta_g}
        info = {"step": self.step_count, "player": (self.px, self.py), "body_dir": self.body,
                "ball": (self.bx, self.by), "d_b": d_b, "theta_b": theta_b, "theta_g": theta_g,
                "progress": self.bx - self.bx0, "outcome": self.outcome}
        return obs, info

    def reset(self, start: Optional[DribbleStart] = None):
        start = start or sample_start(self.rng)
        self.px, self.py, self.body = start.x, start.y, start.heading
        h = math.radians(self.body)
        self.bx = self.px + BALL_START * math.cos(h)
        self.by = self.py + BALL_START * math.sin(h)
        self.bx0 = self.bx
        self.step_count = self.cycle_count = 0
        self.outcome = RUNNING
        self._done = False
        obs, info = self._observe()
        info["start"] = start
        return obs, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        h = math.radians(self.body)
        bx_before = self.bx
        if action == KICK:
            if math.hypot(self.bx - self.px, self.by - self.py) <= POSSESSION:
                self.bx += KICK_STEP * math.cos(h)
                self.by += KICK_STEP * math.sin(h)
        elif action == DASH:
            self.px += DASH_STEP * math.cos(h)
            self.py += DASH_STEP * math.sin(h)
        else:
            self.body = wrap_deg(self.body + (TURN if action == TURN_POS else -TURN))
        self.step_count += 1
        self.cycle_count += 1

        reward = (self.bx - bx_before) - STEP_COST
        d_b = math.hypot(self.bx - self.px, self.by - self.py)
        outside = (abs(self.bx) > HALF_LENGTH or abs(self.by) > HALF_WIDTH
                   or abs(self.px) > HALF_LENGTH or abs(self.py) > HALF_WIDTH)
        if outside:
            self.outcome = OUT
        elif d_b > self.loss_radius:
            self.outcome = LOST
        elif self.bx - self.bx0 >= GOAL_ADVANCE:
            self.outcome = SUCCESS
        if self.outcome in (OUT, LOST):
            reward += R_LOSS
        elif self.outcome == SUCCESS:
            reward += R_SUCCESS
        terminated = self.outcome != RUNNING
        truncated = (not terminated) and self.step_count >= self.t_max
        self._done = terminated or truncated
        obs, info = self._observe()
        info["success"] = self.outcome == SUCCESS
        return obs, reward, terminated, truncated, info


def scripted_action(obs: Dict[str, float]) -> int:
    """Constructive controller. In possession: turn until the goal is within +/-17.5 deg (turning
    does not move the ball), then kick. Otherwise: turn until the ball is within +/-17.5 deg, then
    dash toward it. A kick sends the ball along the heading, so the chase is a straight line."""
    if obs["d_b"] <= POSSESSION:
        if abs(obs["theta_g"]) > TURN / 2:
            return TURN_POS if obs["theta_g"] > 0 else TURN_NEG
        return KICK
    if abs(obs["theta_b"]) > TURN / 2:
        return TURN_POS if obs["theta_b"] > 0 else TURN_NEG
    return DASH


# ------------------------------------------------------------------ representation
def default_discretizer():
    """36 states: d_b {<= 0.8, (0.8, 2], (2, 4]} x theta_b {front +/-17.5, right, left}
    x theta_g {front +/-17.5, right, left, behind |theta| > 90}. The distance to the goal is not
    in the state: it stays within about [32, 92] m and never changes the best action."""
    from src.task_discretizer import Feature, ProductDiscretizer
    return ProductDiscretizer([
        Feature("d_b", "bins", (0.8, 2.0), ("≤ 0.8 m", "0.8–2 m", "2–4 m"), right_closed=True),
        Feature("theta_b", "angle", (17.5, 180.0), ("frente", "der.", "izq."), split=(False, True)),
        Feature("theta_g", "angle", (17.5, 90.0, 180.0), ("frente", "der.", "izq.", "atrás"),
                split=(False, True, False)),
    ])


def fine_discretizer():
    """245 states (aliasing diagnostic): d_b {<= 0.8, 0.8-1.4, 1.4-2, 2-3, 3-4}, and theta_b,
    theta_g each with rings at 17.5 / 52.5 / 90 / 180 deg, every ring split into right and left
    (the side of the goal when it is behind the player is visible)."""
    from src.task_discretizer import Feature, ProductDiscretizer
    rings = (17.5, 52.5, 90.0, 180.0)
    labels = ("frente", "der. 17–52", "izq. 17–52", "der. 52–90", "izq. 52–90", "der. atrás", "izq. atrás")
    return ProductDiscretizer([
        Feature("d_b", "bins", (0.8, 1.4, 2.0, 3.0), ("≤ 0.8 m", "0.8–1.4 m", "1.4–2 m", "2–3 m", "3–4 m"),
                right_closed=True),
        Feature("theta_b", "angle", rings, labels, split=(False, True, True, True)),
        Feature("theta_g", "angle", rings, labels, split=(False, True, True, True)),
    ])


def task_spec(TaskSpec):
    def summarize(out, info):
        return {"success": info["outcome"] == SUCCESS, "outcome": info["outcome"],
                "progress": info["progress"]}

    def extra(results):
        outcomes = np.array([r["outcome"] for r in results])
        ok = [r["steps"] for r in results if r["outcome"] == SUCCESS]
        return {"lost_rate": float(np.mean(outcomes == LOST)),
                "out_rate": float(np.mean(outcomes == OUT)),
                "timeout_rate": float(np.mean(outcomes == RUNNING)),
                "mean_progress": float(np.mean([r["progress"] for r in results])),
                "mean_steps_success": float(np.mean(ok)) if ok else None}

    return TaskSpec(
        name="dribbling", action_names=list(ACTION_NAMES),
        make_env=lambda seed, t_max=T_MAX, **kw: DribblingEnv(t_max=t_max, seed=seed, **kw),
        representations={"default": default_discretizer, "fine": fine_discretizer},
        eval_starts=lambda n, seed, variant: evaluation_starts(n, seed, variant),
        eval_variants=["default"], summarize=summarize,
        history_keys=["outcome", "progress"], extra_metrics=extra,
        trajectory_keys=["step", "player", "body_dir", "ball", "d_b", "outcome"],
        default_t_max=T_MAX)
