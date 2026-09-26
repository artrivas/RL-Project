"""2v1 Cooperation and Passing (Cooperación 2v1 y pase) with kit-style simple kinematics.

A 30 x 20 m zone centred at the origin (x in [-15, 15], y in [-10, 10]); angles in degrees, positive
from +x toward +y (``src/SPEC.md`` §1). The agent controls whichever attacker holds the ball; the
other attacker and the defender are scripted.

* Start: holder uniform in the inner 24 x 14 m, random heading; teammate 6-10 m from the holder
  and defender 4-8 m from the holder, both at random angles inside the zone (rejection sampling).
* Actions:
  - ``PASE``: resolved within the step. The pass line is *blocked* if the defender is closer than
    ``BLOCK_DIST`` to the holder-teammate segment; a blocked pass is intercepted with probability
    ``P_INTERCEPT`` (-30, terminal). Otherwise it is completed (+30): the teammate becomes the
    holder, facing the pass direction, and the previous holder becomes the scripted teammate.
  - ``DRIBLE``: the holder (with the ball) moves ``DRIBBLE_STEP`` m along its heading.
  - ``GIRAR``: the holder turns 35 deg in the direction that increases the bearing to the defender;
    if the defender is already within 35 deg of straight behind, it turns toward the zone centre.
  - ``DESPEJE``: clears the ball; the episode ends with reward 0.
* After the action: the teammate moves up to ``MATE_SPEED`` toward its support point (``SUPPORT``
  m from the holder, at +/-60 deg from the defender-to-holder direction, on the teammate's side,
  clipped into the zone); the defender moves ``DEF_SPEED`` toward the holder with heading noise
  ``N(0, DEF_NOISE^2)``. Then: holder outside the zone -> -10, terminal; defender closer than
  ``TACKLE_DIST`` -> tackle with probability ``P_TACKLE`` (-30, terminal).
* ``gamma = 0.99``, ``T_MAX = 60``. Success: at least ``SUCCESS_POSSESSION`` steps of possession
  (brief: more than 50) and at least ``SUCCESS_PASSES`` completed passes.

Known property of the brief's reward: an unblocked pass is risk-free and earns +30, so passing
whenever the line is open is expected to be optimal. The defender chases the holder, which tends to
put it on the pass line after each pass, so the agent must dribble or turn to reopen it.

The policy sees ``{"d_comp", "theta_comp", "d_def", "theta_def", "blocked", "edge", "theta_centre"}``
(distances, bearings relative to the holder's heading, the blocked flag, the distance to the nearest
zone edge and the bearing to the zone centre; the default representation does not use the last one).
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.sampler import wrap_deg

HALF_X, HALF_Y = 15.0, 10.0
ACTION_NAMES = ["PASE", "DRIBLE", "GIRAR", "DESPEJE"]
PASE, DRIBLE, GIRAR, DESPEJE = range(4)
DRIBBLE_STEP, TURN = 1.0, 35.0
MATE_SPEED, DEF_SPEED, DEF_NOISE = 0.6, 0.6, 15.0
SUPPORT, SUPPORT_ANGLE = 8.0, 60.0
BLOCK_DIST, P_INTERCEPT = 1.5, 0.8
TACKLE_DIST, P_TACKLE = 1.0, 0.5
R_PASS, R_LOSS, R_OUT = 30.0, -30.0, -10.0
T_MAX = 60
SUCCESS_POSSESSION, SUCCESS_PASSES = 51, 3

RUNNING, INTERCEPTED, TACKLED, OUT, CLEARED = range(5)
OUTCOME_NAMES = {RUNNING: "posesión hasta el final", INTERCEPTED: "intercepción", TACKLED: "quite",
                 OUT: "fuera de la zona", CLEARED: "despeje"}


@dataclass(frozen=True)
class PassStart:
    holder: Tuple[float, float]
    heading: float
    mate: Tuple[float, float]
    defender: Tuple[float, float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inside(p, margin: float = 0.0) -> bool:
    return abs(p[0]) <= HALF_X - margin and abs(p[1]) <= HALF_Y - margin


def _around(rng: np.random.Generator, c, d_min: float, d_max: float):
    while True:
        d = float(rng.uniform(d_min, d_max))
        a = math.radians(float(rng.uniform(-180.0, 180.0)))
        p = (c[0] + d * math.cos(a), c[1] + d * math.sin(a))
        if inside(p, 0.5):
            return p


def sample_start(rng: np.random.Generator) -> PassStart:
    holder = (float(rng.uniform(-12.0, 12.0)), float(rng.uniform(-7.0, 7.0)))
    heading = wrap_deg(float(rng.uniform(-180.0, 180.0)))
    return PassStart(holder, heading, _around(rng, holder, 6.0, 10.0), _around(rng, holder, 4.0, 8.0))


def evaluation_starts(n: int, seed: int, variant: str = "default") -> list:
    rng = np.random.default_rng(seed)
    return [sample_start(rng) for _ in range(n)]


def seg_dist(p, a, b) -> float:
    """Distance from point p to segment ab."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    L2 = ax * ax + ay * ay
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L2))
    return math.hypot(p[0] - (a[0] + t * ax), p[1] - (a[1] + t * ay))


def _bearing(frm, to, heading) -> float:
    return wrap_deg(math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0])) - heading)


def _move_toward(p, target, step):
    dx, dy = target[0] - p[0], target[1] - p[1]
    d = math.hypot(dx, dy)
    if d <= step:
        return (target[0], target[1])
    return (p[0] + step * dx / d, p[1] + step * dy / d)


def support_point(holder, defender, mate):
    """``SUPPORT`` m from the holder at +/-``SUPPORT_ANGLE`` from the defender-to-holder
    direction, the side closer to the teammate, clipped into the zone (0.5 m margin)."""
    ux, uy = holder[0] - defender[0], holder[1] - defender[1]
    base = math.atan2(uy, ux) if (ux or uy) else 0.0
    best = None
    for sign in (1.0, -1.0):
        a = base + sign * math.radians(SUPPORT_ANGLE)
        c = (holder[0] + SUPPORT * math.cos(a), holder[1] + SUPPORT * math.sin(a))
        c = (max(-HALF_X + 0.5, min(HALF_X - 0.5, c[0])), max(-HALF_Y + 0.5, min(HALF_Y - 0.5, c[1])))
        d = math.hypot(c[0] - mate[0], c[1] - mate[1])
        if best is None or d < best[0]:
            best = (d, c)
    return best[1]


class PassingEnv:
    def __init__(self, t_max: int = T_MAX, seed: Optional[int] = None,
                 p_tackle: float = P_TACKLE, p_intercept: float = P_INTERCEPT):
        self.t_max = t_max
        self.p_tackle, self.p_intercept = p_tackle, p_intercept
        self.rng = np.random.default_rng(seed)
        self.step_count = self.cycle_count = 0
        self._done = True

    def blocked(self) -> bool:
        return seg_dist(self.defender, self.holder, self.mate) < BLOCK_DIST

    def _observe(self):
        obs = {"d_comp": math.hypot(self.mate[0] - self.holder[0], self.mate[1] - self.holder[1]),
               "theta_comp": _bearing(self.holder, self.mate, self.heading),
               "d_def": math.hypot(self.defender[0] - self.holder[0], self.defender[1] - self.holder[1]),
               "theta_def": _bearing(self.holder, self.defender, self.heading),
               "blocked": 1.0 if self.blocked() else 0.0,
               "edge": min(HALF_X - abs(self.holder[0]), HALF_Y - abs(self.holder[1])),
               "theta_centre": _bearing(self.holder, (0.0, 0.0), self.heading)}
        info = {"step": self.step_count, "holder": self.holder, "heading": self.heading,
                "mate": self.mate, "defender": self.defender, "ball": self.holder,
                "passes": self.passes, "possession": self.possession, "outcome": self.outcome,
                "blocked": bool(obs["blocked"]), "event": self.event}
        return obs, info

    def reset(self, start: Optional[PassStart] = None):
        start = start or sample_start(self.rng)
        self.holder, self.heading = start.holder, start.heading
        self.mate, self.defender = start.mate, start.defender
        self.step_count = self.cycle_count = 0
        self.passes, self.possession = 0, 0
        self.outcome, self.event = RUNNING, None
        self._done = False
        obs, info = self._observe()
        info["start"] = start
        return obs, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        reward, self.event = 0.0, None
        if action == PASE:
            if self.blocked() and float(self.rng.random()) < self.p_intercept:
                self.outcome, self.event, reward = INTERCEPTED, "intercepción", R_LOSS
            else:
                self.heading = wrap_deg(math.degrees(math.atan2(self.mate[1] - self.holder[1],
                                                                self.mate[0] - self.holder[0])))
                self.holder, self.mate = self.mate, self.holder
                self.passes += 1
                self.event, reward = "pase", R_PASS
        elif action == DRIBLE:
            h = math.radians(self.heading)
            self.holder = (self.holder[0] + DRIBBLE_STEP * math.cos(h),
                           self.holder[1] + DRIBBLE_STEP * math.sin(h))
        elif action == GIRAR:
            theta_def = _bearing(self.holder, self.defender, self.heading)
            if abs(theta_def) < 180.0 - TURN:
                self.heading = wrap_deg(self.heading + (-TURN if theta_def > 0 else TURN))
            else:
                # Defender already within one turn of straight behind: "away" is undefined (it
                # would flip sides every step), so turn toward the zone centre instead.
                theta_c = _bearing(self.holder, (0.0, 0.0), self.heading)
                self.heading = wrap_deg(self.heading + (TURN if theta_c > 0 else -TURN))
        else:
            self.outcome, self.event = CLEARED, "despeje"

        if self.outcome == RUNNING:
            self.mate = _move_toward(self.mate, support_point(self.holder, self.defender, self.mate),
                                     MATE_SPEED)
            noise = math.radians(float(self.rng.normal(0.0, DEF_NOISE)))
            a = math.atan2(self.holder[1] - self.defender[1], self.holder[0] - self.defender[0]) + noise
            d = math.hypot(self.holder[0] - self.defender[0], self.holder[1] - self.defender[1])
            step = min(DEF_SPEED, d)
            self.defender = (self.defender[0] + step * math.cos(a), self.defender[1] + step * math.sin(a))
            if not inside(self.holder):
                self.outcome, self.event = OUT, "fuera"
                reward += R_OUT
            elif (math.hypot(self.holder[0] - self.defender[0], self.holder[1] - self.defender[1])
                  < TACKLE_DIST and float(self.rng.random()) < self.p_tackle):
                self.outcome, self.event = TACKLED, "quite"
                reward += R_LOSS
        self.step_count += 1
        self.cycle_count += 1
        if self.outcome == RUNNING:
            self.possession = self.step_count
        terminated = self.outcome != RUNNING
        truncated = (not terminated) and self.step_count >= self.t_max
        self._done = terminated or truncated
        obs, info = self._observe()
        info["success"] = self.possession >= SUCCESS_POSSESSION and self.passes >= SUCCESS_PASSES
        return obs, reward, terminated, truncated, info


def heuristic_action(obs: Dict[str, float]) -> int:
    """Reference (not a bound): pass when the line is open; otherwise turn away from a defender in
    front, and dribble when it is behind (turning when close to the zone edge)."""
    if not obs["blocked"]:
        return PASE
    if abs(obs["theta_def"]) < 90.0 or obs["edge"] < 1.5:
        return GIRAR
    return DRIBLE


# ------------------------------------------------------------------ representation
def default_discretizer():
    """324 states: d_comp {< 5, 5-10, >= 10} x theta_comp {front +/-45, side, behind |theta| > 135}
    x d_def {< 2, 2-5, >= 5} x theta_def (same) x pass line {open, blocked} x zone {touchline
    (< 3 m from an edge), centre}."""
    from src.task_discretizer import Feature, ProductDiscretizer
    ang = dict(kind="angle", edges=(45.0, 135.0, 180.0), labels=("frente", "lado", "atrás"),
               split=(False, False, False))
    return ProductDiscretizer([
        Feature("d_comp", "bins", (5.0, 10.0), ("< 5 m", "5–10 m", "≥ 10 m")),
        Feature("theta_comp", **ang),
        Feature("d_def", "bins", (2.0, 5.0), ("< 2 m", "2–5 m", "≥ 5 m")),
        Feature("theta_def", **ang),
        Feature("blocked", "bins", (0.5,), ("libre", "bloqueada")),
        Feature("edge", "bins", (3.0,), ("banda", "centro")),
    ])


def fine_discretizer():
    """1 944 states (representation diagnostic): the default features with finer defender distance
    {< 1.5, 1.5-3, 3-5, >= 5} and edge distance {< 1.5, 1.5-3, >= 3}, plus the bearing to the zone
    centre {front +/-45, side, behind} (which way is away from the edge)."""
    from src.task_discretizer import Feature, ProductDiscretizer
    ang = dict(kind="angle", edges=(45.0, 135.0, 180.0), labels=("frente", "lado", "atrás"),
               split=(False, False, False))
    return ProductDiscretizer([
        Feature("d_comp", "bins", (5.0, 10.0), ("< 5 m", "5–10 m", "≥ 10 m")),
        Feature("theta_comp", **ang),
        Feature("d_def", "bins", (1.5, 3.0, 5.0), ("< 1.5 m", "1.5–3 m", "3–5 m", "≥ 5 m")),
        Feature("theta_def", **ang),
        Feature("blocked", "bins", (0.5,), ("libre", "bloqueada")),
        Feature("edge", "bins", (1.5, 3.0), ("< 1.5 m", "1.5–3 m", "centro")),
        Feature("theta_centre", **ang),
    ])


def task_spec(TaskSpec):
    def summarize(out, info):
        return {"success": info["success"], "outcome": info["outcome"], "passes": info["passes"],
                "possession": info["possession"]}

    def extra(results):
        outcomes = np.array([r["outcome"] for r in results])
        return {"intercepted_rate": float(np.mean(outcomes == INTERCEPTED)),
                "tackled_rate": float(np.mean(outcomes == TACKLED)),
                "out_rate": float(np.mean(outcomes == OUT)),
                "cleared_rate": float(np.mean(outcomes == CLEARED)),
                "full_possession_rate": float(np.mean(outcomes == RUNNING)),
                "mean_passes": float(np.mean([r["passes"] for r in results])),
                "mean_possession": float(np.mean([r["possession"] for r in results]))}

    return TaskSpec(
        name="passing", action_names=list(ACTION_NAMES),
        make_env=lambda seed, t_max=T_MAX, **kw: PassingEnv(t_max=t_max, seed=seed, **kw),
        representations={"default": default_discretizer, "fine": fine_discretizer},
        eval_starts=lambda n, seed, variant: evaluation_starts(n, seed, variant),
        eval_variants=["default"], summarize=summarize,
        history_keys=["outcome", "passes", "possession"], extra_metrics=extra,
        trajectory_keys=["step", "holder", "heading", "mate", "defender", "passes", "event", "outcome"],
        default_t_max=T_MAX)
