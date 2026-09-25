"""Episode start distribution, shared by the simulator and the live environment.

Relative configuration first, absolute placement second:

1. d ~ U[d_min, d_max], relative bearing ~ U[-180, 180), body heading ~ U[-180, 180).
2. The ball offset (dx, dy) = d * (cos, sin)(heading + bearing) is then fixed.
3. The player position is uniform on the rectangle where both the player and
   the ball lie inside the pitch shrunk by ``margin``: the intersection of that
   rectangle with a copy shifted by -(dx, dy).

Step 3 never rejects, so the marginals of d, bearing and heading are exactly
the uniform ones in step 1; only the absolute position depends on them. The
intersection is non-empty whenever |dx| <= 2(HALF_LENGTH - margin) and
|dy| <= 2(HALF_WIDTH - margin), which holds for d_max = 40 and margin <= 14.

Angles follow the server's global frame (see ``src/SPEC.md``): positive angles
rotate from +x toward +y.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

HALF_LENGTH = 52.5  # pitch is 105 m x 68 m, centred at the origin
HALF_WIDTH = 34.0


def wrap_deg(angle: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = math.fmod(angle + 180.0, 360.0)
    if a <= 0.0:
        a += 360.0
    return a - 180.0


@dataclass(frozen=True)
class StartConfig:
    distance: float      # m, player centre to ball centre
    bearing: float       # deg, ball direction relative to body heading
    heading: float       # deg, body direction in the global frame
    player_x: float
    player_y: float
    ball_x: float
    ball_y: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


DISTANCE_DISTRIBUTIONS = {
    "uniform": "uniform in distance on [d_min, d_max] (default)",
    "area": "uniform over the annulus area: density proportional to d (more far starts)",
    "near": "triangular with mode at d_min: density proportional to (d_max - d) (more near starts)",
}


def _draw_distance(rng: np.random.Generator, d_min: float, d_max: float, kind: str) -> float:
    if kind == "uniform":
        return float(rng.uniform(d_min, d_max))
    if kind == "area":
        return float(math.sqrt(rng.uniform(d_min ** 2, d_max ** 2)))
    if kind == "near":
        return float(rng.triangular(d_min, d_min, d_max))
    raise ValueError(f"unknown distance distribution {kind!r}; options: {sorted(DISTANCE_DISTRIBUTIONS)}")


def sample_start(rng: np.random.Generator, d_min: float = 5.0, d_max: float = 40.0,
                 margin: float = 1.0, distance_dist: str = "uniform") -> StartConfig:
    """Draw one start. ``distance_dist`` changes only the distance law (see
    ``DISTANCE_DISTRIBUTIONS``); the default consumes the same random draws as always."""
    distance = _draw_distance(rng, d_min, d_max, distance_dist)
    bearing = wrap_deg(float(rng.uniform(-180.0, 180.0)))
    heading = wrap_deg(float(rng.uniform(-180.0, 180.0)))
    direction = math.radians(heading + bearing)
    dx, dy = distance * math.cos(direction), distance * math.sin(direction)

    lx, ly = HALF_LENGTH - margin, HALF_WIDTH - margin
    x_lo, x_hi = max(-lx, -lx - dx), min(lx, lx - dx)
    y_lo, y_hi = max(-ly, -ly - dy), min(ly, ly - dy)
    if x_lo > x_hi or y_lo > y_hi:
        raise ValueError(f"no feasible placement for d={distance:.2f}, margin={margin}")
    px, py = float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi))
    return StartConfig(distance, bearing, heading, px, py, px + dx, py + dy)


def kit_start(rng: np.random.Generator) -> StartConfig:
    """The starter kit's own reset (``BallPursuitSimEnv.reset`` in
    ``agente_cero_mc_control_persecucion.ipynb``): player at (-15 + U(-2, 2), U(-4, 4)) with
    heading U(-180, 180), ball at (U(-2, 2), U(-3, 3)); d0 is therefore about 11-19 m."""
    px = -15.0 + float(rng.uniform(-2.0, 2.0))
    py = float(rng.uniform(-4.0, 4.0))
    heading = wrap_deg(float(rng.uniform(-180.0, 180.0)))
    bx = float(rng.uniform(-2.0, 2.0))
    by = float(rng.uniform(-3.0, 3.0))
    dx, dy = bx - px, by - py
    bearing = wrap_deg(math.degrees(math.atan2(dy, dx)) - heading)
    return StartConfig(math.hypot(dx, dy), bearing, heading, px, py, bx, by)


def draw_start(rng: np.random.Generator, law: str = "uniform", **kwargs) -> StartConfig:
    """``law``: a key of ``DISTANCE_DISTRIBUTIONS`` (over [5, 40] m) or ``"kit"`` (kit reset)."""
    if law == "kit":
        return kit_start(rng)
    return sample_start(rng, distance_dist=law, **kwargs)


def evaluation_starts(n: int, seed: int, law: str = "uniform", **kwargs) -> list:
    """Fixed, reproducible list of starts used for every evaluation condition."""
    rng = np.random.default_rng(seed)
    return [draw_start(rng, law, **kwargs) for _ in range(n)]
