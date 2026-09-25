"""State discretization (``src/SPEC.md`` §5) — single source of truth for training and reporting.

A state is (distance bin, angle bin, speed bin), plus one extra ``UNKNOWN``
state for a ball that has not been acquired yet. Capture is terminal and has
no state index.

* Distance: ``distance_edges`` are interior edges; bins are half-open
  ``[e_i, e_{i+1})``, the first bin also holds estimates below its upper edge
  (an estimate can read < 0.8 m before true capture), the last is unbounded.
* Angle: ``angle_edges`` are positive, increasing, ending at 180. Bin 0 is
  frontal ``|theta| <= e_0``; each further edge adds a right bin
  ``(e_{i-1}, e_i]`` and a left bin ``[-e_i, -e_{i-1})``. Boundaries belong to
  the bin nearer the front; ``theta`` is wrapped to (-180, 180] first.
* Speed: ``speed_edges`` interior edges on the sense_body speed (empty = no
  speed feature).
* Optional ``angle_edges_moving``: angle edges used in every speed bin above the
  first (the turn quantum shrinks with speed, so finer heading bins are only
  reachable while moving). Must have as many edges as ``angle_edges``; ``None``
  (default) uses ``angle_edges`` at all speeds.
"""

import bisect
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from src.perception import UNKNOWN, Observation
from src.sampler import wrap_deg


@dataclass(frozen=True)
class DiscretizerConfig:
    # Chosen representation "C" (see train.discretization_configs and SPEC §5).
    distance_edges: Tuple[float, ...] = (3.0, 10.0, 20.0)
    angle_edges: Tuple[float, ...] = (17.5, 90.0, 180.0)
    speed_edges: Tuple[float, ...] = (0.2,)
    angle_edges_moving: Optional[Tuple[float, ...]] = None

    def as_dict(self) -> dict:
        return {k: (list(v) if v is not None else None) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "DiscretizerConfig":
        """Inverse of ``as_dict``; also reads configs saved before optional fields existed."""
        return cls(**{k: (tuple(v) if v is not None else None) for k, v in d.items()})


class Discretizer:
    def __init__(self, config: DiscretizerConfig = DiscretizerConfig()):
        if list(config.distance_edges) != sorted(config.distance_edges):
            raise ValueError("distance_edges must be increasing")
        for edges in (config.angle_edges, config.angle_edges_moving):
            if edges is None:
                continue
            edges = list(edges)
            if edges != sorted(edges) or edges[-1] != 180.0 or edges[0] <= 0.0:
                raise ValueError("angle edges must be positive, increasing and end at 180")
        if (config.angle_edges_moving is not None
                and len(config.angle_edges_moving) != len(config.angle_edges)):
            raise ValueError("angle_edges_moving must have as many edges as angle_edges")
        self.config = config
        self.n_dist = len(config.distance_edges) + 1
        self.n_angle = 2 * len(config.angle_edges) - 1
        self.n_speed = len(config.speed_edges) + 1
        self.unknown_state = self.n_dist * self.n_angle * self.n_speed
        self.n_states = self.unknown_state + 1

    def distance_bin(self, d: float) -> int:
        return bisect.bisect_right(self.config.distance_edges, d)

    def angle_bin(self, theta: float, speed_bin: int = 0) -> int:
        """0 = front; then (right_1, left_1, right_2, left_2, ...) moving backwards."""
        theta = wrap_deg(theta)
        edges = self.config.angle_edges
        if speed_bin > 0 and self.config.angle_edges_moving is not None:
            edges = self.config.angle_edges_moving
        ring = bisect.bisect_left(edges, abs(theta))
        if ring == 0:
            return 0
        return 2 * ring - 1 if theta > 0 else 2 * ring

    def speed_bin(self, speed: float) -> int:
        return bisect.bisect_right(self.config.speed_edges, speed)

    def __call__(self, obs: Observation) -> int:
        if obs.status == UNKNOWN:
            return self.unknown_state
        v = self.speed_bin(obs.speed)
        return ((self.distance_bin(obs.d) * self.n_angle + self.angle_bin(obs.theta, v))
                * self.n_speed + v)

    # ------------------------------------------------------------- labels
    def angle_labels(self) -> List[str]:
        e = self.config.angle_edges
        labels = [f"front |θ|≤{e[0]:g}"]
        for i in range(1, len(e)):
            labels += [f"right ({e[i-1]:g},{e[i]:g}]", f"left [-{e[i]:g},-{e[i-1]:g})"]
        return labels

    def distance_labels(self) -> List[str]:
        e = self.config.distance_edges
        return ([f"d<{e[0]:g}"] + [f"[{a:g},{b:g})" for a, b in zip(e, e[1:])]
                + [f"d≥{e[-1]:g}"])

    def state_to_label(self, state: int) -> str:
        if state == self.unknown_state:
            return "UNKNOWN"
        rest, s = divmod(state, self.n_speed)
        d, a = divmod(rest, self.n_angle)
        label = f"{self.distance_labels()[d]} | {self.angle_labels()[a]}"
        return label + (f" | speed bin {s}" if self.n_speed > 1 else "")

    def decompose(self, state: int) -> Tuple[int, int, int]:
        """(distance bin, angle bin, speed bin); raises for UNKNOWN."""
        if state == self.unknown_state:
            raise ValueError("UNKNOWN has no bins")
        rest, s = divmod(state, self.n_speed)
        d, a = divmod(rest, self.n_angle)
        return d, a, s
