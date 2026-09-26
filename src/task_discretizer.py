"""Product discretizer for the multi-variable catalog tasks (shooting, dribbling, 2v1 passing).

An observation is a mapping ``{key: value}``. Each ``Feature`` maps one value to a bin, and a
state is the mixed-radix index of the feature bins (the first feature is the most significant).
It exposes the same API as ``src.discretizer.Discretizer``: ``__call__(obs) -> int``,
``n_states``, ``state_to_label`` and ``decompose``.

Feature kinds:

* ``"bins"``: interior ``edges``; bin ``i`` is ``[e_{i-1}, e_i)`` (``bisect_right``), the first
  and last bins are unbounded. ``none_label`` adds a leading bin for a ``None`` value (e.g. no
  goalkeeper); ``labels`` then name only the numeric bins (``bin_labels`` lists all).
* ``"angle"``: a signed angle in degrees, wrapped to (-180, 180]. ``edges`` are positive,
  increasing and end at 180; ring 0 is ``|theta| <= e_0`` (front). Ring ``i`` covers
  ``(e_{i-1}, e_i]`` and is split into right (theta > 0) and left bins when ``split[i]`` is true.
  Boundaries belong to the ring nearer the front, as in ``Discretizer``.
"""

import bisect
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from src.sampler import wrap_deg


@dataclass(frozen=True)
class Feature:
    key: str
    kind: str                       # "bins" or "angle"
    edges: Tuple[float, ...]
    labels: Tuple[str, ...]
    split: Tuple[bool, ...] = ()    # "angle" only: one flag per ring (ring 0 is never split)
    none_label: Optional[str] = None

    def __post_init__(self):
        if list(self.edges) != sorted(self.edges):
            raise ValueError(f"{self.key}: edges must be increasing")
        if self.kind == "angle":
            if not self.edges or self.edges[0] <= 0.0 or self.edges[-1] != 180.0:
                raise ValueError(f"{self.key}: angle edges must be positive and end at 180")
            if len(self.split) != len(self.edges) or self.split[0]:
                raise ValueError(f"{self.key}: one split flag per ring, ring 0 unsplit")
        elif self.kind != "bins":
            raise ValueError(f"{self.key}: unknown kind {self.kind!r}")
        if len(self.bin_labels) != self.n_bins:
            raise ValueError(f"{self.key}: {self.n_bins} bins but {len(self.bin_labels)} labels")

    @property
    def n_bins(self) -> int:
        if self.kind == "angle":
            return sum(2 if s else 1 for s in self.split)
        return len(self.edges) + 1 + (self.none_label is not None)

    @property
    def bin_labels(self) -> Tuple[str, ...]:
        """One label per bin, the ``None`` bin (if any) first."""
        return ((self.none_label,) if self.none_label is not None else ()) + tuple(self.labels)

    def bin(self, value: Any) -> int:
        if self.kind == "bins":
            offset = 0
            if self.none_label is not None:
                if value is None:
                    return 0
                offset = 1
            return offset + bisect.bisect_right(self.edges, value)
        theta = wrap_deg(value)
        ring = bisect.bisect_left(self.edges, abs(theta))
        index = sum(2 if s else 1 for s in self.split[:ring])
        return index + (1 if self.split[ring] and theta < 0 else 0)


class ProductDiscretizer:
    def __init__(self, features: Sequence[Feature]):
        keys = [f.key for f in features]
        if len(set(keys)) != len(keys):
            raise ValueError("feature keys must be unique")
        self.features = tuple(features)
        self.sizes = [f.n_bins for f in self.features]
        self.n_states = 1
        for n in self.sizes:
            self.n_states *= n

    def bins(self, obs: Mapping[str, Any]) -> Tuple[int, ...]:
        return tuple(f.bin(obs[f.key]) for f in self.features)

    def __call__(self, obs: Mapping[str, Any]) -> int:
        state = 0
        for f, n in zip(self.features, self.sizes):
            state = state * n + f.bin(obs[f.key])
        return state

    def compose(self, bins: Sequence[int]) -> int:
        state = 0
        for b, n in zip(bins, self.sizes):
            if not 0 <= b < n:
                raise ValueError(f"bin {b} out of range {n}")
            state = state * n + b
        return state

    def decompose(self, state: int) -> Tuple[int, ...]:
        if not 0 <= state < self.n_states:
            raise ValueError(f"state {state} out of range")
        out = []
        for n in reversed(self.sizes):
            state, b = divmod(state, n)
            out.append(b)
        return tuple(reversed(out))

    def state_to_label(self, state: int) -> str:
        return " | ".join(f"{f.key}={f.bin_labels[b]}"
                          for f, b in zip(self.features, self.decompose(state)))

    def as_dict(self) -> dict:
        return {"features": [{"key": f.key, "kind": f.kind, "edges": list(f.edges),
                              "labels": list(f.labels), "split": list(f.split),
                              "none_label": f.none_label} for f in self.features]}
