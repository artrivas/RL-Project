"""Exact minimum number of steps to capture in the starter-kit environment (A* search).

The kit environment (``src/kit_env.py``) is deterministic, so for each start the
best possible capture time over *all* action sequences can be computed exactly:

* state: player position and heading index k (heading = phi0 + 35 k; since 35 and
  360 share a factor 5, k mod 72 gives the 72 reachable headings);
* actions: DASH 100 (1.0 m), DASH 50 (0.5 m), TURN +35, TURN -35; capture is checked
  at the end of each step, exactly as in the environment;
* admissible heuristic: while the ball is behind (|bearing| > 90 deg) no step reduces
  the distance, so ceil((|bearing| - 90) / 35) turns are needed first; afterwards each
  step shortens the distance by at most 1.0 m;
* duplicate states are merged after rounding the position to 1e-6 m.

The search stops at ``budget`` steps (39 = "fewer than 40"). If the node cap is hit
the start is reported as undetermined (None), never guessed.

    python -m src.kit_optimal --n 1000
"""

import argparse
import heapq
import json
import math
from typing import Dict, List, Optional

import numpy as np

from src.env_base import CAPTURE_RADIUS
from src.sampler import StartConfig, evaluation_starts, wrap_deg

STEP = {0: 1.0, 1: 0.5}
UNDETERMINED = "undetermined"


def _heuristic(px: float, py: float, heading: float, bx: float, by: float) -> int:
    dx, dy = bx - px, by - py
    d = math.hypot(dx, dy)
    if d <= CAPTURE_RADIUS:
        return 0
    bearing = abs(wrap_deg(math.degrees(math.atan2(dy, dx)) - heading))
    turns = math.ceil(max(0.0, bearing - 90.0) / 35.0 - 1e-12)
    return turns + math.ceil(d - CAPTURE_RADIUS - 1e-12)


def min_steps(start: StartConfig, budget: int = 39, max_nodes: int = 400_000):
    """Minimum capture step for ``start`` (int), None if impossible within ``budget``,
    or ``UNDETERMINED`` if the node cap was reached."""
    bx, by, phi0 = start.ball_x, start.ball_y, start.heading
    h0 = _heuristic(start.player_x, start.player_y, phi0, bx, by)
    if h0 > budget:
        return None
    frontier = [(h0, 0, start.player_x, start.player_y, 0)]
    seen = set()
    nodes = 0
    while frontier:
        f, g, px, py, k = heapq.heappop(frontier)
        key = (round(px, 6), round(py, 6), k % 72)
        if key in seen:
            continue
        seen.add(key)
        nodes += 1
        if nodes > max_nodes:
            return UNDETERMINED
        heading = phi0 + 35.0 * k
        for a in (0, 1, 2, 3):
            if a in STEP:
                rad = math.radians(heading)
                nx, ny, nk = px + STEP[a] * math.cos(rad), py + STEP[a] * math.sin(rad), k
            else:
                nx, ny, nk = px, py, k + (1 if a == 2 else -1)
            ng = g + 1
            if math.hypot(bx - nx, by - ny) <= CAPTURE_RADIUS:
                return ng       # A* with an admissible heuristic: first goal popped is optimal,
                                # and a goal generated at depth ng cannot be beaten (h >= 1 elsewhere)
            nh = _heuristic(nx, ny, phi0 + 35.0 * nk, bx, by)
            if ng + nh <= budget:
                heapq.heappush(frontier, (ng + nh, ng, nx, ny, nk))
    return None


def summarize(results: List) -> Dict:
    det = [r for r in results if r != UNDETERMINED]
    n_und = len(results) - len(det)
    caught = [r for r in det if r is not None]
    out = {"n": len(results), "undetermined": n_und,
           "optimal_capture_rate_lt40": len(caught) / len(det) if det else float("nan")}
    # Bounds that treat undetermined starts as failures / successes.
    out["rate_bounds_lt40"] = [len(caught) / len(results), (len(caught) + n_und) / len(results)]
    out["mean_optimal_steps"] = float(np.mean(caught)) if caught else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--laws", nargs="*", default=["uniform", "kit"])
    ap.add_argument("--out", default="notebooks/artifacts/kit_optimal.json")
    args = ap.parse_args()
    out = {"method": __doc__.splitlines()[0], "n": args.n, "seed": args.seed, "laws": {}}
    for law in args.laws:
        starts = evaluation_starts(args.n, seed=args.seed, law=law)
        res = [min_steps(s) for s in starts]
        out["laws"][law] = {"summary": summarize(res),
                            "per_start": [{"distance": s.distance, "bearing": s.bearing,
                                           "min_steps": (r if r != UNDETERMINED else "undetermined")}
                                          for s, r in zip(starts, res)]}
        print(law, out["laws"][law]["summary"], flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
