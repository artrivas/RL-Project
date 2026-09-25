"""Fixed action sequences on the live server vs. the simulator, from identical starts (Milestone 1.7).

Each sequence runs ``--repeats`` times live (noise is real) and in the simulator
with noise (same count) and without noise (the model's prediction). Per step we
record ground-truth position and body direction, so discrepancies can be
separated into model bias (live mean vs noiseless prediction) and noise spread.

    python -m src.sim_vs_live --repeats 5
"""

import argparse
import json
from typing import Any, Dict, List

import numpy as np

from src.controllers import DASH_100, DASH_50, TURN_NEG, TURN_POS
from src.live_env import LiveBallPursuitEnv
from src.sampler import StartConfig
from src.sim_env import SimBallPursuitEnv

SEQUENCES = {
    "dash100_straight": [DASH_100] * 15,
    "turn_at_rest": [TURN_POS] * 6,
    "turn_while_moving": [DASH_100] * 6 + [TURN_POS] * 4 + [DASH_100] * 4,
    "dash50_then_turn_neg": [DASH_50] * 6 + [TURN_NEG] * 3 + [DASH_50] * 3,
}
# Ball 30 m to the right (+y) so no sequence can capture it.
START = StartConfig(30.0, 90.0, 0.0, -30.0, -15.0, -30.0, 15.0)


def run(env, actions: List[int]) -> List[Dict[str, float]]:
    _, info = env.reset(START)
    rows = [info]
    for a in actions:
        _, _, term, trunc, info = env.step(a)
        rows.append(info)
        if term or trunc:
            break
    return [{"x": r["player"][0], "y": r["player"][1], "body": r["body_dir"]} for r in rows]


def circular_mean_deg(angles: np.ndarray, axis: int = 0) -> np.ndarray:
    rad = np.radians(angles)
    return np.degrees(np.arctan2(np.sin(rad).mean(axis), np.cos(rad).mean(axis)))


def summarize(entry: Dict[str, Any]) -> Dict[str, float]:
    """Model bias (live mean vs noiseless prediction) and live noise spread, per sequence."""
    lx = np.array([[p["x"] for p in r] for r in entry["live"]])
    ly = np.array([[p["y"] for p in r] for r in entry["live"]])
    ex = np.array([p["x"] for p in entry["sim_exact"]])
    ey = np.array([p["y"] for p in entry["sim_exact"]])
    bias = np.hypot(lx.mean(0) - ex, ly.mean(0) - ey)
    lb = np.array([[p["body"] for p in r] for r in entry["live"]])
    eb = np.array([p["body"] for p in entry["sim_exact"]])
    dbody = (circular_mean_deg(lb) - eb + 180.0) % 360.0 - 180.0
    spread_body = np.degrees(np.sqrt(-2 * np.log(np.hypot(np.cos(np.radians(lb[:, -1])).mean(),
                                                          np.sin(np.radians(lb[:, -1])).mean()))))
    return {"max_position_bias_m": float(bias.max()),
            "final_position_bias_m": float(bias[-1]),
            "max_heading_bias_deg": float(np.abs(dbody).max()),
            "live_final_spread_m": float(np.hypot(lx[:, -1].std(), ly[:, -1].std())),
            "live_final_heading_spread_deg": float(spread_body)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="notebooks/artifacts/sim_vs_live_sequences.json")
    ap.add_argument("--only", nargs="*", help="run only these sequences")
    args = ap.parse_args()

    live = LiveBallPursuitEnv.connect()
    params = live.params
    out: Dict[str, Any] = {"start": START.as_dict(), "repeats": args.repeats, "sequences": {}}
    try:
        for name, actions in SEQUENCES.items():
            if args.only and name not in args.only:
                continue
            sim_noisy = SimBallPursuitEnv(params, seed=7)
            sim_exact = SimBallPursuitEnv(params, motion_noise=False, sensor_noise=False)
            entry = {"actions": actions,
                     "live": [run(live, actions) for _ in range(args.repeats)],
                     "sim_noisy": [run(sim_noisy, actions) for _ in range(args.repeats)],
                     "sim_exact": run(sim_exact, actions)}
            entry["summary"] = summarize(entry)
            out["sequences"][name] = entry
            print(name, entry["summary"], flush=True)
        out["stats"] = dict(live.stats)
    finally:
        live.close()
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
