"""Greedy policies saved during training (Q snapshots), run on the live server and the simulator.

Shows how the 2D trajectory changes as the agent learns, on ``rcssserver`` itself:
the same few starts are replayed with the Q-table saved after 0, 500, ... episodes.

    python -m src.live_snapshots --run notebooks/artifacts/runs/qlearning_eps_decay_1.0_to_0.1/seed_4
"""

import argparse
import json
import os

import numpy as np

from src.agents import GreedyQPolicy
from src.controllers import rollout
from src.discretizer import Discretizer, DiscretizerConfig
from src.live_env import LiveBallPursuitEnv
from src.live_eval import episode_record
from src.sampler import evaluation_starts
from src.sim_env import SimBallPursuitEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with q_snapshots.npz")
    ap.add_argument("--indices", nargs="*", type=int, default=[44, 0, 5, 39],
                    help="indices into evaluation_starts(200, start_seed); default: ball ahead "
                         "at 20 m, right at 11 m, behind-left at 18 m, behind-right at 22 m")
    ap.add_argument("--start-seed", type=int, default=2026)
    ap.add_argument("--out", default="notebooks/artifacts/live_snapshots.json")
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    disc = Discretizer(DiscretizerConfig.from_dict(cfg["discretizer"]))
    snaps = np.load(os.path.join(args.run, "q_snapshots.npz"))
    keys = sorted(snaps.files, key=lambda k: int(k.split("_")[1]))
    pool = evaluation_starts(200, seed=args.start_seed)
    starts = [pool[i] for i in args.indices]

    live = LiveBallPursuitEnv.connect()
    sim = SimBallPursuitEnv(live.params, seed=args.start_seed)
    out = {"run": args.run, "start_seed": args.start_seed, "indices": args.indices, "snapshots": {}}
    try:
        for key in keys:
            policy = GreedyQPolicy(snaps[key], disc)
            out["snapshots"][key] = {
                "live": [episode_record(i, rollout(live, policy, s), True) for i, s in zip(args.indices, starts)],
                "sim": [episode_record(i, rollout(sim, policy, s), True) for i, s in zip(args.indices, starts)],
            }
            caps = [e["captured"] for e in out["snapshots"][key]["live"]]
            print(f"{key}: live captured {sum(caps)}/{len(caps)} "
                  f"steps {[e['steps'] for e in out['snapshots'][key]['live']]}", flush=True)
        out["stats"] = dict(live.stats)
    finally:
        live.close()
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
