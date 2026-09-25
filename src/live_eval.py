"""Evaluate a policy on the live server and on the simulator over the same fixed starts.

Doubles as the Milestone 1 long-run stability check: every episode records
reset failures, missed commands, stale see messages and play-mode changes.
Results are written incrementally, so an interrupted run keeps what it did.

    python -m src.live_eval --policy greedy --episodes 500
    python -m src.live_eval --policy notebooks/artifacts/runs/<run>/seed_0 --episodes 200
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from src.agents import GreedyQPolicy
from src.controllers import GreedyPursuit, rollout
from src.discretizer import Discretizer, DiscretizerConfig
from src.live_env import LiveBallPursuitEnv, ResetError
from src.sampler import DISTANCE_DISTRIBUTIONS, evaluation_starts
from src.sim_env import SimBallPursuitEnv


def policy_cycles_per_step(spec: str) -> int:
    if spec == "greedy":
        return 1
    with open(os.path.join(spec, "config.json")) as f:
        return int(json.load(f).get("cycles_per_step", 1))


def load_policy(spec: str, params: Dict[str, Any]):
    if spec == "greedy":
        return GreedyPursuit(params)
    with open(os.path.join(spec, "config.json")) as f:
        cfg = json.load(f)
    disc = Discretizer(DiscretizerConfig.from_dict(cfg["discretizer"]))
    return GreedyQPolicy(np.load(os.path.join(spec, "q.npy")), disc)


def summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    done = [e for e in episodes if "error" not in e]
    if not done:
        return {"n": 0}
    steps = np.array([e["capture_step"] or 10**6 for e in done])
    n = len(done)
    out = {"n": n, "mean_return": float(np.mean([e["return"] for e in done])),
           "mean_steps": float(np.mean([e["steps"] for e in done])),
           "mean_cycles": float(np.mean([e["cycles"] for e in done]))}
    for label, budget in {"<40": 39, "<=40": 40}.items():
        p = float(np.mean(steps <= budget))
        out[f"capture_rate_{label}"] = {"rate": p, "stderr": float(np.sqrt(p * (1 - p) / n))}
    return out


def episode_record(i: int, res: Dict[str, Any], keep_trajectory: bool) -> Dict[str, Any]:
    cycles = res["trajectory"][-1].get("cycles")
    rec = {"index": i, "captured": res["captured"], "capture_step": res["capture_step"],
           "steps": res["steps"], "cycles": cycles, "return": res["return"],
           "start": res["start"].as_dict(),
           "actions": res["actions"]}
    if keep_trajectory:
        rec["trajectory"] = [{k: t[k] for k in ("step", "cycle", "player", "body_dir", "ball",
                                                "true_d", "true_theta", "ball_status")}
                             for t in res["trajectory"]]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="greedy", help="'greedy' or a run directory with q.npy")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--start-seed", type=int, default=2026, help="seed of the fixed start list")
    ap.add_argument("--keep-trajectories", type=int, default=20)
    ap.add_argument("--distance-dist", default="uniform", choices=sorted(DISTANCE_DISTRIBUTIONS) + ["kit"],
                    help="start law: d0 law over [5, 40] m (uniform = implemented baseline) "
                         "or 'kit' (the starter kit's reset, d0 about 11-19 m)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = "greedy" if args.policy == "greedy" else os.path.basename(os.path.normpath(args.policy))
    out = args.out or f"notebooks/artifacts/live_eval_{name}_{args.episodes}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    starts = evaluation_starts(args.episodes, seed=args.start_seed, law=args.distance_dist)
    k = policy_cycles_per_step(args.policy)

    env = LiveBallPursuitEnv.connect(cycles_per_step=k)
    policy = load_policy(args.policy, env.params)
    sim = SimBallPursuitEnv(env.params, seed=args.start_seed, cycles_per_step=k)
    result: Dict[str, Any] = {"policy": args.policy, "start_seed": args.start_seed,
                              "distance_dist": args.distance_dist, "cycles_per_step": k,
                              "episodes_requested": args.episodes, "live": [], "sim": [],
                              "mode_changes": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save():
        result["summary"] = {"live": summarize(result["live"]), "sim": summarize(result["sim"]),
                             "stats": dict(env.stats), "server_errors": list(env.player.errors)}
        with open(out, "w") as f:
            json.dump(result, f)

    t0 = time.monotonic()
    try:
        for i, start in enumerate(starts):
            keep = i < args.keep_trajectories
            result["sim"].append(episode_record(i, rollout(sim, policy, start), keep))
            try:
                result["live"].append(episode_record(i, rollout(env, policy, start), keep))
            except (ResetError, TimeoutError) as exc:
                result["live"].append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
            if env.player.play_mode not in (None, "play_on"):
                result["mode_changes"].append({"episode": i, "mode": env.player.play_mode})
            if (i + 1) % 25 == 0:
                save()
                s = result["summary"]["live"]
                print(f"[{i + 1}/{args.episodes}] {time.monotonic() - t0:.0f}s "
                      f"live <40: {s.get('capture_rate_<40', {}).get('rate', float('nan')):.3f} "
                      f"sim <40: {result['summary']['sim']['capture_rate_<40']['rate']:.3f} "
                      f"stats={env.stats}", flush=True)
    finally:
        result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        env.close()
    print(json.dumps(result["summary"], indent=2))
    print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
