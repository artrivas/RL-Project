"""Live smoke check of the trainer-controlled rcssserver configuration.

Run inside the rl-agent container:
    docker compose -f docker/docker-compose.yml exec rl-agent python -m src.live_check

Checks, each printed as PASS/FAIL:
1. Trainer and player handshakes; server params snapshot saved to JSON.
2. Clock: sense_body time advances by one per cycle in play_on; see cadence.
3. Reset: trainer move + recover acknowledged; ground truth matches the request.
4. Turn sign (src/SPEC.md §1): TURN +35 reduces |bearing| of a ball at +20 deg.
5. Straight dash: per-cycle displacement vs. the kinematic model from parsed params.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List

from src.client import RoboCupPlayer
from src.params import save_params
from src.sampler import wrap_deg
from src.trainer import RoboCupTrainer

TEAM = "UTEC_RL"


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def truth_bearing(view: Dict[str, Any], unum: int) -> Dict[str, float]:
    p, b = view["players"][(TEAM, unum)], view["ball"]
    dx, dy = b["x"] - p["x"], b["y"] - p["y"]
    return {"d": math.hypot(dx, dy),
            "theta": wrap_deg(math.degrees(math.atan2(dy, dx)) - p["body"]),
            "speed": math.hypot(p["vx"], p["vy"]), "x": p["x"], "y": p["y"], "body": p["body"]}


def wait_truth(player: RoboCupPlayer, trainer: RoboCupTrainer, after: int) -> Dict[str, Any]:
    """Advance to the next cycle and return (body, ground truth for that cycle)."""
    body, _ = player.wait_for_sense_body(after_time=after, timeout=1.0)
    for _ in range(20):
        trainer.poll(0.02)
        view = trainer.truth_at(body["time"])
        if view is not None:
            return {"body": body, "view": view}
    raise TimeoutError(f"no see_global for cycle {body['time']}")


def reset(player, trainer, x, y, heading, ball_x, ball_y) -> int:
    """Apply a reset; returns the first cycle whose sense_body arrived after it (t_r)."""
    t_before = player.last_body["time"] if player.last_body else -1
    trainer.move_player(TEAM, player.unum, x, y, heading)
    trainer.move_ball(ball_x, ball_y)
    trainer.recover()
    body, _ = player.wait_for_sense_body(after_time=t_before, timeout=1.0)
    if abs(body.get("head_angle", 0.0)) > 1e-6:
        player.turn_neck(-body["head_angle"])
        body, _ = player.wait_for_sense_body(after_time=body["time"], timeout=1.0)
    return body["time"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/artifacts",
                    help="directory for the server params snapshot")
    args = ap.parse_args()
    ok = True

    trainer, player = RoboCupTrainer(), RoboCupPlayer(team_name=TEAM)
    try:
        trainer.connect()
        player.connect()
        ok &= report("handshakes", True, f"side={player.side} unum={player.unum} "
                     f"player_types={len(player.player_types)}")
        os.makedirs(args.out, exist_ok=True)
        snap = os.path.join(args.out, "server_params.json")
        save_params(player.params, snap)
        print(f"       params snapshot -> {snap}")

        trainer.eye(True)
        trainer.change_mode("play_on")

        # 2. Clock and see cadence over 30 cycles.
        body, _ = player.wait_for_sense_body(timeout=2.0)
        times, see_times, see_with_ball = [body["time"]], [], 0
        for _ in range(30):
            body, events = player.wait_for_sense_body(after_time=times[-1], timeout=1.0)
            times.append(body["time"])
            for kind, data in events:
                if kind == "see":
                    see_times.append(data["time"])
                    see_with_ball += data["ball"] is not None
        steps = {b - a for a, b in zip(times, times[1:])}
        ok &= report("clock advances 1 per cycle", steps == {1}, f"deltas={sorted(steps)}")
        print(f"       see messages: {len(see_times)} in 30 cycles, with ball: {see_with_ball}")

        # 3. Reset to a known configuration.
        t_r = reset(player, trainer, -10.0, 0.0, 0.0, -5.0, 0.0)
        s = wait_truth(player, trainer, t_r)
        tb = truth_bearing(s["view"], player.unum)
        ok &= report("reset matches request",
                     abs(tb["x"] + 10) < 0.05 and abs(tb["y"]) < 0.05 and abs(tb["body"]) < 0.5
                     and tb["speed"] < 1e-3 and abs(tb["d"] - 5) < 0.05,
                     f"player=({tb['x']:.2f},{tb['y']:.2f}) body={tb['body']:.1f} "
                     f"speed={tb['speed']:.3f} d={tb['d']:.2f}")
        ok &= report("neck zeroed", abs(s["body"].get("head_angle", 0.0)) < 1e-6,
                     f"head_angle={s['body'].get('head_angle')}")

        # 4. Turn sign: ball at body-relative +20 deg, 5 m ahead.
        a = math.radians(20.0)
        t_r = reset(player, trainer, -10.0, 0.0, 0.0, -10.0 + 5 * math.cos(a), 5 * math.sin(a))
        before = truth_bearing(wait_truth(player, trainer, t_r)["view"], player.unum)
        player.turn(35.0)
        s = wait_truth(player, trainer, player.last_body["time"])
        after = truth_bearing(s["view"], player.unum)
        ok &= report("TURN +35 reduces |bearing| of a ball at +20",
                     abs(after["theta"]) < abs(before["theta"]),
                     f"theta {before['theta']:.1f} -> {after['theta']:.1f}")

        # 5. Straight dash from rest vs. the kinematic model.
        t_r = reset(player, trainer, -20.0, 0.0, 0.0, 10.0, 0.0)
        prev = truth_bearing(wait_truth(player, trainer, t_r)["view"], player.unum)
        rate = player.params["dash_power_rate"] * player.params.get("effort_max", 1.0)
        v, rows = 0.0, []
        for _ in range(10):
            player.dash(100.0)
            s = wait_truth(player, trainer, player.last_body["time"])
            cur = truth_bearing(s["view"], player.unum)
            u = min(v + 100.0 * rate, player.params["player_speed_max"])
            v = u * player.params["player_decay"]
            rows.append((cur["x"] - prev["x"], u))
            prev = cur
        err = max(abs(m - p) for m, p in rows)
        ok &= report("dash displacement matches model (noise tolerance 0.15 m)", err < 0.15,
                     "measured/model: " + ", ".join(f"{m:.2f}/{p:.2f}" for m, p in rows))
        if player.errors:
            print("       server errors/warnings:", list(player.errors))
    finally:
        player.close()
        trainer.close()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
