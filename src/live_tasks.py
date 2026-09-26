"""Execution checks of the shooting, dribbling and 2v1 policies on rcssserver (plan M5).

These are *execution* checks: the greedy policies learned in the kit-style environments are run
on the real server, commands are sent every cycle and what happens is logged. They are not
performance claims about transfer, because the kit-style MDPs ignore the server's physics.

* The policy input is computed from the trainer's ground truth (``see_global``), matching the full
  observation of the kit-style MDPs. Player perception (``see``) is not used here.
* Resets use the offline trainer (``src/trainer.py``); the server runs in ``play_on``.
* Abstract actions are mapped to server commands, documented per task below.

    python -m src.live_tasks shooting --episodes 20 --out notebooks/artifacts/live_shooting.json
    python -m src.live_tasks dribbling --episodes 10 --out notebooks/artifacts/live_dribbling.json
    python -m src.live_tasks passing --episodes 10 --out notebooks/artifacts/live_passing.json

Each run opens one server session (one trainer, at most 3 players). The offline trainer is not
released after ``(bye)`` and the server allows 11 player connections per session, so restart the
server before each run: ``docker compose restart rcssserver``.
"""

import argparse
import json
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.client import RoboCupPlayer
from src.sampler import wrap_deg
from src.trainer import RoboCupTrainer

TEAM, OPP = "UTEC_RL", "UTEC_OPP"
ART = "notebooks/artifacts"


# ------------------------------------------------------------------ session
class Session:
    """Trainer plus players; ``step()`` advances one server cycle and returns the ground truth."""

    def __init__(self, n_team: int, n_opp: int = 0):
        self.trainer = RoboCupTrainer()
        self.trainer.connect()
        self.players: List[RoboCupPlayer] = []
        for team, n in ((TEAM, n_team), (OPP, n_opp)):
            for _ in range(n):
                p = RoboCupPlayer(team_name=team)
                p.connect()
                self.players.append(p)
        self.trainer.eye(True)
        self.trainer.change_mode("play_on")
        self.clock = self.players[0]
        body, _ = self.clock.wait_for_sense_body(timeout=2.0)
        self.t = body["time"]
        self.missed = 0

    def key(self, p: RoboCupPlayer):
        return (p.team_name, p.unum)

    def step(self) -> Dict[str, Any]:
        """Wait for the next cycle; drain the other players' sockets; return the truth view."""
        body, _ = self.clock.wait_for_sense_body(after_time=self.t, timeout=1.0)
        self.missed += max(0, body["time"] - self.t - 1)
        self.t = body["time"]
        for p in self.players[1:]:
            p.poll(0.0)
        for _ in range(25):
            self.trainer.poll(0.02)
            view = self.trainer.truth_at(self.t)
            if view is not None:
                return view
        raise TimeoutError(f"no see_global for cycle {self.t}")

    def place(self, placements: Dict[Any, tuple], ball: tuple) -> Dict[str, Any]:
        """``placements``: {player: (x, y, body)}; ball (x, y). Returns the first truth after it."""
        for p, (x, y, b) in placements.items():
            self.trainer.move_player(p.team_name, p.unum, x, y, b)
        self.trainer.move_ball(ball[0], ball[1])
        self.trainer.change_mode("play_on")
        self.trainer.recover()
        self.step()
        return self.step()

    def close(self):
        for p in self.players:
            p.close()
        self.trainer.close()


def me(view, s: Session, p: RoboCupPlayer) -> Dict[str, float]:
    return view["players"][s.key(p)]


def kickable(view, s: Session, p: RoboCupPlayer) -> bool:
    q, b = me(view, s, p), view["ball"]
    prm = p.params
    reach = prm.get("kickable_margin", 0.7) + prm.get("player_size", 0.3) + prm.get("ball_size", 0.085)
    return math.hypot(b["x"] - q["x"], b["y"] - q["y"]) <= reach


def load_q(path: str) -> np.ndarray:
    return np.load(path)


# ------------------------------------------------------------------ shooting
def run_shooting(n: int, q_path: str) -> Dict[str, Any]:
    """Greedy shooting policy. Shot = ``kick 100`` toward the target point on the goal line;
    ``CONDUCIR`` = ``kick 5`` toward the goal centre, then dash toward the ball until kickable
    (at most 8 cycles). The keeper variant adds an idle opponent on the goal line at y_k."""
    from src import shooting_env as se
    Q, disc = load_q(q_path), se.default_discretizer()
    out = {"policy": q_path, "episodes": [], "missed_cycles": 0}
    s = Session(1, 1)          # one trainer per server session: it is not released after (bye)
    try:
        att, gk = s.players
        for variant in ("open", "keeper"):
            keeper = gk if variant == "keeper" else None
            for start in se.evaluation_starts(500, 12345, variant)[:n]:
                bx, by = se.GOAL_X - start.d, start.y
                aim = math.atan2(0.0 - by, se.GOAL_X - bx)
                place = {att: (bx - 0.4 * math.cos(aim), by - 0.4 * math.sin(aim), math.degrees(aim))}
                # open goal: the keeper client is parked in a far corner, away from play
                place[gk] = ((se.GOAL_X - 0.5, start.keeper_y, 180.0) if keeper is not None
                             else (-50.0, 32.0, 0.0))
                view = s.place(place, (bx, by))
                ep = {"variant": variant, "start": start.as_dict(), "decisions": [], "outcome": None}
                for _ in range(se.T_MAX):
                    b = view["ball"]
                    obs = {"d_g": se.GOAL_X - b["x"], "y": b["y"], "keeper": start.keeper_y}
                    a = int(np.argmax(Q[disc(obs)]))
                    ep["decisions"].append({"action": se.ACTION_NAMES[a], "ball": (b["x"], b["y"])})
                    body = me(view, s, att)["body"]
                    if a == se.CONDUCIR:
                        att.kick(5.0, wrap_deg(math.degrees(math.atan2(-b["y"], se.GOAL_X - b["x"])) - body))
                        for _ in range(8):
                            view = s.step()
                            if kickable(view, s, att):
                                break
                            q, bb = me(view, s, att), view["ball"]
                            d = wrap_deg(math.degrees(math.atan2(bb["y"] - q["y"], bb["x"] - q["x"])) - q["body"])
                            att.turn(d) if abs(d) > 15 else att.dash(60.0)
                        continue
                    target = se.TARGETS[a]
                    direction = math.degrees(math.atan2(target - b["y"], se.GOAL_X - b["x"]))
                    att.kick(100.0, wrap_deg(direction - body))
                    track, crossed = [], None
                    for _ in range(40):
                        view = s.step()
                        bb = view["ball"]
                        track.append((bb["x"], bb["y"]))
                        if bb["x"] >= se.GOAL_X:
                            x0, y0 = track[-2] if len(track) > 1 else (b["x"], b["y"])
                            f = (se.GOAL_X - x0) / max(1e-9, bb["x"] - x0)
                            crossed = y0 + f * (bb["y"] - y0)
                            break
                        if len(track) > 3 and math.hypot(bb["vx"], bb["vy"]) < 0.05:
                            break
                    moved = track[min(2, len(track) - 1)]
                    actual = math.degrees(math.atan2(moved[1] - b["y"], moved[0] - b["x"]))
                    if crossed is None:
                        ep["outcome"] = "saved" if keeper is not None else "stopped"
                    elif abs(crossed) >= se.POST_Y:
                        ep["outcome"] = "off"
                    else:
                        ep["outcome"] = "goal"
                    ep.update({"target_y": target, "crossing_y": crossed,
                               "angular_error_deg": wrap_deg(actual - direction),
                               "ball_speed_after_kick": math.hypot(*np.subtract(track[0], (b["x"], b["y"])))})
                    break
                out["episodes"].append(ep)
    finally:
        out["missed_cycles"] = s.missed
        s.close()
    for v in ("open", "keeper"):
        eps = [e for e in out["episodes"] if e["variant"] == v]
        out[f"{v}_goal_rate"] = float(np.mean([e["outcome"] == "goal" for e in eps])) if eps else None
    errs = [e["angular_error_deg"] for e in out["episodes"] if e.get("angular_error_deg") is not None]
    out["angular_error_std_deg"] = float(np.std(errs)) if errs else None
    return out


# ------------------------------------------------------------------ dribbling
def run_dribbling(n: int, q_path: str) -> Dict[str, Any]:
    """Greedy dribbling policy with the brief's literal commands: ``KICK 25`` = ``(kick 25 0)``
    (only sent when the server considers the ball kickable), ``DASH 80`` = ``(dash 80)``,
    ``TURN +/-35`` = ``(turn +/-35)``. Outcome as in the env (30 m advance, ball > 4 m away, out)."""
    from src import dribbling_env as de
    Q, disc = load_q(q_path), de.default_discretizer()
    s = Session(1)
    out = {"policy": q_path, "episodes": []}
    try:
        p = s.players[0]
        for start in de.evaluation_starts(500, 12345)[:n]:
            h = math.radians(start.heading)
            bx0, by0 = start.x + 0.5 * math.cos(h), start.y + 0.5 * math.sin(h)
            view = s.place({p: (start.x, start.y, start.heading)}, (bx0, by0))
            ep = {"start": start.as_dict(), "outcome": "timeout", "kicks": [], "actions": []}
            for t in range(de.T_MAX):
                q, b = me(view, s, p), view["ball"]
                obs = {"d_b": math.hypot(b["x"] - q["x"], b["y"] - q["y"]),
                       "theta_b": wrap_deg(math.degrees(math.atan2(b["y"] - q["y"], b["x"] - q["x"])) - q["body"]),
                       "theta_g": wrap_deg(math.degrees(math.atan2(-q["y"], de.GOAL[0] - q["x"])) - q["body"])}
                a = int(np.argmax(Q[disc(obs)]))
                ep["actions"].append(de.ACTION_NAMES[a])
                if a == de.KICK:
                    if kickable(view, s, p):
                        p.kick(25.0, 0.0)
                        ep["kicks"].append({"t": t, "ball": (b["x"], b["y"])})
                elif a == de.DASH:
                    p.dash(80.0)
                else:
                    p.turn(de.TURN if a == de.TURN_POS else -de.TURN)
                view = s.step()
                q, b = me(view, s, p), view["ball"]
                if ep["kicks"] and ep["kicks"][-1]["t"] == t:
                    # ball speed one cycle after the kick -> free-rolling distance v / (1 - ball_decay)
                    v0 = math.hypot(b["vx"], b["vy"])
                    ep["kicks"][-1].update({"speed_after": v0,
                                            "free_travel": v0 / (1.0 - p.params.get("ball_decay", 0.94))})
                if abs(b["x"]) > 52.5 or abs(b["y"]) > 34 or abs(q["x"]) > 52.5 or abs(q["y"]) > 34:
                    ep["outcome"] = "out"; break
                if math.hypot(b["x"] - q["x"], b["y"] - q["y"]) > 4.0:
                    ep["outcome"] = "lost"; break
                if b["x"] - bx0 >= de.GOAL_ADVANCE:
                    ep["outcome"] = "success"; break
            ep["steps"], ep["advance"] = t + 1, view["ball"]["x"] - bx0
            out["episodes"].append(ep)
    finally:
        out["missed_cycles"] = s.missed
        s.close()
    travels = [k["free_travel"] for e in out["episodes"] for k in e["kicks"] if "free_travel" in k]
    out["success_rate"] = float(np.mean([e["outcome"] == "success" for e in out["episodes"]]))
    out["outcomes"] = {o: sum(e["outcome"] == o for e in out["episodes"])
                       for o in {e["outcome"] for e in out["episodes"]}}
    out["kick25_free_travel_m"] = {"mean": float(np.mean(travels)), "min": float(np.min(travels)),
                                   "max": float(np.max(travels)), "n": len(travels),
                                   "model_m": 2.0} if travels else None
    return out


# ------------------------------------------------------------------ 2v1
def run_passing(n: int, q_path: str) -> Dict[str, Any]:
    """Greedy 2v1 policy for the ball holder; scripted teammate (turn/dash toward its support
    point) and defender (turn/dash toward the holder, ``dash 60``). Mapping: ``PASE`` = kick
    toward the teammate with power for the distance; ``DRIBLE`` = ``kick 10`` along the heading
    if kickable, else dash toward the ball; ``GIRAR`` = ``turn`` 35 deg by the env's rule;
    ``DESPEJE`` = ``kick 100`` along the heading. The holder is the attacker who can kick the ball;
    while the ball travels, nobody holds it. Events are logged; the zone is 30 x 20 m around the
    centre spot."""
    from src import passing_env as pe
    Q, disc = load_q(q_path), pe.default_discretizer()
    s = Session(2, 1)
    out = {"policy": q_path, "episodes": []}

    def goto(p, view, target, power=80.0):
        q = me(view, s, p)
        d = wrap_deg(math.degrees(math.atan2(target[1] - q["y"], target[0] - q["x"])) - q["body"])
        if math.hypot(target[0] - q["x"], target[1] - q["y"]) < 0.3:
            return
        p.turn(d) if abs(d) > 20 else p.dash(power)

    try:
        a1, a2, dfd = s.players
        for start in pe.evaluation_starts(500, 12345)[:n]:
            view = s.place({a1: (*start.holder, start.heading), a2: (*start.mate, 0.0),
                            dfd: (*start.defender, 0.0)}, start.holder)
            ep = {"start": start.as_dict(), "events": [], "passes": 0, "outcome": "timeout"}
            holder, mate = a1, a2
            for t in range(pe.T_MAX):
                hp, mp, dp = me(view, s, holder), me(view, s, mate), me(view, s, dfd)
                b = view["ball"]
                if kickable(view, s, holder):
                    heading = hp["body"]
                    obs = {"d_comp": math.hypot(mp["x"] - hp["x"], mp["y"] - hp["y"]),
                           "theta_comp": pe._bearing((hp["x"], hp["y"]), (mp["x"], mp["y"]), heading),
                           "d_def": math.hypot(dp["x"] - hp["x"], dp["y"] - hp["y"]),
                           "theta_def": pe._bearing((hp["x"], hp["y"]), (dp["x"], dp["y"]), heading),
                           "blocked": 1.0 if pe.seg_dist((dp["x"], dp["y"]), (hp["x"], hp["y"]),
                                                         (mp["x"], mp["y"])) < pe.BLOCK_DIST else 0.0,
                           "edge": min(pe.HALF_X - abs(hp["x"]), pe.HALF_Y - abs(hp["y"]))}
                    a = int(np.argmax(Q[disc(obs)]))
                    ep["events"].append((t, pe.ACTION_NAMES[a]))
                    if a == pe.PASE:
                        dist = obs["d_comp"]
                        power = min(100.0, 0.06 * dist / 0.027 * 1.6)
                        holder.kick(power, pe._bearing((hp["x"], hp["y"]), (mp["x"], mp["y"]), heading))
                        holder, mate = mate, holder
                        ep["passes"] += 1
                    elif a == pe.DRIBLE:
                        holder.kick(10.0, 0.0)
                    elif a == pe.GIRAR:
                        th = obs["theta_def"]
                        holder.turn(-pe.TURN if th > 0 else pe.TURN)
                    else:
                        holder.kick(100.0, 0.0)
                        ep["outcome"] = "cleared"
                else:
                    goto(holder, view, (b["x"], b["y"]))     # chase the ball (dribble or reception)
                sp = pe.support_point((hp["x"], hp["y"]), (dp["x"], dp["y"]), (mp["x"], mp["y"]))
                goto(mate, view, sp, 60.0)
                goto(dfd, view, (b["x"], b["y"]), 60.0)
                if ep["outcome"] == "cleared":
                    break
                view = s.step()
                b = view["ball"]
                if not pe.inside((b["x"], b["y"])):
                    ep["outcome"] = "out"; break
                if kickable(view, s, dfd):
                    ep["outcome"] = "defender_ball"; break
            ep["steps"] = t + 1
            out["episodes"].append(ep)
    finally:
        out["missed_cycles"] = s.missed
        s.close()
    out["mean_passes_attempted"] = float(np.mean([e["passes"] for e in out["episodes"]]))
    out["outcomes"] = {o: sum(e["outcome"] == o for e in out["episodes"])
                       for o in {e["outcome"] for e in out["episodes"]}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["shooting", "dribbling", "passing"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--q", default=None, help="Q-table (default: Q-Learning, decaying epsilon, seed 0)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    default_q = {"shooting": f"{ART}/runs_shooting/shooting_qlearning_eps_decay_1_to_0.1/seed_0/q.npy",
                 "dribbling": f"{ART}/runs_dribbling/dribbling_qlearning_eps_decay_1_to_0.1/seed_0/q.npy",
                 "passing": f"{ART}/runs_passing/passing_qlearning_eps_decay_1_to_0.1/seed_0/q.npy"}
    q = args.q or default_q[args.task]
    t0 = time.time()
    res = {"shooting": run_shooting, "dribbling": run_dribbling, "passing": run_passing}[args.task](args.episodes, q)
    res["wall_seconds"] = time.time() - t0
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    summary = {k: v for k, v in res.items() if k != "episodes"}
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
