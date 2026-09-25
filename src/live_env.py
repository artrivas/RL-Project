"""Ball Pursuit on the live rcssserver (player client + trainer), same interface as the simulator.

Reset follows ``src/SPEC.md`` §9, and every reset is verified against trainer
ground truth before step 0. Per decision cycle:

1. Wait for ``sense_body(t)``; then wait up to ``see_wait`` seconds (the
   decision deadline) for a ``see(t)``. The see phase within a cycle depends on
   the connection (measured 0/50 ms or 41/91 ms after ``sense_body``); a
   ``see(t-1)`` that missed its deadline is passed on as ``late_see``, older
   ones are counted as stale and discarded.
2. Ground truth for cycle ``t`` comes from the trainer's ``see_global``.
3. Command execution is checked through the ``sense_body`` counters; a
   command that did not execute in its cycle is counted in ``self.stats``.

Requires the trainer-controlled server configuration in
``docker/docker-compose.yml``.
"""

import math
import time
from typing import Any, Dict, Optional, Tuple

from src.client import RoboCupPlayer
from src.env_base import BallPursuitEnvBase, T_MAX
from src.sampler import StartConfig, wrap_deg
from src.trainer import RoboCupTrainer


class ResetError(RuntimeError):
    pass


class LiveBallPursuitEnv(BallPursuitEnvBase):
    def __init__(self, player: RoboCupPlayer, trainer: RoboCupTrainer, t_max: int = T_MAX,
                 seed: Optional[int] = None, see_wait: float = 0.06,
                 truth_wait: float = 0.1, pos_tol: float = 0.05, dir_tol: float = 0.5):
        self.player, self.trainer = player, trainer
        self.params = player.params
        self.see_wait, self.truth_wait = see_wait, truth_wait
        self.pos_tol, self.dir_tol = pos_tol, dir_tol
        self.stats = {"missed_commands": 0, "late_see": 0, "stale_see": 0, "cycles": 0,
                      "resets": 0}
        self._counts: Dict[str, int] = {}
        super().__init__(t_max=t_max, seed=seed)

    @classmethod
    def connect(cls, team_name: str = "UTEC_RL", **kwargs) -> "LiveBallPursuitEnv":
        """Connect player and trainer, enable per-cycle ground truth, enter play_on."""
        trainer, player = RoboCupTrainer(), RoboCupPlayer(team_name=team_name)
        trainer.connect()
        player.connect()
        trainer.eye(True)
        trainer.change_mode("play_on")
        player.wait_for_sense_body(timeout=2.0)
        return cls(player, trainer, **kwargs)

    def close(self) -> None:
        self.player.close()
        self.trainer.close()

    # ------------------------------------------------------------ cycles
    def _next_cycle(self, after: int):
        """Return (body, see(t) before the deadline, late see(t-1)) for the next cycle."""
        body, events = self.player.wait_for_sense_body(after_time=after, timeout=1.0)
        t = body["time"]
        see = late = None
        for kind, data in events:
            if kind == "see":
                if data["time"] == t:
                    see = data
                elif data["time"] == t - 1 and t - 1 == after:
                    late = data
                else:
                    self.stats["stale_see"] += 1
        deadline = time.monotonic() + self.see_wait
        while see is None and time.monotonic() < deadline:
            for kind, data in self.player.poll(deadline - time.monotonic()):
                if kind == "see" and data["time"] == t:
                    see = data
        self.stats["cycles"] += 1
        return body, see, late

    def _truth(self, cycle: int) -> Dict[str, float]:
        deadline = time.monotonic() + self.truth_wait
        while True:
            view = self.trainer.truth_at(cycle)
            if view is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no trainer ground truth for cycle {cycle}")
            self.trainer.poll(0.01)
        p = view["players"][(self.player.team_name, self.player.unum)]
        b = view["ball"]
        return {"px": p["x"], "py": p["y"], "body": p["body"], "bx": b["x"], "by": b["y"],
                "cycle": cycle, "vx": p["vx"], "vy": p["vy"], "neck": p["neck"]}

    # ------------------------------------------------------------ hooks
    def _backend_reset(self, start: StartConfig):
        last = self.player.last_body["time"] if self.player.last_body else -1
        self.trainer.move_player(self.player.team_name, self.player.unum,
                                 start.player_x, start.player_y, start.heading)
        self.trainer.move_ball(start.ball_x, start.ball_y)
        self.trainer.recover()
        body, _ = self.player.wait_for_sense_body(after_time=last, timeout=1.0)
        if abs(body.get("head_angle") or 0.0) > 1e-6:
            self.player.turn_neck(-body["head_angle"])
            body, _ = self.player.wait_for_sense_body(after_time=body["time"], timeout=1.0)
        # Observations up to t_r may predate the reset; step 0 is cycle t_r + 1.
        body, see, _late = self._next_cycle(after=body["time"])
        truth = self._truth(body["time"])
        self._check_reset(start, body, truth)
        self._counts = dict(body.get("counts", {}))
        self.stats["resets"] += 1
        return body, see, None, truth

    def _check_reset(self, start: StartConfig, body: Dict[str, Any], truth: Dict[str, float]):
        problems = []
        if math.hypot(truth["px"] - start.player_x, truth["py"] - start.player_y) > self.pos_tol:
            problems.append("player position")
        if math.hypot(truth["bx"] - start.ball_x, truth["by"] - start.ball_y) > self.pos_tol:
            problems.append("ball position")
        if abs(wrap_deg(truth["body"] - start.heading)) > self.dir_tol:
            problems.append("heading")
        if math.hypot(truth["vx"], truth["vy"]) > 1e-3:
            problems.append("player velocity")
        if abs(body.get("head_angle") or 0.0) > 1e-6:
            problems.append("neck angle")
        if problems:
            raise ResetError(f"reset mismatch ({', '.join(problems)}): truth={truth}, start={start}")

    def _backend_step(self, command: Tuple[str, float]):
        kind, arg = command
        t = self.player.last_body["time"]
        if kind == "dash":
            self.player.dash(arg)
        else:
            self.player.turn(arg)
        body, see, late = self._next_cycle(after=t)
        counts = body.get("counts", {})
        if counts.get(kind, 0) != self._counts.get(kind, 0) + 1:
            self.stats["missed_commands"] += 1
        self._counts = dict(counts)
        self.stats["late_see"] += late is not None
        return body, see, late, self._truth(body["time"])
