"""End-to-end test of LiveBallPursuitEnv against a fake rcssserver.

The fake server runs SimBallPursuitEnv physics behind the player and trainer
UDP protocols (format per protocol v15 as we understand it). It validates our
plumbing — reset sequence, cycle/see synchronization, command verification,
ground-truth alignment — not the real server's behaviour.
"""

import math
import socket
import threading
import time

from src.client import RoboCupPlayer
from src.controllers import GreedyPursuit, rollout
from src.live_env import LiveBallPursuitEnv
from src.params import DEFAULT_PARAMS
from src.sampler import StartConfig
from src.sim_env import SimBallPursuitEnv
from src.trainer import RoboCupTrainer

CYCLE = 0.02  # seconds per fake cycle


class FakeRcss:
    def __init__(self):
        self.sim = SimBallPursuitEnv(motion_noise=False, sensor_noise=False, seed=0)
        self.sim.reset(StartConfig(10, 0, 0, 0, 0, 10, 0))
        self.sim._see_phase_ms = 0.0
        self.lock = threading.Lock()
        self.pending = None
        self.counts = {"dash": 0, "turn": 0, "turn_neck": 0}
        self.player_addr = self.trainer_addr = None
        self.eye = False
        self.stop = False
        self.psock = self._sock()
        self.tsock = self._sock()
        for target in (self._player_loop, self._trainer_loop, self._cycle_loop):
            threading.Thread(target=target, daemon=True).start()

    @staticmethod
    def _sock():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        s.settimeout(0.005)
        return s

    def _send(self, sock, addr, msg):
        if addr is not None:
            sock.sendto((msg + "\x00").encode(), addr)

    def _recv_loop(self, sock, handle):
        while not self.stop:
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                continue
            with self.lock:
                handle(data.decode().rstrip("\x00"), addr)

    def _player_loop(self):
        def handle(msg, addr):
            if msg.startswith("(init"):
                self.player_addr = addr
                self._send(self.psock, addr, "(init l 1 before_kick_off)")
                self._send(self.psock, addr, "(server_param (player_decay 0.4)(inertia_moment 5)"
                                             "(dash_power_rate 0.006)(send_step 150)"
                                             "(simulator_step 100)(visible_angle 90))")
                self._send(self.psock, addr, "(player_param (player_types 1))")
                self._send(self.psock, addr, "(player_type (id 0)(player_speed_max 1.05))")
            elif msg.startswith("(dash") or msg.startswith("(turn "):
                kind, arg = msg.strip("()").split()[:2]
                self.pending = (kind, float(arg))
        self._recv_loop(self.psock, handle)

    def _trainer_loop(self):
        def handle(msg, addr):
            self.trainer_addr = addr
            sim = self.sim
            if msg.startswith("(init"):
                self._send(self.tsock, addr, "(init ok)")
                return
            if msg.startswith("(move (ball)"):
                x, y = map(float, msg.split()[2:4])
                sim.bx, sim.by = x, y
            elif msg.startswith("(move (player"):
                x, y, d = map(float, msg.rstrip(")").split()[4:7])
                sim.px, sim.py, sim.body, sim.vx, sim.vy = x, y, d, 0.0, 0.0
            head = msg.strip("()").split()[0]
            if head == "eye":
                self.eye = True
                self._send(self.tsock, addr, "(ok eye on)")
            elif head in ("move", "recover", "change_mode"):
                self._send(self.tsock, addr, f"(ok {head})")
        self._recv_loop(self.tsock, handle)

    def _cycle_loop(self):
        while not self.stop:
            time.sleep(CYCLE)
            with self.lock:
                sim = self.sim
                cmd = self.pending or ("turn", 0.0)
                if self.pending:
                    self.counts[self.pending[0]] += 1
                self.pending = None
                body, see, _late, truth = sim._backend_step(cmd)  # phase 0: never late
                t = body["time"]
                self._send(self.psock, self.player_addr,
                           f"(sense_body {t} (view_mode high normal) (stamina 8000 1 130600) "
                           f"(speed {body['speed']:.4f} {body['speed_dir']:.2f}) (head_angle 0) "
                           f"(kick 0) (dash {self.counts['dash']}) (turn {self.counts['turn']}) "
                           f"(say 0) (turn_neck 0))")
                if self.eye:
                    self._send(self.tsock, self.trainer_addr,
                               f"(see_global {t} ((b) {sim.bx:.4f} {sim.by:.4f} 0 0) "
                               f'((p "UTEC_RL" 1) {sim.px:.4f} {sim.py:.4f} {sim.vx:.4f} '
                               f"{sim.vy:.4f} {sim.body:.4f} 0))")
                see_msg = None
                if see is not None:
                    ball = see["ball"]
                    obj = "" if ball is None else (
                        f" (({'b' if ball['named'] else 'B'}) {ball['dist']:.4f} {ball['dir_head']:.4f})")
                    see_msg = f"(see {t}{obj})"
            if see_msg is not None:
                if t % 2:
                    time.sleep(CYCLE / 4)  # late see: arrives after sense_body
                self._send(self.psock, self.player_addr, see_msg)


def test_live_env_episode_against_fake_server():
    srv = FakeRcss()
    trainer = RoboCupTrainer("127.0.0.1", srv.tsock.getsockname()[1])
    player = RoboCupPlayer("127.0.0.1", srv.psock.getsockname()[1])
    try:
        trainer.connect(timeout=1.0)
        player.connect(timeout=1.0, param_wait=0.2)
        trainer.eye(True)
        trainer.change_mode("play_on")
        player.wait_for_sense_body(timeout=1.0)
        env = LiveBallPursuitEnv(player, trainer, see_wait=CYCLE / 2, seed=0)

        start = StartConfig(8.0, 60.0, 0.0, -10.0, 0.0,
                            -10.0 + 8 * math.cos(math.radians(60)), 8 * math.sin(math.radians(60)))
        res = rollout(env, GreedyPursuit(DEFAULT_PARAMS), start)
        assert res["captured"], res["steps"]
        first = res["trajectory"][0]
        assert math.isclose(first["true_d"], 8.0, abs_tol=0.01)
        assert env.stats["missed_commands"] == 0
        assert env.stats["resets"] == 1

        # Second episode reuses the session (reset while in play_on).
        res2 = rollout(env, GreedyPursuit(DEFAULT_PARAMS), StartConfig(6.0, 0.0, 90.0, 0, 0, 0, 6.0))
        assert res2["captured"] and res2["steps"] <= 8
    finally:
        srv.stop = True
        player.close()
        trainer.close()
