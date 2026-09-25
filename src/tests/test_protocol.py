"""Offline tests for the protocol layer: parsing plus socket handshakes against a fake server.

Message formats follow rcssserver protocol version 15. They validate our parsing
logic, not the live server; live behaviour is checked by ``src/live_check.py``.
"""

import socket
import threading

import pytest

from src import sexp
from src.client import RoboCupPlayer, parse_see, parse_sense_body
from src.params import effective_params, parse_param_message
from src.trainer import RoboCupTrainer, TrainerError, parse_global_view

SENSE_BODY = ("(sense_body 42 (view_mode high normal) (stamina 7900.5 1 130000) "
              "(speed 0.62 -12) (head_angle 30) (kick 0) (dash 7) (turn 3) (say 0) "
              "(turn_neck 1) (catch 0) (move 1) (change_view 0) (arm (movable 0) "
              "(expires 0) (target 0 0) (count 0)) (focus (target none) (count 0)) "
              "(tackle (expires 0) (count 0)) (collision none) (foul (charged 0) (card none)))")


def test_sexp_nested_and_strings():
    tree = sexp.parse('(see 12 ((b) 5.2 -10) ((p "UTEC RL" 1) 3 4))\x00')
    assert tree == ["see", 12.0, [["b"], 5.2, -10.0], [["p", "UTEC RL", 1.0], 3.0, 4.0]]


def test_sexp_unbalanced():
    with pytest.raises(ValueError):
        sexp.parse("(see 1 ((b) 2 3)")


def test_sense_body():
    body = parse_sense_body(sexp.parse(SENSE_BODY))
    assert body["time"] == 42
    assert body["head_angle"] == 30.0
    assert (body["speed"], body["speed_dir"]) == (0.62, -12.0)
    assert (body["stamina"], body["effort"]) == (7900.5, 1.0)
    assert body["counts"]["dash"] == 7 and body["counts"]["turn"] == 3
    assert body["view_width"] == "normal" and body["collision"] == ["none"]


def test_see_ball_present_absent_and_unnamed():
    full = parse_see(sexp.parse("(see 7 ((f c) 10 3) ((b) 12.2 -20 0.1 0.2))"))
    assert full["ball"] == {"named": True, "dist": 12.2, "dir_head": -20.0,
                            "dist_chng": 0.1, "dir_chng": 0.2}
    # A see without the ball must not report a ball (no stale carry-over).
    assert parse_see(sexp.parse("(see 8 ((f c) 10 3))"))["ball"] is None
    close = parse_see(sexp.parse("(see 9 ((B) 1.5 170))"))["ball"]
    assert close["named"] is False and close["dist"] == 1.5
    low = parse_see(sexp.parse("(see 10 ((b) -35))"))["ball"]
    assert low["dist"] is None and low["dir_head"] == -35.0


def test_effective_params_player_type_overrides():
    sp = parse_param_message("(server_param (player_decay 0.4)(dash_power_rate 0.006)"
                             "(visible_angle 90)(team_l_start \"\"))")["values"]
    pt = parse_param_message("(player_type (id 0)(player_speed_max 1.05)(player_decay 0.45))")
    params = effective_params(sp, [pt["values"]])
    assert params["player_decay"] == 0.45 and params["visible_angle"] == 90.0
    assert params["team_l_start"] == ""


def test_global_view():
    tree = sexp.parse('(ok look 55 ((g l) -52.5 0) ((b) 1 2 0 0) '
                      '((p "UTEC_RL" 1) -10 3 0.5 0 45 -20))')
    view = parse_global_view(tree, 2)
    assert view["time"] == 55 and view["ball"]["y"] == 2.0
    assert view["players"][("UTEC_RL", 1)] == {"x": -10.0, "y": 3.0, "vx": 0.5, "vy": 0.0,
                                               "body": 45.0, "neck": -20.0}


# ------------------------------------------------------------ fake server
class FakeServer:
    """Replies from a fresh per-client port, like rcssserver does."""

    def __init__(self, handler):
        self.listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen.bind(("127.0.0.1", 0))
        self.port = self.listen.getsockname()[1]
        self.reply = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reply.bind(("127.0.0.1", 0))
        self.handler = handler
        self.received = []
        self.stop = False
        threading.Thread(target=self._serve, args=(self.listen,), daemon=True).start()
        threading.Thread(target=self._serve, args=(self.reply,), daemon=True).start()

    def _serve(self, sock):
        sock.settimeout(0.1)
        while not self.stop:
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                continue
            msg = data.decode().rstrip("\x00")
            self.received.append(msg)
            for out in self.handler(msg):
                self.reply.sendto((out + "\x00").encode(), addr)


def test_player_handshake_and_clock():
    def handler(msg):
        if msg.startswith("(init"):
            return ["(init l 1 before_kick_off)",
                    "(server_param (player_decay 0.4)(dash_power_rate 0.006))",
                    "(player_param (player_types 18))",
                    "(player_type (id 0)(player_speed_max 1.05))",
                    "(see 0 ((b) 10 5))", SENSE_BODY]
        return []

    srv = FakeServer(handler)
    player = RoboCupPlayer("127.0.0.1", srv.port)
    try:
        player.connect(timeout=2.0)
        assert (player.side, player.unum) == ("l", 1)
        assert player.server_addr[1] != srv.port  # switched to the per-client port
        assert player.params["player_speed_max"] == 1.05
        body, events = player.wait_for_sense_body(after_time=-1, timeout=1.0)
        assert body["time"] == 42
        player.dash(100)
        with pytest.raises(TimeoutError):
            player.wait_for_sense_body(after_time=42, timeout=0.2)
        assert any(m.startswith("(dash 100") for m in srv.received)
    finally:
        srv.stop = True
        player.close()


def test_trainer_acks_and_errors():
    def handler(msg):
        if msg.startswith("(init"):
            return ["(init ok)"]
        if msg.startswith("(move"):
            return ["(see_global 3 ((b) 0 0 0 0))", "(ok move)"]
        if msg == "(recover)":
            return ["(ok recover)"]
        if msg.startswith("(change_mode"):
            return ["(error illegal_mode)"]
        return []

    srv = FakeServer(handler)
    trainer = RoboCupTrainer("127.0.0.1", srv.port)
    try:
        trainer.connect(timeout=2.0)
        trainer.move_player("UTEC_RL", 1, -10, 0, 90)
        trainer.recover()
        assert trainer.truth_at(3)["ball"]["x"] == 0.0
        with pytest.raises(TrainerError):
            trainer.change_mode("bogus")
        with pytest.raises(TrainerError):
            trainer.eye(True)  # fake server never acknowledges
    finally:
        srv.stop = True
        trainer.close()


def test_trainer_missing_coach_mode():
    silent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    silent.bind(("127.0.0.1", 0))
    trainer = RoboCupTrainer("127.0.0.1", silent.getsockname()[1])
    with pytest.raises(ConnectionError, match="coach=true"):
        trainer.connect(timeout=0.3)
    silent.close()


def test_player_team_full_error_is_actionable():
    srv = FakeServer(lambda msg: ["(error no_more_player_or_goalie_or_illegal_client_version)"]
                     if msg.startswith("(init") else [])
    player = RoboCupPlayer("127.0.0.1", srv.port)
    try:
        with pytest.raises(ConnectionError, match="docker compose restart rcssserver"):
            player.connect(timeout=1.0)
    finally:
        srv.stop = True
        player.close()
