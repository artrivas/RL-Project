"""UDP player client for rcssserver (the project's "cliente" module).

Adapted from the starter kit's ``workspace/robocup_client.py`` with the defects
that matter for Ball Pursuit fixed:

* A ``see`` message without the ball yields ``ball=None`` for that message; a
  previous ball observation is never carried forward silently.
* ``server_param`` / ``player_param`` / ``player_type`` from the handshake are
  parsed and exposed as ``self.params`` (the simulator is built from them).
* ``sense_body`` is fully parsed (time, head angle, speed with direction,
  stamina/effort, command counters) — it is the per-cycle clock.
* ``turn_neck`` and ``change_view`` are supported.

Observation directions are exactly as the server sends them: relative to the
head (body + neck). Converting to body-relative bearings is the estimator's job
(see ``src/SPEC.md``).
"""

import os
import select
import socket
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from src import sexp
from src.params import effective_params, parse_param_message

Event = Tuple[str, Dict[str, Any]]


def parse_sense_body(tree: List[Any]) -> Dict[str, Any]:
    """Parse a ``(sense_body TIME (field ...) ...)`` tree into a flat dict."""
    body: Dict[str, Any] = {"time": int(tree[1]), "counts": {}}
    for item in tree[2:]:
        if not isinstance(item, list) or not item:
            continue
        key, vals = item[0], item[1:]
        if key == "view_mode":
            body["view_quality"], body["view_width"] = vals[0], vals[1]
        elif key == "stamina":
            body["stamina"] = vals[0]
            body["effort"] = vals[1] if len(vals) > 1 else None
            body["stamina_capacity"] = vals[2] if len(vals) > 2 else None
        elif key == "speed":
            body["speed"] = vals[0]
            body["speed_dir"] = vals[1] if len(vals) > 1 else None
        elif key == "head_angle":
            body["head_angle"] = vals[0]
        elif key == "collision":
            body["collision"] = [v if isinstance(v, str) else v[0] for v in vals]
        elif key in ("kick", "dash", "turn", "say", "turn_neck", "catch",
                     "move", "change_view", "change_focus"):
            body["counts"][key] = int(vals[0])
    return body


def parse_see(tree: List[Any]) -> Dict[str, Any]:
    """Parse a ``(see TIME ((obj) ...) ...)`` tree; only the ball is extracted.

    ``ball`` is ``None`` when the message does not contain the ball. ``named``
    is False for the capitalised ``(B)`` form (ball sensed within
    ``visible_distance`` but outside the view cone). ``dist`` may be ``None``
    when the server only sends a direction.
    """
    see: Dict[str, Any] = {"time": int(tree[1]), "ball": None}
    for item in tree[2:]:
        if not (isinstance(item, list) and item and isinstance(item[0], list) and item[0]):
            continue
        name = item[0][0]
        if name not in ("b", "B"):
            continue
        nums = [v for v in item[1:] if isinstance(v, float)]
        ball: Dict[str, Any] = {"named": name == "b", "dist": None, "dir_head": None,
                                "dist_chng": None, "dir_chng": None}
        if len(nums) == 1:
            ball["dir_head"] = nums[0]
        elif len(nums) >= 2:
            ball["dist"], ball["dir_head"] = nums[0], nums[1]
            if len(nums) >= 4:
                ball["dist_chng"], ball["dir_chng"] = nums[2], nums[3]
        see["ball"] = ball
        break
    return see


class RoboCupPlayer:
    """Non-blocking UDP player client. One instance = one player on the pitch."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 team_name: str = "UTEC_RL", version: int = 15):
        self.host = host or os.environ.get("SERVER_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("SERVER_PORT", 6000))
        self.team_name = team_name
        self.version = version
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.server_addr: Tuple[str, int] = (self.host, self.port)
        self.side: Optional[str] = None
        self.unum: Optional[int] = None
        self.play_mode: Optional[str] = None
        self.connected = False

        self.server_param: Dict[str, Any] = {}
        self.player_param: Dict[str, Any] = {}
        self.player_types: List[Dict[str, Any]] = []
        self.params: Dict[str, Any] = effective_params()

        self.last_body: Optional[Dict[str, Any]] = None
        self.last_see: Optional[Dict[str, Any]] = None
        self._pending: List[Event] = []  # events read during connect(), not yet returned
        self.errors: Deque[str] = deque(maxlen=100)

    # ------------------------------------------------------------------ I/O
    def _send(self, cmd: str) -> None:
        self.sock.sendto(f"{cmd}\x00".encode("ascii"), self.server_addr)

    def _recv(self, timeout: float) -> Optional[str]:
        ready, _, _ = select.select([self.sock], [], [], max(0.0, timeout))
        if not ready:
            return None
        data, addr = self.sock.recvfrom(16384)
        if not self.connected:
            self.server_addr = addr  # the server answers from a per-client port
        return data.decode("latin1").rstrip("\x00")

    def _handle(self, text: str) -> Optional[Event]:
        head = text[1:text.find(" ")] if text.startswith("(") else ""
        if head == "sense_body":
            self.last_body = parse_sense_body(sexp.parse(text))
            return ("sense_body", self.last_body)
        if head == "see":
            self.last_see = parse_see(sexp.parse(text))
            return ("see", self.last_see)
        if head in ("server_param", "player_param", "player_type"):
            msg = parse_param_message(text)
            if msg["kind"] == "server_param":
                self.server_param = msg["values"]
            elif msg["kind"] == "player_param":
                self.player_param = msg["values"]
            else:
                self.player_types.append(msg["values"])
            self.params = effective_params(self.server_param, self.player_types)
            return (msg["kind"], msg["values"])
        if head in ("error", "warning"):
            self.errors.append(text)
            return (head, {"text": text})
        if head == "hear":
            tree = sexp.parse(text)
            if len(tree) >= 4 and tree[2] == "referee":
                self.play_mode = str(tree[3])
            return ("hear", {"text": text})
        return None

    def poll(self, timeout: float = 0.0) -> List[Event]:
        """Read every pending datagram (waiting up to ``timeout`` for the first)."""
        events, self._pending = self._pending, []
        text = self._recv(0.0 if events else timeout)
        while text is not None:
            ev = self._handle(text)
            if ev is not None:
                events.append(ev)
            text = self._recv(0.0)
        return events

    def wait_for_sense_body(self, after_time: int = -1, timeout: float = 1.0
                            ) -> Tuple[Dict[str, Any], List[Event]]:
        """Block until a ``sense_body`` with ``time > after_time`` arrives.

        Returns that body and every event received meanwhile (in order), so the
        caller can feed ``see`` messages to the estimator. Raises
        ``TimeoutError`` if the server stays silent.
        """
        deadline = time.monotonic() + timeout
        received: List[Event] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no sense_body after t={after_time} within {timeout}s")
            events = self.poll(remaining)
            received.extend(events)
            for kind, data in events:
                if kind == "sense_body" and data["time"] > after_time:
                    return data, received

    # ------------------------------------------------------------ handshake
    def connect(self, timeout: float = 3.0, param_wait: float = 0.5) -> None:
        """Send ``init``, parse the reply and the parameter messages that follow."""
        self._send(f"(init {self.team_name} (version {self.version}))")
        text = self._recv(timeout)
        if text is None:
            raise ConnectionError(f"no reply from rcssserver at {self.host}:{self.port}")
        tree = sexp.parse(text)
        if "no_more_player" in text:
            raise ConnectionError(
                "rcssserver rejected the player: the team already used all 11 uniform numbers. "
                "The server does not reuse numbers after (bye), so each server session allows 11 "
                "player connections. Restart it: docker compose restart rcssserver")
        if not (isinstance(tree, list) and tree[0] == "init" and len(tree) >= 4):
            raise ConnectionError(f"unexpected init reply: {text!r}")
        self.side, self.unum, self.play_mode = str(tree[1]), int(tree[2]), str(tree[3])
        self.connected = True
        deadline = time.monotonic() + param_wait
        while True:
            remaining = deadline - time.monotonic()
            text = self._recv(remaining) if remaining > 0 else None
            if text is None:
                break
            ev = self._handle(text)
            if ev is not None and ev[0] in ("see", "sense_body", "hear"):
                self._pending.append(ev)  # returned by the first poll()
            n_types = int(self.player_param.get("player_types", 1))
            if self.server_param and self.player_param and len(self.player_types) >= n_types:
                break
        if not self.server_param:
            raise ConnectionError("server_param not received after init")

    def close(self) -> None:
        if self.connected:
            try:
                self._send("(bye)")
            except OSError:
                pass
        self.sock.close()
        self.connected = False

    # ------------------------------------------------------------- commands
    def dash(self, power: float) -> None:
        self._send(f"(dash {power:.2f})")

    def turn(self, moment: float) -> None:
        self._send(f"(turn {moment:.2f})")

    def kick(self, power: float, direction: float) -> None:
        """``direction`` is relative to the body; the server ignores it unless the ball is kickable."""
        self._send(f"(kick {power:.2f} {direction:.2f})")

    def turn_neck(self, angle: float) -> None:
        """Relative neck turn; may be sent in the same cycle as a body command."""
        self._send(f"(turn_neck {angle:.2f})")

    def change_view(self, width: str = "normal", quality: str = "high") -> None:
        self._send(f"(change_view {width} {quality})")

    def move(self, x: float, y: float) -> None:
        """Self-placement; only honoured in before-kick-off modes. Resets use the trainer."""
        self._send(f"(move {x:.2f} {y:.2f})")
