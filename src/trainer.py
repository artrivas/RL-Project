"""Offline-coach (trainer) client for rcssserver.

Used for episode resets and ground-truth measurement only — never as a policy
input (see ``src/SPEC.md``). Requires the server to run with
``server::coach=true`` (see ``docker/docker-compose.yml``); otherwise nothing
listens on the trainer port and ``connect()`` times out.

Every command waits for the server's ``(ok ...)`` acknowledgement and raises on
``(error ...)`` or silence, so a reset can never be half-applied unnoticed.
"""

import os
import select
import socket
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from src import sexp


class TrainerError(RuntimeError):
    pass


def parse_global_view(tree: List[Any], time_index: int) -> Dict[str, Any]:
    """Parse ``(ok look TIME ...)`` / ``(see_global TIME ...)`` object lists.

    Returns ``{"time", "ball": {x, y, vx, vy}, "players": {(team, unum): {...}}}``
    with positions/velocities in the server's global field frame and ``body`` /
    ``neck`` angles in degrees (neck relative to body).
    """
    view: Dict[str, Any] = {"time": int(tree[time_index]), "ball": None, "players": {}}
    for item in tree[time_index + 1:]:
        if not (isinstance(item, list) and item and isinstance(item[0], list) and item[0]):
            continue
        name, nums = item[0], [v for v in item[1:] if isinstance(v, float)]
        if name[0] == "b" and len(nums) >= 4:
            view["ball"] = dict(zip(("x", "y", "vx", "vy"), nums[:4]))
        elif name[0] == "p" and len(name) >= 3 and len(nums) >= 6:
            key = (str(name[1]), int(name[2]))
            view["players"][key] = dict(zip(("x", "y", "vx", "vy", "body", "neck"), nums[:6]))
    return view


class RoboCupTrainer:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 version: int = 15, history: int = 2000):
        self.host = host or os.environ.get("SERVER_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("TRAINER_PORT", 6001))
        self.version = version
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.server_addr: Tuple[str, int] = (self.host, self.port)
        self.connected = False
        self._history = history
        self.global_views: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.messages: List[str] = []  # unhandled messages, for debugging

    # ------------------------------------------------------------------ I/O
    def _send(self, cmd: str) -> None:
        self.sock.sendto(f"{cmd}\x00".encode("ascii"), self.server_addr)

    def _recv(self, timeout: float) -> Optional[str]:
        ready, _, _ = select.select([self.sock], [], [], max(0.0, timeout))
        if not ready:
            return None
        data, addr = self.sock.recvfrom(16384)
        if not self.connected:
            self.server_addr = addr
        return data.decode("latin1").rstrip("\x00")

    def _store_view(self, view: Dict[str, Any]) -> None:
        self.global_views[view["time"]] = view
        while len(self.global_views) > self._history:
            self.global_views.popitem(last=False)

    def _handle(self, text: str) -> Tuple[str, Any]:
        tree = sexp.parse(text)
        head = tree[0] if isinstance(tree, list) and tree else ""
        if head == "see_global":
            view = parse_global_view(tree, 1)
            self._store_view(view)
            return ("see_global", view)
        if head == "ok" and len(tree) > 1 and tree[1] == "look":
            view = parse_global_view(tree, 2)
            self._store_view(view)
            return ("ok", ["look", view])
        if head in ("ok", "error", "warning"):
            return (head, tree[1:])
        self.messages.append(text)
        return ("other", tree)

    def poll(self, timeout: float = 0.0) -> None:
        """Consume pending messages (stores ``see_global`` views)."""
        text = self._recv(timeout)
        while text is not None:
            self._handle(text)
            text = self._recv(0.0)

    def _request(self, cmd: str, ack: str, timeout: float = 1.0) -> List[Any]:
        """Send ``cmd`` and wait for ``(ok <ack> ...)``; raise on error or silence."""
        self._send(cmd)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TrainerError(f"no acknowledgement for {cmd!r} within {timeout}s")
            text = self._recv(remaining)
            if text is None:
                continue
            kind, payload = self._handle(text)
            if kind == "error":
                raise TrainerError(f"{cmd!r} -> {text}")
            if kind == "ok" and payload and payload[0] == ack:
                return payload

    # ------------------------------------------------------------ handshake
    def connect(self, timeout: float = 3.0) -> None:
        self._send(f"(init (version {self.version}))")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConnectionError(
                    f"no trainer reply at {self.host}:{self.port}; is the server "
                    "running with server::coach=true?")
            text = self._recv(remaining)
            if text is None:
                continue
            tree = sexp.parse(text)
            if isinstance(tree, list) and tree[:2] == ["init", "ok"]:
                self.connected = True
                return
            if isinstance(tree, list) and tree and tree[0] == "error":
                raise ConnectionError(f"trainer init rejected: {text}")

    def close(self) -> None:
        if self.connected:
            try:
                self._send("(bye)")
            except OSError:
                pass
        self.sock.close()
        self.connected = False

    # ------------------------------------------------------------- commands
    def move_ball(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self._request(f"(move (ball) {x:.3f} {y:.3f} 0 {vx:.3f} {vy:.3f})", "move")

    def move_player(self, team: str, unum: int, x: float, y: float, body_dir: float,
                    vx: float = 0.0, vy: float = 0.0) -> None:
        self._request(f"(move (player {team} {unum}) {x:.3f} {y:.3f} {body_dir:.3f} "
                      f"{vx:.3f} {vy:.3f})", "move")

    def change_mode(self, mode: str = "play_on") -> None:
        self._request(f"(change_mode {mode})", "change_mode")

    def recover(self) -> None:
        """Restore stamina, recovery, effort and hear capacity of all players."""
        self._request("(recover)", "recover")

    def eye(self, on: bool = True) -> None:
        """Toggle per-cycle ``see_global`` messages (ground truth every cycle)."""
        self._request(f"(eye {'on' if on else 'off'})", "eye")

    def look(self) -> Dict[str, Any]:
        return self._request("(look)", "look")[1]

    def truth_at(self, cycle: int) -> Optional[Dict[str, Any]]:
        """Ground truth for ``cycle`` from stored ``see_global`` messages, if any."""
        return self.global_views.get(cycle)
