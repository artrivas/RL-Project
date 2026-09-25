"""Physics/protocol parameters for Ball Pursuit.

The live server sends its real values during the player handshake
(``server_param``, ``player_param``, ``player_type`` messages). The simulator and
the feasibility estimate must use those parsed values; ``DEFAULT_PARAMS`` is only
a fallback for offline work before a server snapshot exists, and each value is
the rcssserver default for a type-0 (default) player.
"""

import json
from typing import Any, Dict, List, Optional

from src import sexp

DEFAULT_PARAMS: Dict[str, float] = {
    # Movement (per 100 ms cycle)
    "dash_power_rate": 0.006,
    "player_decay": 0.4,
    "player_speed_max": 1.05,
    "player_accel_max": 1.0,
    "inertia_moment": 5.0,
    "player_rand": 0.1,
    "max_dash_power": 100.0,
    "min_dash_power": -100.0,
    "maxmoment": 180.0,
    "minmoment": -180.0,
    "maxneckang": 90.0,
    "minneckang": -90.0,
    # Bodies
    "player_size": 0.3,
    "ball_size": 0.085,
    "kickable_margin": 0.7,
    "ball_decay": 0.94,
    # Vision
    "visible_angle": 90.0,
    "visible_distance": 3.0,
    "send_step": 150.0,
    "simulator_step": 100.0,
    "quantize_step": 0.1,
    # Stamina
    "stamina_max": 8000.0,
    "stamina_inc_max": 45.0,
    "effort_max": 1.0,
    "effort_dec_thr": 0.3,
    "recover_dec_thr": 0.3,
}

# Player-type fields that override server_param values for the player's type.
_TYPE_FIELDS = (
    "player_speed_max", "stamina_inc_max", "player_decay", "inertia_moment",
    "dash_power_rate", "player_size", "kickable_margin", "kick_rand",
    "extra_stamina", "effort_max", "effort_min",
)


def parse_param_message(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``(server_param ...)``, ``(player_param ...)`` or ``(player_type ...)``.

    Returns ``{"kind": <head>, "values": {...}}`` or ``None`` for other messages.
    """
    tree = sexp.parse(text)
    if not isinstance(tree, list) or not tree or tree[0] not in (
            "server_param", "player_param", "player_type"):
        return None
    return {"kind": tree[0], "values": sexp.pairs_to_dict(tree[1:])}


def effective_params(server_param: Optional[Dict[str, Any]] = None,
                     player_types: Optional[List[Dict[str, Any]]] = None,
                     player_type_id: int = 0) -> Dict[str, Any]:
    """Merge defaults <- server_param <- player_type[player_type_id]."""
    params: Dict[str, Any] = dict(DEFAULT_PARAMS)
    if server_param:
        params.update(server_param)
    for ptype in player_types or []:
        if int(ptype.get("id", -1)) == player_type_id:
            params.update({k: ptype[k] for k in _TYPE_FIELDS if k in ptype})
    return params


def save_params(params: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)


def load_params(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
