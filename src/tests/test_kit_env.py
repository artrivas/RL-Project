import math

import numpy as np
import pytest

from src.env_base import CAPTURE_BONUS, STEP_PENALTY
from src.kit_env import KitBallPursuitEnv
from src.sampler import StartConfig


def start_at(bearing, distance=10.0, heading=0.0, px=-10.0, py=0.0):
    a = math.radians(heading + bearing)
    return StartConfig(distance, bearing, heading, px, py,
                       px + distance * math.cos(a), py + distance * math.sin(a))


def test_kit_kinematics_exact():
    env = KitBallPursuitEnv()
    env.reset(start_at(0.0, 20.0))
    env.step(0)                                   # DASH 100 -> exactly 1.0 m
    assert math.isclose(env.px, -9.0, abs_tol=1e-12)
    env.step(1)                                   # DASH 50 -> exactly 0.5 m
    assert math.isclose(env.px, -8.5, abs_tol=1e-12)
    env.step(2)                                   # TURN +35 -> heading +35 exactly
    assert math.isclose(env.body, 35.0, abs_tol=1e-12)
    env.step(3)
    assert math.isclose(env.body, 0.0, abs_tol=1e-12)


def test_kit_observation_is_true_polar_and_turn_sign():
    env = KitBallPursuitEnv()
    obs, info = env.reset(start_at(20.0, 5.0))
    assert math.isclose(obs.theta, 20.0, abs_tol=1e-9) and math.isclose(obs.d, 5.0, abs_tol=1e-9)
    obs, *_ = env.step(2)                         # TURN +35 reduces a positive bearing
    assert math.isclose(obs.theta, -15.0, abs_tol=1e-9)


def test_kit_capture_reward_and_truncation():
    env = KitBallPursuitEnv(t_max=5)
    env.reset(start_at(0.0, 2.5))
    _, r, term, trunc, _ = env.step(0)            # 2.5 -> 1.5
    assert not term and math.isclose(r, 1.0 - STEP_PENALTY)
    _, r, term, trunc, _ = env.step(0)            # 1.5 -> 0.5: captured
    assert term and not trunc and math.isclose(r, 1.0 - STEP_PENALTY + CAPTURE_BONUS)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(start_at(180.0, 30.0))
    for _ in range(5):
        _, _, term, trunc, _ = env.step(2)
    assert trunc and not term and env.step_count == 5
