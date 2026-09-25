import math

import numpy as np
import pytest

from src.controllers import DASH_100, TURN_POS, GreedyPursuit, rollout
from src.env_base import CAPTURE_BONUS, STEP_PENALTY
from src.params import DEFAULT_PARAMS
from src.perception import SEEN, TRACKED, UNKNOWN
from src.sampler import StartConfig
from src.sim_env import SimBallPursuitEnv


def start_at(bearing, distance=10.0, heading=0.0, px=-10.0, py=0.0):
    a = math.radians(heading + bearing)
    return StartConfig(distance, bearing, heading, px, py,
                       px + distance * math.cos(a), py + distance * math.sin(a))


def noiseless(**kw):
    return SimBallPursuitEnv(motion_noise=False, sensor_noise=False, seed=0, **kw)


def test_dash_displacements_match_server_model():
    env = noiseless()
    env.reset(start_at(0.0, 30.0))
    xs = [env.px]
    for _ in range(4):
        env.step(DASH_100)
        xs.append(env.px)
    steps = np.diff(xs)
    assert np.allclose(steps, [0.6, 0.84, 0.936, 0.9744], atol=1e-9)


def test_turn_sign_and_speed_dependence():
    env = noiseless()
    env.reset(start_at(20.0, 5.0))
    _, _, _, _, info = env.step(TURN_POS)
    assert abs(info["true_theta"]) < 20.0          # SPEC §1 acceptance test
    assert math.isclose(info["true_theta"], -15.0, abs_tol=1e-9)

    env.reset(start_at(0.0, 30.0))
    for _ in range(5):
        env.step(DASH_100)
    speed = math.hypot(env.vx, env.vy)
    before = env.body
    env.step(TURN_POS)
    assert math.isclose(env.body - before, 35.0 / (1 + 5.0 * speed), rel_tol=1e-9)


def test_capture_terminates_with_bonus():
    env = noiseless()
    env.reset(start_at(0.0, 1.5))
    _, r, term, trunc, info = env.step(DASH_100)   # moves 0.6 -> d = 0.9
    assert not term and math.isclose(r, 0.6 - STEP_PENALTY, abs_tol=1e-9)
    _, r, term, trunc, info = env.step(DASH_100)   # moves 0.84 -> d = 0.06
    assert term and not trunc and info["captured"]
    assert math.isclose(r, 0.84 - STEP_PENALTY + CAPTURE_BONUS, abs_tol=1e-9)
    with pytest.raises(RuntimeError):
        env.step(DASH_100)


def test_truncation_at_t_max():
    env = noiseless(t_max=5)
    env.reset(start_at(180.0, 30.0))
    for i in range(5):
        _, _, term, trunc, _ = env.step(DASH_100)
    assert trunc and not term and env.step_count == 5


def test_see_cadence_two_of_three():
    env = noiseless()
    env.reset(start_at(0.0, 30.0))
    seen = [env._see_this_cycle()]
    for _ in range(29):
        env.step(TURN_POS)
        seen.append(env._see_this_cycle())
    assert sum(seen) == 20


def test_vision_cone_and_close_range():
    env = noiseless()
    env.reset(start_at(0.0, 10.0))
    env._see_phase_ms = 0.0                            # force a see in cycle 0
    for bearing, dist, expected in [(40.0, 10.0, True), (50.0, 10.0, None),
                                    (180.0, 10.0, None), (170.0, 2.0, False)]:
        s = start_at(bearing, dist)
        env.bx, env.by = s.ball_x, s.ball_y
        _, see, _ = env._sense()
        ball = see["ball"]
        assert (ball["named"] if ball else None) == expected, (bearing, dist)


def test_estimator_tracks_ball_out_of_view():
    """Acquire the ball, turn it out of the cone, dash: tracked estimate stays close to truth."""
    env = SimBallPursuitEnv(seed=3, motion_noise=True, sensor_noise=True)
    obs, info = env.reset(start_at(10.0, 20.0))
    while obs.status == UNKNOWN:
        obs, *_ = env.step(TURN_POS)
    errors = []
    for action in [TURN_POS, TURN_POS, DASH_100, DASH_100, DASH_100, TURN_POS]:
        obs, _, _, _, info = env.step(action)
        if obs.status == TRACKED:
            errors.append((abs(obs.d - info["true_d"]), abs(obs.theta - info["true_theta"])))
    assert errors, "ball never left the view cone"
    assert max(e[0] for e in errors) < 1.0
    assert max(e[1] for e in errors) < 8.0


def test_estimator_exact_without_noise():
    env = noiseless()
    obs, info = env.reset(start_at(0.0, 20.0))
    for action in [TURN_POS, TURN_POS, DASH_100, DASH_100, TURN_POS, DASH_100]:
        obs, _, _, _, info = env.step(action)
        assert obs.status != UNKNOWN
        assert math.isclose(obs.d, info["true_d"], abs_tol=1e-6)
        assert math.isclose(obs.theta, info["true_theta"], abs_tol=1e-6)


def test_greedy_controller_captures_easy_ball():
    env = noiseless(see_phase_ms=0.0)                  # first see on time at step 0
    res = rollout(env, GreedyPursuit(DEFAULT_PARAMS), start_at(0.0, 10.0))
    assert res["captured"] and res["steps"] <= 11


def test_reset_reproducible_with_seed():
    a = SimBallPursuitEnv(seed=11)
    b = SimBallPursuitEnv(seed=11)
    ra = rollout(a, GreedyPursuit(DEFAULT_PARAMS))
    rb = rollout(b, GreedyPursuit(DEFAULT_PARAMS))
    assert ra["actions"] == rb["actions"] and ra["return"] == rb["return"]


def test_late_see_is_delivered_next_cycle():
    """Phase 91 ms: sees at 91, 241, 391, 541 ms -> cycles 0 (late), 2 (41 ms), 3 (late), 5."""
    env = noiseless()
    env.reset(start_at(0.0, 20.0))
    env._see_phase_ms, env.cycle, env._late_see = 91.0, 0, None
    assert env._see_offset_ms() == 91.0
    got = []
    for _ in range(5):                                   # cycles 1..5
        body, see, late, truth = env._backend_step(("dash", 100.0))
        got.append((see is not None, late["time"] if late else None))
    assert got == [(False, None), (True, None), (False, None), (False, 3), (True, None)]


def test_estimator_uses_late_sighting_exactly():
    """With a late sighting only, the noiseless estimate is still exact after propagation."""
    env = noiseless(see_phase_ms=91.0)                 # every other sighting is late
    obs, _ = env.reset(start_at(0.0, 20.0))
    for action in [TURN_POS, DASH_100, DASH_100, TURN_POS, DASH_100, DASH_100]:
        obs, _, _, _, info = env.step(action)
        if obs.status != UNKNOWN:
            assert math.isclose(obs.d, info["true_d"], abs_tol=1e-6)
            assert math.isclose(obs.theta, info["true_theta"], abs_tol=1e-6)


def test_sensitivity_information_modes():
    from src.sensitivity import run_episode
    env = noiseless(see_phase_ms=0.0)
    policy = GreedyPursuit(DEFAULT_PARAMS)
    # Ball behind at 10 m: with sensing the player must search first; told once, it can turn at once.
    start = start_at(180.0, 10.0)
    sensing = run_episode(env, policy, start, "sensing")
    once = run_episode(env, policy, start, "once")
    truth = run_episode(env, policy, start, "truth")
    assert once is not None and truth is not None and sensing is not None
    assert once <= sensing and truth <= sensing
