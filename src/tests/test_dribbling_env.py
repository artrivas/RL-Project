import math

import numpy as np
import pytest

from src import dribbling_env as de


def env_at(x=-30.0, y=0.0, heading=0.0, **kw):
    env = de.DribblingEnv(seed=0, **kw)
    obs, info = env.reset(de.DribbleStart(x, y, heading))
    return env, obs, info


def test_start_ball_ahead_and_observation():
    env, obs, info = env_at(-30.0, 5.0, 90.0)
    assert info["ball"] == pytest.approx((-30.0, 5.5))
    assert obs["d_b"] == pytest.approx(de.BALL_START)
    assert obs["theta_b"] == pytest.approx(0.0, abs=1e-9)
    # goal centre (52.5, 0) seen from (-30, 5) with heading 90: bearing about -93.5 deg
    assert obs["theta_g"] == pytest.approx(math.degrees(math.atan2(-5.0, 82.5)) - 90.0)


def test_kick_dash_turn_kinematics_and_reward():
    env, obs, _ = env_at(-30.0, 0.0, 0.0)
    obs, r, term, trunc, info = env.step(de.KICK)
    assert info["ball"] == pytest.approx((-27.5, 0.0))
    assert r == pytest.approx(2.0 - de.STEP_COST) and not term
    obs, r, *_ = env.step(de.DASH)
    assert info["player"] == (-30.0, 0.0) and env.px == pytest.approx(-29.2)
    assert r == pytest.approx(-de.STEP_COST)
    env.step(de.TURN_POS)
    assert env.body == pytest.approx(35.0)
    env.step(de.TURN_NEG)
    env.step(de.TURN_NEG)
    assert env.body == pytest.approx(-35.0)


def test_kick_needs_possession():
    env, *_ = env_at()
    env.step(de.KICK)                      # ball now 2.5 m ahead
    before = (env.bx, env.by)
    _, r, *_ = env.step(de.KICK)
    assert (env.bx, env.by) == before and r == pytest.approx(-de.STEP_COST)


def test_possession_boundary_matches_discretizer():
    disc = de.default_discretizer()
    assert disc.features[0].bin(de.POSSESSION) == 0          # d_b <= 0.8 is "possession"
    assert disc.features[0].bin(de.POSSESSION + 1e-9) == 1


def test_loss_is_terminal_with_penalty():
    env, *_ = env_at()
    env.step(de.KICK)                      # ball 2.5 m ahead
    for _ in range(5):                     # turn around and walk away
        env.step(de.TURN_POS)
    total, term = 0.0, False
    while not term:
        _, r, term, trunc, info = env.step(de.DASH)
        total = r
    assert info["outcome"] == de.LOST and total == pytest.approx(-de.STEP_COST + de.R_LOSS)


def test_out_of_pitch_is_terminal():
    env, *_ = env_at(-30.0, 33.0, 90.0)    # facing the touchline, 1 m away
    _, r, term, _, info = env.step(de.KICK)
    assert term and info["outcome"] == de.OUT and r == pytest.approx(-de.STEP_COST + de.R_LOSS)


def test_success_after_30_m_and_scripted_controller_never_fails():
    env = de.DribblingEnv(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(300):
        obs, _ = env.reset(de.sample_start(rng))
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(de.scripted_action(obs))
            total += r
            done = term or trunc
        assert info["outcome"] == de.SUCCESS and info["success"]
        assert info["progress"] >= de.GOAL_ADVANCE
        # return = ball advance - step costs + success bonus
        assert total == pytest.approx(info["progress"] - de.STEP_COST * env.step_count + de.R_SUCCESS)


def test_truncation_and_loss_radius_parameter():
    env, *_ = env_at(t_max=3)
    for _ in range(3):
        _, _, term, trunc, info = env.step(de.TURN_POS)
    assert trunc and not term and not info["success"]
    env, *_ = env_at(loss_radius=8.0)
    env.step(de.KICK)
    for _ in range(5):
        env.step(de.TURN_POS)
    for _ in range(4):                     # 3.2 m away from a ball 2.5 m behind: 5.7 m < 8 m
        _, _, term, _, _ = env.step(de.DASH)
    assert not term


def test_discretizer_covers_all_states():
    disc = de.default_discretizer()
    assert disc.n_states == 36
    seen = {disc({"d_b": d, "theta_b": tb, "theta_g": tg})
            for d in (0.5, 1.5, 3.0) for tb in (0, 60, -60) for tg in (0, 45, -45, 150)}
    assert seen == set(range(36))


def test_fine_discretizer_covers_all_states():
    disc = de.fine_discretizer()
    assert disc.n_states == 5 * 7 * 7
    angles = (0, 30, -30, 70, -70, 150, -150)
    seen = {disc({"d_b": d, "theta_b": tb, "theta_g": tg})
            for d in (0.5, 1.0, 1.7, 2.5, 3.5) for tb in angles for tg in angles}
    assert seen == set(range(disc.n_states))
