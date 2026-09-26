import math

import numpy as np
import pytest

from src import passing_env as pe


def env_with(holder, heading, mate, defender, **kw):
    env = pe.PassingEnv(seed=0, **kw)
    obs, info = env.reset(pe.PassStart(holder, heading, mate, defender))
    return env, obs


def test_starts_inside_zone_with_requested_distances():
    for s in pe.evaluation_starts(300, 3):
        assert pe.inside(s.holder) and pe.inside(s.mate) and pe.inside(s.defender)
        assert 6.0 <= math.dist(s.holder, s.mate) <= 10.0
        assert 4.0 <= math.dist(s.holder, s.defender) <= 8.0
    assert pe.evaluation_starts(20, 3) == pe.evaluation_starts(20, 3)


def test_segment_distance():
    assert pe.seg_dist((0, 1), (-1, 0), (1, 0)) == pytest.approx(1.0)
    assert pe.seg_dist((3, 0), (-1, 0), (1, 0)) == pytest.approx(2.0)     # beyond the end


def test_open_pass_switches_control():
    env, obs = env_with((0.0, 0.0), 90.0, (6.0, 0.0), (-5.0, 0.0))
    assert obs["blocked"] == 0.0
    _, r, term, _, info = env.step(pe.PASE)
    assert r == pe.R_PASS and not term and info["passes"] == 1
    assert info["holder"] == (6.0, 0.0)                    # receiver holds the ball
    assert info["heading"] == pytest.approx(0.0)           # facing the pass direction


def test_blocked_pass_is_intercepted():
    env, obs = env_with((0.0, 0.0), 0.0, (8.0, 0.0), (4.0, 0.5), p_intercept=1.0)
    assert obs["blocked"] == 1.0
    _, r, term, _, info = env.step(pe.PASE)
    assert term and info["outcome"] == pe.INTERCEPTED and r == pe.R_LOSS and not info["success"]


def test_dribble_moves_one_metre_and_out_of_zone_ends():
    env, _ = env_with((0.0, 0.0), 0.0, (0.0, 7.0), (-6.0, -2.0))
    env.step(pe.DRIBLE)
    assert env.holder == pytest.approx((1.0, 0.0))
    env, _ = env_with((14.5, 0.0), 0.0, (8.0, 5.0), (8.0, -5.0))
    _, r, term, _, info = env.step(pe.DRIBLE)
    assert term and info["outcome"] == pe.OUT and r == pe.R_OUT


def test_girar_turns_away_and_toward_centre_when_defender_behind():
    env, _ = env_with((0.0, 0.0), 0.0, (0.0, 7.0), (5.0, 1.0))      # defender in front, right
    env.step(pe.GIRAR)
    assert env.heading == pytest.approx(-35.0)                      # turned away (to the left)
    env, _ = env_with((10.0, -8.0), -60.0, (3.0, -3.0), (10.0 - 3 * math.cos(math.radians(-60)),
                                                           -8.0 - 3 * math.sin(math.radians(-60))))
    theta_c = pe._bearing((10.0, -8.0), (0.0, 0.0), -60.0)
    env.step(pe.GIRAR)                                              # defender straight behind
    assert env.heading == pytest.approx(pe.wrap_deg(-60.0 + math.copysign(35.0, theta_c)))


def test_tackle_and_clearance_are_terminal():
    env, _ = env_with((0.0, 0.0), 0.0, (0.0, 7.0), (-0.9, 0.0), p_tackle=1.0)
    _, r, term, _, info = env.step(pe.GIRAR)
    assert term and info["outcome"] == pe.TACKLED and r == pe.R_LOSS
    env, _ = env_with((0.0, 0.0), 0.0, (0.0, 7.0), (-6.0, 0.0))
    _, r, term, _, info = env.step(pe.DESPEJE)
    assert term and info["outcome"] == pe.CLEARED and r == 0.0 and info["possession"] == 0


def test_success_needs_possession_and_three_passes():
    env = pe.PassingEnv(seed=1, t_max=pe.T_MAX, p_tackle=0.0, p_intercept=0.0)
    obs, _ = env.reset(pe.PassStart((0.0, 0.0), 0.0, (6.0, 0.0), (-6.0, 6.0)))
    done, passes = False, 0
    while not done:
        # pass three times, then just turn in place (never loses without tackles/interceptions)
        a = pe.PASE if passes < 3 else pe.GIRAR
        obs, r, term, trunc, info = env.step(a)
        passes = info["passes"]
        done = term or trunc
    assert trunc and info["possession"] == pe.T_MAX and info["success"]
    env = pe.PassingEnv(seed=1, p_tackle=0.0)
    env.reset(pe.PassStart((0.0, 0.0), 0.0, (6.0, 0.0), (-6.0, 6.0)))
    for _ in range(pe.T_MAX):
        *_, trunc, info = env.step(pe.GIRAR)
    assert trunc and not info["success"]                           # no passes


def test_discretizer_covers_all_states():
    disc = pe.default_discretizer()
    assert disc.n_states == 324
    seen = {disc({"d_comp": dc, "theta_comp": tc, "d_def": dd, "theta_def": td, "blocked": b, "edge": e})
            for dc in (3, 7, 12) for tc in (0, 90, 170) for dd in (1, 3, 6) for td in (0, -90, 170)
            for b in (0.0, 1.0) for e in (1.0, 5.0)}
    assert seen == set(range(324))


def test_fine_discretizer_size_and_centre_bearing():
    disc = pe.fine_discretizer()
    assert disc.n_states == 3 * 3 * 4 * 3 * 2 * 3 * 3
    env, obs = env_with((10.0, 0.0), 0.0, (4.0, 3.0), (4.0, -3.0))
    assert abs(obs["theta_centre"]) == pytest.approx(180.0)       # centre straight behind
    assert 0 <= disc(obs) < disc.n_states
