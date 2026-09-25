import math

import numpy as np

from src.feasibility import dash_cycles, turn_then_dash_cycles
from src.params import DEFAULT_PARAMS
from src.sampler import HALF_LENGTH, HALF_WIDTH, evaluation_starts, sample_start, wrap_deg


def test_wrap_deg_range_and_boundaries():
    assert wrap_deg(180.0) == 180.0
    assert wrap_deg(-180.0) == 180.0
    assert wrap_deg(540.0) == 180.0
    assert wrap_deg(-190.0) == 170.0
    assert wrap_deg(0.0) == 0.0


def test_starts_inside_pitch_and_consistent():
    rng = np.random.default_rng(1)
    margin = 1.0
    for _ in range(5000):
        s = sample_start(rng, margin=margin)
        for x, y in ((s.player_x, s.player_y), (s.ball_x, s.ball_y)):
            assert abs(x) <= HALF_LENGTH - margin + 1e-9
            assert abs(y) <= HALF_WIDTH - margin + 1e-9
        dx, dy = s.ball_x - s.player_x, s.ball_y - s.player_y
        assert math.isclose(math.hypot(dx, dy), s.distance, rel_tol=1e-9)
        rel = wrap_deg(math.degrees(math.atan2(dy, dx)) - s.heading)
        assert abs(wrap_deg(rel - s.bearing)) < 1e-6


def test_relative_marginals_uniform():
    rng = np.random.default_rng(2)
    starts = [sample_start(rng) for _ in range(20000)]
    d = np.array([s.distance for s in starts])
    b = np.array([s.bearing for s in starts])
    # Deciles of U[5,40] and U[-180,180] within sampling tolerance.
    assert np.allclose(np.quantile(d, np.linspace(0.1, 0.9, 9)),
                       np.linspace(8.5, 36.5, 9), atol=0.6)
    assert np.allclose(np.quantile(b, np.linspace(0.1, 0.9, 9)),
                       np.linspace(-144, 144, 9), atol=6.0)


def test_evaluation_starts_reproducible():
    assert evaluation_starts(20, seed=7) == evaluation_starts(20, seed=7)
    assert evaluation_starts(20, seed=7) != evaluation_starts(20, seed=8)


def test_dash_kinematics_default_params():
    # From rest: 0.6, 0.84, 0.936, ... -> about 1 m per cycle.
    assert dash_cycles(0.6, DEFAULT_PARAMS) == 1
    assert dash_cycles(1.44, DEFAULT_PARAMS) == 2
    assert dash_cycles(39.2, DEFAULT_PARAMS) == 40


def test_turn_then_dash_counts_turns():
    # Relaxed turns: residual after turning must be within asin(0.8/10) = 4.59 deg.
    ahead = turn_then_dash_cycles(10.0, 0.0, DEFAULT_PARAMS)
    assert turn_then_dash_cycles(10.0, 4.5, DEFAULT_PARAMS) == ahead
    assert turn_then_dash_cycles(10.0, 39.5, DEFAULT_PARAMS) == ahead + 1
    assert turn_then_dash_cycles(10.0, 40.0, DEFAULT_PARAMS) == ahead + 2
    assert turn_then_dash_cycles(10.0, -180.0, DEFAULT_PARAMS) == ahead + 6


def test_provable_min_cycles():
    from src.feasibility import provable_min_cycles
    ahead = provable_min_cycles(10.0, 0.0, DEFAULT_PARAMS)
    assert provable_min_cycles(10.0, 89.0, DEFAULT_PARAMS) == ahead      # no turn needed to progress
    assert provable_min_cycles(10.0, 125.0, DEFAULT_PARAMS) == ahead + 1
    assert provable_min_cycles(10.0, -180.0, DEFAULT_PARAMS) == ahead + 3


def test_distance_distributions():
    from src.sampler import DISTANCE_DISTRIBUTIONS
    # Default path unchanged: same draws as an explicit "uniform".
    a = [sample_start(np.random.default_rng(3)) for _ in range(5)]
    b = [sample_start(np.random.default_rng(3), distance_dist="uniform") for _ in range(5)]
    assert a == b
    means = {}
    for kind in DISTANCE_DISTRIBUTIONS:
        rng = np.random.default_rng(4)
        d = np.array([sample_start(rng, distance_dist=kind).distance for _ in range(20000)])
        assert d.min() >= 5.0 and d.max() <= 40.0
        means[kind] = d.mean()
    assert abs(means["uniform"] - 22.5) < 0.3
    assert abs(means["area"] - (2 / 3) * (40**3 - 5**3) / (40**2 - 5**2)) < 0.3    # 26.9
    assert abs(means["near"] - (5 + 5 + 40) / 3) < 0.3                             # 16.7


def test_kit_start_matches_starter_kit_reset():
    from src.sampler import draw_start, kit_start
    rng = np.random.default_rng(0)
    starts = [kit_start(rng) for _ in range(5000)]
    d = np.array([s.distance for s in starts])
    assert 10.5 < d.min() and d.max() < 19.9                    # starter kit: about 11-19 m
    for s in starts[:50]:
        assert -17.0 <= s.player_x <= -13.0 and abs(s.player_y) <= 4.0
        assert abs(s.ball_x) <= 2.0 and abs(s.ball_y) <= 3.0
        rel = wrap_deg(math.degrees(math.atan2(s.ball_y - s.player_y, s.ball_x - s.player_x)) - s.heading)
        assert abs(wrap_deg(rel - s.bearing)) < 1e-9
    assert draw_start(np.random.default_rng(3)) == sample_start(np.random.default_rng(3))
