import math

import numpy as np
import pytest

from src import shooting_env as se
from src.shooting_optimal import Chains, outcome_probs


def env_at(d, y, ky, seed=0, **kw):
    env = se.ShootingEnv(seed=seed, **kw)
    env.reset(se.ShootStart(d, y, ky))
    return env


def test_noiseless_shots_hit_their_target():
    for target_id, target in enumerate(se.TARGETS):
        env = env_at(15.0, 4.0, None, sigma_deg=1e-9)
        _, r, term, trunc, info = env.step(target_id)
        assert term and not trunc
        assert info["shot_end"][1] == pytest.approx(target, abs=1e-6)
        assert info["outcome"] == se.GOAL and r == se.R_GOAL and info["success"]


def test_off_target_and_saved():
    assert se.shot_outcome(7.01, None, 2.0) == se.OFF_TARGET        # the post counts as out
    assert se.shot_outcome(None, None, 2.0) == se.OFF_TARGET
    assert se.shot_outcome(0.5, 0.0, 2.0) == se.SAVED
    assert se.shot_outcome(2.0, 0.0, 2.0) == se.GOAL                # reach is strict
    env = env_at(15.0, 0.0, 0.0, sigma_deg=1e-9)                    # keeper in the middle
    _, r, term, _, info = env.step(2)                               # centre shot
    assert info["outcome"] == se.SAVED and r == se.R_SAVED and term


def test_conducir_moves_and_floors():
    env = env_at(14.0, 3.0, None)
    _, r, term, trunc, info = env.step(se.CONDUCIR)
    assert r == se.R_CONDUCIR and not term and not trunc
    assert info["d_g"] == pytest.approx(12.0) and info["y"] == 3.0
    env.step(se.CONDUCIR)
    assert env.d == se.D_MIN
    env.step(se.CONDUCIR)
    assert env.d == se.D_MIN                                        # no-op at the floor


def test_truncation_after_t_max_conducir_without_keeper():
    env = env_at(25.0, 0.0, None)
    for t in range(se.T_MAX):
        _, _, term, trunc, info = env.step(se.CONDUCIR)
        assert not term
    assert trunc and info["outcome"] == se.RUNNING and not info["success"]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_keeper_blocks_conducir_with_probability_p_out():
    blocked = 0
    n = 4000
    env = se.ShootingEnv(seed=3, p_keeper_out=0.25)
    for _ in range(n):
        env.reset(se.ShootStart(20.0, 0.0, 1.0))
        _, r, term, _, info = env.step(se.CONDUCIR)
        if term:
            assert info["outcome"] == se.BLOCKED and r == se.R_CONDUCIR + se.R_SAVED
            blocked += 1
    assert abs(blocked / n - 0.25) < 0.03


def test_exact_probabilities_match_simulation():
    params = se.ShootingParams(sigma_deg=6.0, keeper_reach=2.0)
    cases = [(22.0, -8.0, 6.0, 1.5), (13.0, 5.0, -3.5, None), (18.0, 0.0, 3.5, 2.5)]
    env = se.ShootingEnv(seed=11, params=params)
    for d, y, target, ky in cases:
        pg, po, ps = outcome_probs(d, y, target, np.nan if ky is None else ky, params)
        assert pg + po + ps == pytest.approx(1.0)
        counts = np.zeros(5)
        n = 20_000
        a = se.TARGETS.index(target)
        for _ in range(n):
            env.reset(se.ShootStart(d, y, ky))
            counts[env.step(a)[4]["outcome"]] += 1
        tol = 4 * math.sqrt(0.25 / n)
        assert counts[se.GOAL] / n == pytest.approx(float(pg), abs=tol)
        assert counts[se.OFF_TARGET] / n == pytest.approx(float(po), abs=tol)
        assert counts[se.SAVED] / n == pytest.approx(float(ps), abs=tol)


def test_true_optimum_is_at_least_every_fixed_policy():
    params = se.ShootingParams()
    rng = np.random.default_rng(0)
    ch = Chains([se.sample_start(rng) for _ in range(300)], params)
    opt = ch.true_optimum()
    rows = np.arange(300)
    pols = np.array([[a, b, c] for a in range(6) for b in range(6) for c in range(6)])
    v, _ = ch.policy_value(pols, rows)
    assert np.all(opt["value"][None, :] >= v - 1e-9)


def test_evaluation_starts_variants_and_determinism():
    open_, keeper = se.evaluation_starts(50, 1, "open"), se.evaluation_starts(50, 1, "keeper")
    assert all(s.keeper_y is None for s in open_)
    assert all(s.keeper_y is not None and abs(s.keeper_y) <= 3 for s in keeper)
    # same (d, y) in both variants: only the keeper differs
    assert [(s.d, s.y) for s in open_] == [(s.d, s.y) for s in keeper]
    assert se.evaluation_starts(50, 1, "open") == open_


def test_discretizer_covers_all_states():
    disc = se.default_discretizer()
    assert disc.n_states == 36
    seen = {disc({"d_g": d, "y": y, "keeper": k})
            for d in (12, 17, 22) for y in (-5, 0, 5) for k in (None, -2, 0, 2)}
    assert seen == set(range(36))
