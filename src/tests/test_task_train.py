import os

import numpy as np

from src.task_train import (METHODS, TaskConfig, decay_schedule, matrix_configs, pick_alphas,
                            train_one)


def test_shooting_all_methods_train_and_are_deterministic():
    for method in METHODS:
        cfg = TaskConfig(name="t", task="shooting", algorithm=method, n_episodes=300,
                         eval_every=150, eval_episodes=20, seeds=[0])
        a, b = train_one(cfg, 0), train_one(cfg, 0)
        assert len(a["evals"]) == 2 and a["evals"] == b["evals"]
        m = a["final_eval"]["variants"]
        assert set(m) == {"open", "keeper"}
        for v in m.values():
            assert 0.0 <= v["success_rate"] <= 1.0
            assert v["success_rate"] == v["goal_rate"]


def test_pursuit_kit_spec_matches_train_py_metrics():
    from src.train import ExperimentConfig, KIT_DISCRETIZATIONS, train_one as old_train_one
    old = ExperimentConfig(name="o", env="kit", params_path=None, n_episodes=400, eval_every=200,
                           eval_episodes=30, alpha=0.1, discretizer=KIT_DISCRETIZATIONS["R3"],
                           schedule=decay_schedule(400), seeds=[0], extra_eval_laws=["kit"])
    new = TaskConfig(name="n", task="pursuit_kit", n_episodes=400, eval_every=200,
                     eval_episodes=30, alpha=0.1, representation="R3",
                     schedule=decay_schedule(400), seeds=[0])
    a, b = old_train_one(old, 0), train_one(new, 0)
    for eo, en in zip(a["evals"], b["evals"]):
        en = en["variants"]["uniform"]
        assert eo["capture_rate_lt40"] == en["capture_rate_lt40"] == en["success_rate"]
        assert eo["mean_return"] == en["mean_return"]
    assert a["final_eval_kit"]["capture_rate_lt40"] == b["final_eval_kit"]["success_rate"]


def test_artifacts_and_alpha_pick(tmp_path):
    out = str(tmp_path)
    for alpha in (0.03, 0.3):
        cfg = TaskConfig(name=f"shooting_sweep_qlearning_alpha{alpha:g}", task="shooting",
                         alpha=alpha, n_episodes=200, eval_every=100, eval_episodes=20, seeds=[5])
        train_one(cfg, 5, out)
    run = os.path.join(out, "shooting_sweep_qlearning_alpha0.03", "seed_5")
    for f in ("q.npy", "history.npz", "eval.json", "config.json"):
        assert os.path.exists(os.path.join(run, f))
    hist = np.load(os.path.join(run, "history.npz"))
    assert {"success", "outcome", "n_conducir", "epsilon"} <= set(hist.files)
    picked = pick_alphas(out, "shooting")
    key = "qlearning|eps_const_0.1"
    assert set(picked["table"][key]) == {"0.03", "0.3"}
    assert picked["alphas"][key] in (0.03, 0.3)
    assert picked["alphas"]["qlearning"] == picked["alphas"][key]   # single schedule swept


def test_matrix_has_every_method_and_both_schedules():
    cfgs = matrix_configs("shooting", 1000, {"qlearning": 0.3})
    assert len(cfgs) == 2 * len(METHODS)
    assert {c.schedule["kind"] for c in cfgs} == {"constant", "decay"}
    assert all(c.alpha == 0.3 for c in cfgs if c.algorithm == "qlearning")


def test_per_schedule_sweep_and_lookup():
    from src.task_train import alpha_sweep_configs, alpha_key, decay_schedule
    cfgs = alpha_sweep_configs("shooting", 1000)
    assert len(cfgs) == 3 * 2 * 3                                  # methods x schedules x alphas
    assert all(c.seeds == [5, 6, 7, 8, 9] for c in cfgs)
    assert len(alpha_sweep_configs("shooting", 1000, per_schedule=False)) == 3 * 3
    const, decay = {"kind": "constant", "eps": 0.1}, decay_schedule(1000)
    alphas = {alpha_key("sarsa", const): 0.3, alpha_key("sarsa", decay): 0.03}
    by_name = {c.name: c.alpha for c in matrix_configs("shooting", 1000, alphas)}
    assert by_name["shooting_sarsa_eps_const_0.1"] == 0.3
    assert by_name["shooting_sarsa_eps_decay_1_to_0.1"] == 0.03
