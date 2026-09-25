import numpy as np

from src.agents import QLearningAgent
from src.discretizer import Discretizer, DiscretizerConfig
from src.exploration import constant_epsilon, decay_reaching, decaying_epsilon
from src.perception import Observation
from src.train import ExperimentConfig, train_one


def obs(d, theta, speed=0.0, status="SEEN"):
    return Observation(d, theta, status, speed, 0)


def test_discretizer_bins_and_boundaries():
    disc = Discretizer(DiscretizerConfig((3.0, 10.0, 20.0), (10.0, 90.0, 180.0), (0.3,)))
    assert disc.n_states == 4 * 5 * 2 + 1
    assert disc.distance_bin(2.99) == 0 and disc.distance_bin(3.0) == 1
    assert disc.distance_bin(0.5) == 0 and disc.distance_bin(55.0) == 3
    assert disc.angle_bin(10.0) == 0 and disc.angle_bin(-10.0) == 0      # boundary -> front
    assert disc.angle_bin(10.01) == 1 and disc.angle_bin(-10.01) == 2
    assert disc.angle_bin(90.0) == 1 and disc.angle_bin(90.01) == 3
    assert disc.angle_bin(180.0) == 3 and disc.angle_bin(-179.9) == 4
    assert disc.angle_bin(-180.0) == 3                                    # wraps to +180
    assert disc(obs(None, None, status="UNKNOWN")) == disc.unknown_state
    states = {disc(obs(d, t, v)) for d in (1, 5, 15, 30) for t in (0, 45, -45, 135, -135)
              for v in (0.1, 0.9)}
    assert states == set(range(disc.unknown_state))
    assert disc.decompose(disc(obs(15, -45, 0.9))) == (2, 2, 1)
    assert disc.state_to_label(disc.unknown_state) == "UNKNOWN"


def test_qlearning_update_rules():
    agent = QLearningAgent(3, 2, alpha=0.5, gamma=0.9, seed=0)
    agent.Q[1] = [2.0, 4.0]
    agent.observe(0, 0, 1.0, 1, None, terminated=False, truncated=False)
    assert np.isclose(agent.Q[0, 0], 0.5 * (1.0 + 0.9 * 4.0))
    agent.observe(0, 1, 1.0, 1, None, terminated=True, truncated=False)   # no bootstrap
    assert np.isclose(agent.Q[0, 1], 0.5)
    agent.observe(2, 0, 1.0, 1, None, terminated=False, truncated=True)   # bootstraps
    assert np.isclose(agent.Q[2, 0], 0.5 * (1.0 + 0.9 * 4.0))


def test_schedules():
    assert constant_epsilon(0.1)(5000) == 0.1
    d = decay_reaching(1.0, 0.1, 1000)
    sched = decaying_epsilon(1.0, 0.1, d)
    assert sched(0) == 1.0 and np.isclose(sched(1000), 0.1) and sched(5000) == 0.1


def test_train_one_smoke_and_determinism():
    cfg = ExperimentConfig(name="t", n_episodes=300, eval_every=150, eval_episodes=20,
                           seeds=[0], params_path=None)
    a, b = train_one(cfg, 0), train_one(cfg, 0)
    assert len(a["evals"]) == 2
    assert a["evals"] == b["evals"]
    assert 0.0 <= a["final_eval"]["capture_rate_lt40"] <= 1.0
