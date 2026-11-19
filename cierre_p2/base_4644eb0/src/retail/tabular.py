"""P1 Q-learning with fresh retail tables and three independent learners."""
from bisect import bisect_right
import numpy as np
from src.agents import QLearningAgent

class RetailDiscretizer:
    """Coarse local observation aggregation, not a claim of sufficient Markov state."""
    sizes = (5, 4, 4, 7, 4)
    n_states = 5*4*4*7*4

    def __call__(self, obs):
        forecast = max(1.0, obs["forecast"])
        components = (bisect_right((0.01, 1, 3, 7), obs["stock"]/forecast),
                      bisect_right((1, 5, 10), obs["forecast"]),
                      bisect_right((1, 3, 7), obs["regional_stock"]/(3*forecast)),
                      obs["weekday"],
                      bisect_right((7, 30, 90), obs["days_remaining"]))
        state = 0
        for value, size in zip(components, self.sizes):
            state = state*size+value
        return state

class CooperativeQLearning:
    name = "T0_cooperative_qlearning"

    def __init__(self, config, seed=0, alpha=0.1, gamma=0.99):
        self.discretizer = RetailDiscretizer()
        self.agents = tuple(QLearningAgent(self.discretizer.n_states, len(config.actions),
                                          alpha=alpha, gamma=gamma, seed=seed+i) for i in range(3))

    def act(self, observations, epsilon=0.0):
        states = tuple(self.discretizer(o) for o in observations)
        if epsilon == 0:
            # Stable ties during evaluation; no RNG or table updates.
            return tuple(int(agent.Q[s].argmax()) for agent,s in zip(self.agents,states))
        return tuple(agent.act(s, epsilon) for agent,s in zip(self.agents,states))

    def observe(self, observations, actions, regional_reward, next_observations, terminated):
        for i, agent in enumerate(self.agents):
            state = self.discretizer(observations[i])
            next_state = 0 if terminated else self.discretizer(next_observations[i])
            agent.observe(state, actions[i], regional_reward, next_state, None,
                          terminated=terminated, truncated=False)

    def save(self, path):
        np.savez_compressed(path, **{f"store_{i}":agent.Q for i,agent in enumerate(self.agents)})
