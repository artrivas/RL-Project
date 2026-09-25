"""Tabular agents sharing one interface; they differ only in the update rule.

Q-learning is the Milestone 2 baseline. SARSA and first-visit MC control
(the optional internal comparison, Milestone 3) plug into the same interface:

* ``act(state, epsilon)`` — epsilon-greedy with random tie-breaking;
* ``observe(s, a, r, s_next, a_next, terminated, truncated)`` after every step;
* ``end_episode()`` after the last step.

TD rule (``src/SPEC.md`` §6): terminal transitions do not bootstrap; truncation
(step cap) bootstraps from ``s_next`` because elapsed time is not in the state.
"""

from typing import Optional

import numpy as np


class TabularAgent:
    name = "tabular"
    needs_next_action = False  # SARSA needs a_next before its update

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.99,
                 seed: Optional[int] = None, q_init: float = 0.0):
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.Q = np.full((n_states, n_actions), q_init, dtype=np.float64)

    def greedy(self, state: int) -> int:
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def act(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(state)

    def observe(self, s: int, a: int, r: float, s_next: int, a_next: Optional[int],
                terminated: bool, truncated: bool) -> None:
        raise NotImplementedError

    def end_episode(self) -> None:
        pass


class QLearningAgent(TabularAgent):
    name = "qlearning"

    def observe(self, s, a, r, s_next, a_next, terminated, truncated):
        target = r if terminated else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class SarsaAgent(TabularAgent):
    """On-policy TD: bootstraps from the action actually selected in ``s_next``."""
    name = "sarsa"
    needs_next_action = True  # also drawn at truncation, to bootstrap from s_next

    def observe(self, s, a, r, s_next, a_next, terminated, truncated):
        target = r if terminated else r + self.gamma * self.Q[s_next, a_next]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class MonteCarloAgent(TabularAgent):
    """On-policy first-visit MC control with sample-average returns (``alpha`` is unused).

    Returns are computed backwards at the end of the episode. A truncated episode's
    return is simply cut at the step cap: MC does not bootstrap (SPEC §6).
    """
    name = "mc_first_visit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = np.zeros_like(self.Q)
        self._episode = []

    def observe(self, s, a, r, s_next, a_next, terminated, truncated):
        self._episode.append((s, a, r))

    def end_episode(self) -> None:
        first = {}
        for t, (s, a, _) in enumerate(self._episode):
            first.setdefault((s, a), t)
        g = 0.0
        for t in range(len(self._episode) - 1, -1, -1):
            s, a, r = self._episode[t]
            g = self.gamma * g + r
            if first[(s, a)] == t:
                self.counts[s, a] += 1
                self.Q[s, a] += (g - self.Q[s, a]) / self.counts[s, a]
        self._episode = []


class MonteCarloConstantAlphaAgent(MonteCarloAgent):
    """First-visit MC control with a constant step size ``alpha`` instead of sample averages.

    With sample averages the step 1/N vanishes, so returns collected early under a
    near-random policy (epsilon ~ 1) keep full weight forever and the greedy policy
    freezes; a constant alpha weights recent returns more, as the TD agents do.
    """
    name = "mc_first_visit_alpha"

    def end_episode(self) -> None:
        first = {}
        for t, (s, a, _) in enumerate(self._episode):
            first.setdefault((s, a), t)
        g = 0.0
        for t in range(len(self._episode) - 1, -1, -1):
            s, a, r = self._episode[t]
            g = self.gamma * g + r
            if first[(s, a)] == t:
                self.counts[s, a] += 1
                self.Q[s, a] += self.alpha * (g - self.Q[s, a])
        self._episode = []


AGENTS = {cls.name: cls for cls in (QLearningAgent, SarsaAgent, MonteCarloAgent,
                                    MonteCarloConstantAlphaAgent)}


class GreedyQPolicy:
    """Deterministic greedy policy from a Q-table (evaluation: epsilon = 0, no learning)."""

    def __init__(self, Q: np.ndarray, discretizer):
        self.Q, self.discretizer = Q, discretizer

    def __call__(self, obs) -> int:
        return int(np.argmax(self.Q[self.discretizer(obs)]))
