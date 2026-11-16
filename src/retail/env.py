"""Simultaneous regional simulator, independent of any RL framework."""
from collections import deque
import math
from .config import STORES

def allocate(requests, available):
    """Largest-remainder proportional integer allocation; ties use store order."""
    if type(available) is not int or available < 0 or any(type(x) is not int or x < 0 for x in requests):
        raise ValueError("allocation requires nonnegative integers")
    total = sum(requests)
    if total <= available:
        return tuple(requests)
    if not total:
        return tuple(0 for _ in requests)
    units = [available * x // total for x in requests]
    order = sorted(range(len(requests)), key=lambda i: (-(available * requests[i] % total), i))
    for i in order[:available - sum(units)]:
        units[i] += 1
    return tuple(units)

class RetailEnv:
    """One episode is a true finite horizon; observation includes days remaining.

    reset/step observations are pre-decision, after that day's arrivals.
    Today's counterfactual demand is available only in post-step diagnostics.
    step returns (observations, regional_reward, terminated, truncated, info).
    """
    def __init__(self, data, config, start_day=1, end_day=None):
        self.data, self.config = data, config
        self.start_day, self.end_day = start_day, data.days if end_day is None else end_day
        if not 1 <= self.start_day <= self.end_day <= data.days:
            raise ValueError("invalid episode range")
        self.done = True

    def reset(self):
        c = self.config
        self.day = self.start_day
        self.stock = [c.initial_store_stock] * 3
        self.transit = [0] * 3
        self.regional = c.initial_regional_stock
        self.history = [deque(self.data.sales[i][max(0, self.day-1-c.history_days):self.day-1],
                              maxlen=c.history_days) for i in range(3)]
        self.done = False
        self._arrive()
        return self.observation()

    def _arrive(self):
        self.received = tuple(self.transit)
        self.stock = [s + t for s, t in zip(self.stock, self.transit)]
        self.transit = [0] * 3
        self.external = min(self.config.regional_supply, self.config.regional_capacity - self.regional)
        self.regional += self.external

    def observation(self):
        if self.done:
            return None
        cal = self.data.calendar[self.day-1]
        return tuple({"store": store, "day": self.day,
                      "days_remaining": self.end_day-self.day+1,
                      "stock": self.stock[i], "transit": self.transit[i],
                      "regional_stock": self.regional,
                      "forecast": sum(self.history[i])/len(self.history[i]) if self.history[i] else 0.0,
                      "sales_history": tuple(self.history[i]),
                      "base_price": self.data.prices[i][self.day-1],
                      "weekday": cal["weekday"], "month": cal["month"],
                      "snap": cal["snap"], "event": cal["event"]}
                     for i, store in enumerate(STORES))

    def step(self, actions):
        if self.done:
            raise RuntimeError("reset before stepping a terminated episode")
        c = self.config
        if len(actions) != 3 or any(type(a) is not int or not 0 <= a < len(c.actions) for a in actions):
            raise ValueError("exactly three valid integer actions required")
        demand = tuple(int(math.floor(self.data.sales[i][self.day-1]*c.actions[a][0]**(-c.elasticities[i]) + 0.5))
                       for i, a in enumerate(actions))
        return self.step_model(actions, demand)

    def step_model(self, actions, demand):
        """Exact accounting with supplied demand; used only for synthetic Dyna transitions."""
        if self.done:
            raise RuntimeError("cannot step a terminal state")
        c = self.config
        if len(actions) != 3 or any(type(a) is not int or not 0 <= a < len(c.actions) for a in actions):
            raise ValueError("exactly three valid integer actions required")
        if len(demand) != 3 or any(type(d) is not int or d < 0 for d in demand):
            raise ValueError("model demand must be three nonnegative integers")
        obs = self.observation()
        before, regional_before = tuple(self.stock), self.regional
        decisions = [c.actions[a] for a in actions]
        requests = tuple(min(c.store_capacity-self.stock[i],
                             max(0, math.ceil(target*obs[i]["forecast"])-self.stock[i]))
                         for i, (_, target) in enumerate(decisions))
        dispatched = allocate(requests, self.regional)
        self.regional -= sum(dispatched)
        sold = tuple(min(self.stock[i], demand[i]) for i in range(3))
        lost = tuple(demand[i]-sold[i] for i in range(3))
        self.stock = [self.stock[i]-sold[i] for i in range(3)]
        self.transit = list(dispatched)
        rewards = tuple(sold[i]*obs[i]["base_price"]*decisions[i][0]
                        - c.holding_cost*self.stock[i] - c.shortage_cost*lost[i] for i in range(3))
        for history, sale in zip(self.history, sold):
            history.append(sale)
        self.done = self.day == self.end_day
        info = {"day": self.day, "stock_before": before, "stock_after": tuple(self.stock),
                "regional_before": regional_before, "regional_after": self.regional,
                "arrivals": self.received, "external_supply": self.external,
                "requests": requests, "dispatched": dispatched, "transit": tuple(self.transit),
                "demand": demand, "sales": sold, "lost": lost, "local_rewards": rewards,
                "base_reward": sum(rewards), "reward_label": "simulated_operating_profit_proxy",
                "terminal_inventory_has_salvage_value": False}
        self.day += 1
        if not self.done:
            self._arrive()
        return self.observation(), sum(rewards), self.done, False, info

    def snapshot(self):
        if self.done:
            raise RuntimeError("terminal state has no decision snapshot")
        return {"day": self.day, "start_day": self.start_day, "end_day": self.end_day,
                "stock": list(self.stock), "transit": list(self.transit), "regional": self.regional,
                "history": [list(h) for h in self.history], "received": list(self.received),
                "external": self.external}

    @classmethod
    def from_snapshot(cls, data, config, state):
        env = cls(data, config, state["start_day"], state["end_day"])
        env.day, env.done = state["day"], False
        env.stock, env.transit = list(state["stock"]), list(state["transit"])
        env.regional = state["regional"]
        env.history = [deque(h, maxlen=config.history_days) for h in state["history"]]
        env.received, env.external = tuple(state["received"]), state["external"]
        return env
