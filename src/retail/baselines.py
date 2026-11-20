"""Untrained controls; none represents observed Walmart inventory policy."""
import random

class HistoricalPriceBaseStock:
    name = "H0_historical_price_base_stock_proxy"

    def __init__(self, config, target_days=7):
        try:
            self.action = config.actions.index((1.0, target_days))
        except ValueError as exc:
            raise ValueError("baseline target must exist in the action grid") from exc

    def act(self, observations):
        return (self.action,) * 3

class RandomPolicy:
    name = "random_smoke_control"

    def __init__(self, config, seed):
        self.rng = random.Random(seed)
        self.n = len(config.actions)

    def act(self, observations):
        return tuple(self.rng.randrange(self.n) for _ in observations)
