"""Explicit simulator assumptions; M5 does not provide inventory or costs."""
from dataclasses import dataclass, asdict
import math

STORES = ("CA_1", "CA_2", "CA_3")

@dataclass(frozen=True)
class RetailConfig:
    store_capacity: int
    regional_capacity: int
    regional_supply: int
    initial_store_stock: int
    initial_regional_stock: int
    holding_cost: float
    shortage_cost: float
    elasticities: tuple
    price_multipliers: tuple = (0.9, 0.95, 1.0, 1.05, 1.1)
    target_days: tuple = (0, 3, 7)
    history_days: int = 28

    def __post_init__(self):
        for name in ("store_capacity", "regional_capacity", "regional_supply",
                     "initial_store_stock", "initial_regional_stock", "history_days"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if min(self.store_capacity, self.regional_capacity, self.history_days) <= 0:
            raise ValueError("capacities and history_days must be positive")
        if self.initial_store_stock > self.store_capacity or self.initial_regional_stock > self.regional_capacity:
            raise ValueError("initial inventory exceeds capacity")
        for name in ("elasticities", "price_multipliers", "target_days"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.elasticities) != 3 or not self.price_multipliers or not self.target_days:
            raise ValueError("three elasticities and nonempty action grids required")
        for value in (*self.elasticities, *self.target_days, self.holding_cost, self.shortage_cost):
            if not math.isfinite(value) or value < 0:
                raise ValueError("costs, elasticities and targets must be finite and nonnegative")
        if any(not math.isfinite(x) or x <= 0 for x in self.price_multipliers):
            raise ValueError("price multipliers must be positive and finite")
        if 1.0 not in self.price_multipliers:
            raise ValueError("historical-price baseline requires multiplier 1.0")
        if len(set(self.price_multipliers)) != len(self.price_multipliers) or len(set(self.target_days)) != len(self.target_days):
            raise ValueError("action grids must not contain duplicates")

    @property
    def actions(self):
        return tuple((p, d) for p in self.price_multipliers for d in self.target_days)

    def as_dict(self):
        return asdict(self)
