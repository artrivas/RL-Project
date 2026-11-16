"""Causal local features with a normalization fitted only on train."""
import math
import numpy as np

class FeatureEncoder:
    version = 1

    def __init__(self, config, price_scale):
        if not math.isfinite(price_scale) or price_scale <= 0:
            raise ValueError("price_scale must be positive")
        self.config, self.price_scale = config, float(price_scale)
        self.size = 12 + config.history_days

    @classmethod
    def fit(cls, data, config, train_end):
        if not 1 <= train_end <= data.days:
            raise ValueError("invalid training boundary")
        return cls(config, max(p for series in data.prices for p in series[:train_end]))

    def encode(self, observations):
        if observations is None:
            return np.zeros((3, self.size), dtype=np.float32)
        rows = []
        c = self.config
        for obs in observations:
            history = list(obs["sales_history"])[-c.history_days:]
            rows.append([
                obs["stock"]/c.store_capacity, obs["transit"]/c.store_capacity,
                obs["regional_stock"]/c.regional_capacity, obs["forecast"]/c.store_capacity,
                obs["base_price"]/self.price_scale, obs["days_remaining"]/1941,
                math.sin(2*math.pi*obs["weekday"]/7), math.cos(2*math.pi*obs["weekday"]/7),
                math.sin(2*math.pi*(obs["month"]-1)/12), math.cos(2*math.pi*(obs["month"]-1)/12),
                float(obs["snap"]), float(obs["event"]),
                *([0.0]*(c.history_days-len(history))), *(v/c.store_capacity for v in history)
            ])
        result = np.asarray(rows, dtype=np.float32)
        if result.shape != (3, self.size) or not np.isfinite(result).all():
            raise ValueError("nonfinite or malformed observations")
        return result

    def state_dict(self):
        return {"version": self.version, "price_scale": self.price_scale, "size": self.size}
