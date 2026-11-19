"""Bounded PBRS and an intentionally naive contrast; base reward is immutable."""
import math

def potential(observations):
    if observations is None:
        return 0.0
    coverage = [min(14.0, (o["stock"]+o["transit"])/max(1.0,o["forecast"])) for o in observations]
    mean = sum(coverage)/3
    variance = sum((x-mean)**2 for x in coverage)/3
    shortfall = sum(max(0.0, 3.0-x)/3 for x in coverage)/3
    return -(variance/196 + shortfall)

def shaped_rewards(base_reward, local_rewards, observations, next_observations,
                   terminated, gamma, mode="none", weight=1.0, reward_mode="regional", scale=1.0):
    if mode not in ("none","pbrs","naive") or reward_mode not in ("regional","local"):
        raise ValueError("unknown reward mode")
    if not 0 <= gamma <= 1 or not math.isfinite(weight) or weight < 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid shaping parameters")
    if reward_mode == "local" and mode != "none":
        raise ValueError("selfish control must keep unshaped local rewards")
    if mode == "pbrs":
        bonus = weight*(gamma*(0.0 if terminated else potential(next_observations))-potential(observations))
    elif mode == "naive":
        # Positive coverage bonus on every day: encourages inventory hoarding.
        # An intentional objective change, not a PBRS approximation.
        bonus = weight*sum(min(14.0,o["stock"]/max(1.0,o["forecast"]))/14 for o in observations)/3
    else:
        bonus = 0.0
    base = [base_reward]*3 if reward_mode == "regional" else list(local_rewards)
    return [(r+bonus)/scale for r in base], bonus
