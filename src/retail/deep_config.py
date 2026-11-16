"""Serializable experimental choices shared by independent DDQN/PPO learners."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class DeepConfig:
    algorithm: str = "ddqn"
    seed: int = 0
    horizon: int = 28
    hidden: int = 64
    gamma: float = 0.99
    learning_rate: float = 0.0003
    reward_scale: float = 100.0
    reward_mode: str = "regional"
    shaping: str = "none"
    shaping_weight: float = 1.0
    batch_size: int = 64
    replay_capacity: int = 20000
    warmup_steps: int = 128
    target_interval: int = 100
    epsilon_decay_steps: int = 10000
    epsilon_final: float = 0.05
    rollout_steps: int = 128
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    dyna_steps: int = 0
    model_warmup: int = 256

    def __post_init__(self):
        if self.algorithm not in ("ddqn","ppo") or self.reward_mode not in ("regional","local"):
            raise ValueError("invalid algorithm/reward_mode")
        if self.shaping not in ("none","pbrs","naive"):
            raise ValueError("invalid shaping")
        if self.reward_mode=="local" and self.shaping!="none":
            raise ValueError("selfish baseline does not use shaping")
        if self.algorithm!="ddqn" and self.dyna_steps:
            raise ValueError("Dyna is implemented only for DDQN")
        for name in ("horizon","hidden","batch_size","replay_capacity","warmup_steps",
                     "target_interval","epsilon_decay_steps","rollout_steps","ppo_epochs","model_warmup"):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed<0 or type(self.dyna_steps) is not int or not 0<=self.dyna_steps<=10:
            raise ValueError("invalid seed or dyna_steps")
        if self.replay_capacity<self.batch_size:
            raise ValueError("replay capacity smaller than minibatch")
        for name in ("gamma","gae_lambda","epsilon_final"):
            if not math.isfinite(getattr(self,name)) or not 0<=getattr(self,name)<=1:
                raise ValueError(f"invalid {name}")
        for name in ("learning_rate","reward_scale","max_grad_norm","clip_epsilon"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"invalid {name}")
        if self.clip_epsilon>=1:
            raise ValueError("clip_epsilon must be below one")
        for name in ("shaping_weight","entropy_coefficient","value_coefficient"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f"invalid {name}")

    def as_dict(self):
        return asdict(self)
