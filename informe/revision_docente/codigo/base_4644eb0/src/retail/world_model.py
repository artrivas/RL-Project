"""One-step hybrid Dyna: learned demand, exact joint inventory/accounting.

Demand targets are privileged POST-transition simulator diagnostics, from train
only. The agent never observes them before acting. No test trajectory is fitted.
Known calendar and reference-price schedules remain exogenous simulator inputs.
"""
from collections import deque
import math
import random
import torch
from torch import nn
from .deep import mlp
from .env import RetailEnv
from .shaping import shaped_rewards

class DemandModel:
    def __init__(self,encoder,actions,config,device="cpu"):
        self.encoder,self.actions,self.config=encoder,actions,config
        self.device=torch.device(device)
        self.net=mlp(encoder.size+actions+3,config.hidden,1).to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=config.learning_rate)
        self.real=deque(maxlen=config.replay_capacity)
        self.rng=random.Random(config.seed+200)
        self.synthetic_steps=0

    def inputs(self,observations,actions):
        x=torch.as_tensor(self.encoder.encode(observations),device=self.device)
        a=nn.functional.one_hot(torch.tensor(actions,device=self.device),self.actions).float()
        return torch.cat((x,a,torch.eye(3,device=self.device)),dim=-1)

    def observe_real(self,snapshot,observations,actions,demand):
        x=self.inputs(observations,actions).detach().cpu()
        target=torch.log1p(torch.tensor(demand,dtype=torch.float32))
        self.real.append({"snapshot":snapshot,"x":x,"target":target})

    def learn(self):
        c=self.config
        if len(self.real)<max(c.model_warmup,c.batch_size):
            return None
        rows=self.rng.sample(list(self.real),c.batch_size)
        x=torch.cat([r["x"] for r in rows]).to(self.device)
        target=torch.cat([r["target"] for r in rows]).to(self.device)
        prediction=nn.functional.softplus(self.net(x).squeeze(-1))
        loss=nn.functional.mse_loss(prediction,target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(),c.max_grad_norm)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def predict(self,observations,actions):
        with torch.no_grad():
            log_demand=nn.functional.softplus(self.net(self.inputs(observations,actions)).squeeze(-1))
            units=torch.expm1(log_demand.clamp(max=math.log1p(10*self.encoder.config.store_capacity)))
        return tuple(int(math.floor(x+0.5)) for x in units.cpu().tolist())

    def plan(self,agent,data,retail_config):
        if len(self.real)<max(self.config.model_warmup,self.config.batch_size):
            return []
        losses=[]
        for _ in range(self.config.dyna_steps):
            row=self.rng.choice(list(self.real))
            env=RetailEnv.from_snapshot(data,retail_config,row["snapshot"])
            obs=env.observation()
            actions=agent.act(self.encoder.encode(obs),epsilon=0.1)
            demand=self.predict(obs,actions)
            nxt,base,terminal,_,info=env.step_model(actions,demand)
            rewards,_=shaped_rewards(base,info["local_rewards"],obs,nxt,terminal,
                                    self.config.gamma,self.config.shaping,self.config.shaping_weight,
                                    self.config.reward_mode,self.config.reward_scale)
            # Synthetic data is NEVER inserted in the real replay or model-fit buffer.
            transition=(torch.tensor(self.encoder.encode(obs)),torch.tensor(actions),
                        torch.tensor(rewards,dtype=torch.float32),torch.tensor(self.encoder.encode(nxt)),
                        torch.full((3,),float(terminal)))
            result=agent.learn([transition])
            self.synthetic_steps+=1
            if result is not None:
                losses.append(result["td_loss"])
        return losses

    def state_dict(self):
        return {"net":self.net.state_dict(),"optimizer":self.optimizer.state_dict(),
                "real":list(self.real),"rng":self.rng.getstate(),"synthetic_steps":self.synthetic_steps}

    def load_state_dict(self,state):
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.real=deque([dict(row,x=row["x"].cpu(),target=row["target"].cpu())
                         for row in state["real"]],maxlen=self.config.replay_capacity)
        self.rng.setstate(state["rng"])
        self.synthetic_steps=state["synthetic_steps"]
