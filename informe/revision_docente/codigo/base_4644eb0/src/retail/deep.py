"""Independent three-store DDQN and PPO; PyTorch is required for this module."""
from collections import deque
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),
                         nn.Linear(hidden,outputs))

def double_q_target(online_next, target_next, rewards, terminals, gamma):
    """Select with online Q, evaluate with target Q (not max target Q)."""
    selected = online_next.argmax(dim=-1,keepdim=True)
    values = target_next.gather(-1,selected).squeeze(-1)
    return rewards+gamma*(1-terminals)*values

def gae(rewards, values, next_values, terminals, gamma, lam):
    """Time-major [T, stores]; rollout cuts bootstrap, true terminals do not."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        mask = 1-terminals[t]
        delta = rewards[t]+gamma*mask*next_values[t]-values[t]
        running = delta+gamma*lam*mask*running
        advantages[t] = running
    return advantages, advantages+values

def clipped_policy_loss(new_logp, old_logp, advantages, clip):
    ratio = (new_logp-old_logp).exp()
    return -torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages).mean()

class DDQN:
    def __init__(self, inputs, actions, config, device="cpu"):
        self.config,self.device,self.actions=config,torch.device(device),actions
        self.online=nn.ModuleList([mlp(inputs,config.hidden,actions) for _ in range(3)]).to(self.device)
        self.target=copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=config.learning_rate)
        self.replay=deque(maxlen=config.replay_capacity)
        self.rng=random.Random(config.seed+100)
        self.updates=0

    def act(self, observations, epsilon=0.0):
        x=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            actions=[int(net(x[i]).argmax().item()) for i,net in enumerate(self.online)]
        if epsilon:
            actions=[self.rng.randrange(self.actions) if self.rng.random()<epsilon else a for a in actions]
        return tuple(actions)

    def remember(self, obs, actions, rewards, nxt, terminal):
        # CPU tensors keep replay portable and weights_only checkpoint compatible.
        self.replay.append((torch.tensor(obs),torch.tensor(actions,dtype=torch.long),
                            torch.tensor(rewards,dtype=torch.float32),torch.tensor(nxt),
                            torch.full((3,),float(terminal))))

    def learn(self, batch=None):
        c=self.config
        if batch is None:
            if len(self.replay)<c.batch_size:
                return None
            batch=self.rng.sample(list(self.replay),c.batch_size)
        obs,actions,rewards,nxt,terminals=[torch.stack(v).to(self.device) for v in zip(*batch)]
        losses=[]
        for i in range(3):
            q=self.online[i](obs[:,i]).gather(1,actions[:,i,None]).squeeze(1)
            with torch.no_grad():
                target=double_q_target(self.online[i](nxt[:,i]),self.target[i](nxt[:,i]),
                                       rewards[:,i],terminals[:,i],c.gamma)
            losses.append(nn.functional.smooth_l1_loss(q,target))
        loss=torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),c.max_grad_norm)
        self.optimizer.step()
        self.updates+=1
        if self.updates%c.target_interval==0:
            self.target.load_state_dict(self.online.state_dict())
        return {"td_loss":float(loss.detach().cpu()),"updates":self.updates}

    def state_dict(self):
        return {"online":self.online.state_dict(),"target":self.target.state_dict(),
                "optimizer":self.optimizer.state_dict(),"replay":list(self.replay),
                "rng":self.rng.getstate(),"updates":self.updates}

    def load_state_dict(self,state):
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay=deque([tuple(t.cpu() for t in row) for row in state["replay"]],
                          maxlen=self.config.replay_capacity)
        self.rng.setstate(state["rng"])
        self.updates=state["updates"]

class PPO:
    def __init__(self,inputs,actions,config,device="cpu"):
        self.config,self.device=config,torch.device(device)
        self.actors=nn.ModuleList([mlp(inputs,config.hidden,actions) for _ in range(3)]).to(self.device)
        self.critics=nn.ModuleList([mlp(inputs,config.hidden,1) for _ in range(3)]).to(self.device)
        self.parameters=list(self.actors.parameters())+list(self.critics.parameters())
        self.optimizer=torch.optim.Adam(self.parameters,lr=config.learning_rate)
        self.buffer=[]
        self.updates=0

    def act(self,observations,deterministic=False):
        x=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            logits=torch.stack([net(x[i]) for i,net in enumerate(self.actors)])
            distribution=Categorical(logits=logits)
            actions=logits.argmax(-1) if deterministic else distribution.sample()
            values=torch.stack([net(x[i]).squeeze(-1) for i,net in enumerate(self.critics)])
        return tuple(int(a) for a in actions.cpu()), distribution.log_prob(actions).cpu(),values.cpu()

    def remember(self,obs,actions,rewards,nxt,terminal,logp,values):
        with torch.no_grad():
            x=torch.as_tensor(nxt,dtype=torch.float32,device=self.device)
            next_values=torch.stack([net(x[i]).squeeze(-1) for i,net in enumerate(self.critics)]).cpu()
        self.buffer.append((torch.tensor(obs),torch.tensor(actions),torch.tensor(rewards,dtype=torch.float32),
                            torch.full((3,),float(terminal)),logp,values,next_values))

    def learn(self):
        if not self.buffer:
            return None
        c=self.config
        obs,actions,rewards,terminals,old_logp,values,next_values=[torch.stack(v).to(self.device)
                                                               for v in zip(*self.buffer)]
        with torch.no_grad():
            advantages,returns=gae(rewards,values,next_values,terminals,c.gamma,c.gae_lambda)
            advantages=(advantages-advantages.mean(dim=0,keepdim=True))/(advantages.std(dim=0,unbiased=False,keepdim=True)+1e-8)
        records=[]
        for _ in range(c.ppo_epochs):
            indices=torch.randperm(len(obs),device=self.device)
            for start in range(0,len(obs),c.batch_size):
                ix=indices[start:start+c.batch_size]
                policy_losses,value_losses,entropies,kls,clips=[],[],[],[],[]
                for i in range(3):
                    dist=Categorical(logits=self.actors[i](obs[ix,i]))
                    logp=dist.log_prob(actions[ix,i])
                    policy_losses.append(clipped_policy_loss(logp,old_logp[ix,i],advantages[ix,i],c.clip_epsilon))
                    prediction=self.critics[i](obs[ix,i]).squeeze(-1)
                    value_losses.append((prediction-returns[ix,i]).square().mean())
                    entropies.append(dist.entropy().mean())
                    with torch.no_grad():
                        logratio=logp-old_logp[ix,i]
                        kls.append((logratio.exp()-1-logratio).mean())
                        clips.append(((logratio.exp()-1).abs()>c.clip_epsilon).float().mean())
                pl,vl,en=[torch.stack(v).mean() for v in (policy_losses,value_losses,entropies)]
                loss=pl+c.value_coefficient*vl-c.entropy_coefficient*en
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters,c.max_grad_norm)
                self.optimizer.step()
                records.append([float(x.detach().cpu()) for x in
                                (pl,vl,en,torch.stack(kls).mean(),torch.stack(clips).mean())])
                self.updates+=1
        self.buffer=[]
        means=np.mean(records,axis=0)
        return dict(zip(("policy_loss","value_loss","entropy","approx_kl","clip_fraction"),map(float,means)))

    def state_dict(self):
        if self.buffer:
            raise RuntimeError("checkpoint only after flushing PPO rollout")
        return {"actors":self.actors.state_dict(),"critics":self.critics.state_dict(),
                "optimizer":self.optimizer.state_dict(),"updates":self.updates}

    def load_state_dict(self,state):
        self.actors.load_state_dict(state["actors"])
        self.critics.load_state_dict(state["critics"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates=state["updates"]
        self.buffer=[]
