"""DDQN/PPO experiments. Train/validation only; final test has a separate CLI."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import time
import torch
from .config import RetailConfig
from .data import load_m5, fingerprint
from .deep_config import DeepConfig
from .features import FeatureEncoder
from .deep import DDQN, PPO
from .world_model import DemandModel
from .checkpoint import save_checkpoint, load_checkpoint, rng_state, restore_rng
from .env import RetailEnv
from .metrics import summarize
from .rollout import demo_data, demo_config
from .shaping import shaped_rewards

def code_fingerprint():
    return {p.name:fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))}

def canonical(value):
    return json.dumps(value,sort_keys=True)

def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)

def evaluate_policy(agent,encoder,data,config,bounds,algorithm):
    env=RetailEnv(data,config,*bounds)
    obs,rows=env.reset(),[]
    while obs is not None:
        features=encoder.encode(obs)
        actions=agent.act(features,deterministic=True)[0] if algorithm=="ppo" else agent.act(features)
        obs,_,_,_,info=env.step(actions)
        rows.append(info)
    return summarize(rows)

def run(data,retail_config,settings,output,episodes,eval_every=10,device="cpu",resume=None):
    output=Path(output)
    if episodes<=0 or eval_every<=0:
        raise ValueError("episodes and eval_every must be positive")
    synthetic=data.manifest.get("source")=="synthetic_test_fixture"
    train_end,validation=(60,(61,90)) if synthetic else (1211,(1212,1576))
    first=retail_config.history_days+1
    if first+settings.horizon-1>train_end:
        raise ValueError("history and horizon do not fit in train")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    encoder=FeatureEncoder.fit(data,retail_config,train_end)
    klass=DDQN if settings.algorithm=="ddqn" else PPO
    agent=klass(encoder.size,len(retail_config.actions),settings,device)
    model=DemandModel(encoder,len(retail_config.actions),settings,device) if settings.dyna_steps else None
    windows=random.Random(settings.seed+300)
    metadata={"settings":settings.as_dict(),"retail_config":retail_config.as_dict(),
              "encoder":encoder.state_dict(),"data":data.manifest,"train_end":train_end,
              "validation_range":validation,"code_sha256":code_fingerprint(),
              "device":device,"torch_version":str(torch.__version__)}
    steps,completed,history=0,0,[]
    if resume is None:
        output.mkdir(parents=True,exist_ok=False)
    else:
        checkpoint=load_checkpoint(resume,device)
        if canonical(checkpoint["metadata"])!=canonical(metadata):
            raise ValueError("resume requires identical code, data, config, device and PyTorch version")
        if not output.is_dir() or Path(resume).resolve().parent!=output.resolve():
            raise ValueError("resume checkpoint must belong to the existing output directory")
        agent.load_state_dict(checkpoint["agent"])
        if model:
            model.load_state_dict(checkpoint["model"])
        windows.setstate(checkpoint["window_rng"])
        restore_rng(checkpoint["rng"])
        steps,completed,history=checkpoint["real_steps"],checkpoint["completed_episodes"],checkpoint["history"]
        if episodes<=completed:
            raise ValueError("target episodes must exceed completed episodes")
    atomic_json(output/"metadata.json",metadata)
    for episode in range(completed,episodes):
        started=time.perf_counter()
        start=windows.randint(first,train_end-settings.horizon+1)
        env=RetailEnv(data,retail_config,start,start+settings.horizon-1)
        obs=env.reset()
        base_total,shaped_total,losses=0.0,0.0,[]
        model_losses,synthetic_losses=[],[]
        while obs is not None:
            snapshot=env.snapshot() if model else None
            features=encoder.encode(obs)
            if settings.algorithm=="ddqn":
                epsilon=1-(1-settings.epsilon_final)*min(1,steps/settings.epsilon_decay_steps)
                actions=agent.act(features,epsilon)
            else:
                actions,logp,values=agent.act(features)
            nxt,base,terminal,_,info=env.step(actions)
            rewards,bonus=shaped_rewards(base,info["local_rewards"],obs,nxt,terminal,
                                        settings.gamma,settings.shaping,settings.shaping_weight,
                                        settings.reward_mode,settings.reward_scale)
            next_features=encoder.encode(nxt)
            steps+=1
            base_total+=base
            shaped_total+=base+bonus
            if settings.algorithm=="ddqn":
                agent.remember(features,actions,rewards,next_features,terminal)
                loss=agent.learn() if steps>=settings.warmup_steps else None
            else:
                agent.remember(features,actions,rewards,next_features,terminal,logp,values)
                loss=agent.learn() if terminal or len(agent.buffer)>=settings.rollout_steps else None
            if loss is not None:
                losses.append(loss)
            if model:
                model.observe_real(snapshot,obs,actions,info["demand"])
                model_loss=model.learn()
                if model_loss is not None:
                    model_losses.append(model_loss)
                    if steps>=settings.warmup_steps:
                        synthetic_losses.extend(model.plan(agent,data,retail_config))
            obs=nxt
        record={"episode":episode+1,"start_day":start,"real_steps":steps,
                "base_return":base_total,"shaped_regional_return":shaped_total,
                "training_seconds":time.perf_counter()-started,"gradient_updates":agent.updates,
                "synthetic_steps":model.synthetic_steps if model else 0,
                "losses":{key:sum(row[key] for row in losses if key in row)/sum(key in row for row in losses)
                          for key in {k for row in losses for k in row} if key!="updates"},
                "model_loss":sum(model_losses)/len(model_losses) if model_losses else None,
                "synthetic_td_loss":sum(synthetic_losses)/len(synthetic_losses) if synthetic_losses else None}
        if (episode+1)%eval_every==0 or episode+1==episodes:
            record["validation"]=evaluate_policy(agent,encoder,data,retail_config,validation,settings.algorithm)
        history.append(record)
        payload={"schema_version":1,"metadata":metadata,"completed_episodes":episode+1,
                 "real_steps":steps,"agent":agent.state_dict(),"model":model.state_dict() if model else None,
                 "window_rng":windows.getstate(),"rng":rng_state(),"history":history}
        # Checkpoint is authoritative. Derived JSON can be rebuilt after interruption.
        save_checkpoint(output/"checkpoint.pt",payload)
        atomic_json(output/"training.json",history)
    report={"schema_version":1,**metadata,"completed_episodes":episodes,"real_steps":steps,
            "synthetic_steps":model.synthetic_steps if model else 0,"test_evaluated":False,
            "validation":history[-1]["validation"],
            "training_seconds":sum(r["training_seconds"] for r in history)}
    atomic_json(output/"manifest.json",report)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo",action="store_true")
    source.add_argument("--data-dir",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--item-id")
    parser.add_argument("--settings",type=Path,help="JSON matching DeepConfig; unspecified fields use documented defaults")
    parser.add_argument("--algorithm",choices=("ddqn","ppo"),default=None)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--eval-every",type=int,default=10)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args(argv)
    settings=DeepConfig(**json.loads(args.settings.read_text(encoding="utf-8"))) if args.settings else DeepConfig()
    if args.algorithm is not None:
        settings=replace(settings,algorithm=args.algorithm)
    if args.seed is not None:
        settings=replace(settings,seed=args.seed)
    if args.demo:
        if args.config:
            parser.error("--demo uses fixed synthetic assumptions")
        data,config=demo_data(),demo_config()
    else:
        if args.config is None:
            parser.error("M5 requires explicit --config")
        config=RetailConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data=load_m5(args.data_dir,args.item_id)
    report=run(data,config,settings,args.output,args.episodes,args.eval_every,args.device,args.resume)
    print(json.dumps(report["validation"],indent=2))

if __name__=="__main__":
    main()
