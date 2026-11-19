"""T0 training entry point for a remote checkout; test split is never evaluated."""
import argparse
import json
from pathlib import Path
import random
import math
import time
from .config import RetailConfig
from .data import load_m5, fingerprint
from .env import RetailEnv
from .rollout import demo_data, demo_config
from .tabular import CooperativeQLearning
from .metrics import summarize

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo",action="store_true")
    source.add_argument("--data-dir",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--item-id")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--horizon",type=int,default=28)
    parser.add_argument("--alpha",type=float,default=0.1)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--reward-scale",type=float,default=100.0)
    parser.add_argument("--eval-every",type=int,default=10)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.episodes<1 or not 1<=args.horizon<=1183 or not 0<args.alpha<=1 or not 0<=args.gamma<=1:
        parser.error("invalid episodes, horizon, alpha or gamma")
    if not math.isfinite(args.reward_scale) or args.reward_scale<=0 or args.eval_every<=0:
        parser.error("invalid reward scale or validation interval")
    if args.demo:
        if args.config:
            parser.error("--demo uses fixed synthetic assumptions")
        data,config,last_train=demo_data(),demo_config(),60
        validation=(61,90)
    else:
        if args.config is None:
            parser.error("M5 requires explicit --config")
        config=RetailConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data=load_m5(args.data_dir,args.item_id)
        last_train,validation=1211,(1212,1576)
    first_day=config.history_days+1
    if first_day+args.horizon-1>last_train:
        parser.error("history plus horizon exceeds train interval")
    args.output.mkdir(parents=True,exist_ok=False)
    agent=CooperativeQLearning(config,args.seed,args.alpha,args.gamma)
    rng=random.Random(args.seed+300)
    steps=0
    history=[]
    with (args.output/"training.jsonl").open("w",encoding="utf-8") as log:
        for episode in range(args.episodes):
            started=time.perf_counter()
            start=rng.randint(first_day,last_train-args.horizon+1)
            env=RetailEnv(data,config,start,start+args.horizon-1)
            obs=env.reset()
            epsilon=1.0-0.95*episode/max(1,args.episodes-1)
            total=0.0
            base_G0=0.0
            discount=1.0
            td_errors=[]
            while obs is not None:
                actions=agent.act(obs,epsilon)
                nxt,reward,terminal,_,_=env.step(actions)
                errors=agent.observe(obs,actions,reward/args.reward_scale,nxt,terminal)
                if errors:
                    td_errors.extend(e for e in errors if e is not None)
                total+=reward
                base_G0+=discount*reward
                discount*=args.gamma
                steps+=1
                obs=nxt
            td_loss=sum(e*e for e in td_errors)/len(td_errors) if td_errors else None
            record={"episode":episode+1,"start_day":start,"base_return":total,"base_G0":base_G0,"td_loss":td_loss,
                    "real_steps":steps,"epsilon":epsilon,"synthetic_steps":0,
                    "training_seconds":time.perf_counter()-started}
            if (episode+1)%args.eval_every==0 or episode+1==args.episodes:
                validation_env=RetailEnv(data,config,*validation)
                validation_obs,validation_rows=validation_env.reset(),[]
                while validation_obs is not None:
                    validation_obs,_,_,_,info=validation_env.step(agent.act(validation_obs))
                    validation_rows.append(info)
                record["validation"]=summarize(validation_rows)
            history.append(record)
            log.write(json.dumps(record)+"\n")
            log.flush()
    (args.output/"training.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
    agent.save(args.output/"q_tables.npz")
    env=RetailEnv(data,config,*validation)
    obs,rows=env.reset(),[]
    while obs is not None:
        obs,_,_,_,info=env.step(agent.act(obs))
        rows.append(info)
    report={"schema_version":1,"algorithm":agent.name,"seed":args.seed,
            "episodes":args.episodes,"horizon":args.horizon,"alpha":args.alpha,"gamma":args.gamma,
            "real_steps":steps,"reward_scale":args.reward_scale,"config":config.as_dict(),"data":data.manifest,
            "validation":summarize(rows),"validation_range":validation,"test_evaluated":False,
            "training_reward":"regional_unshaped_profit_proxy",
            "epsilon_schedule":"linear_1.0_to_0.05_over_episodes",
            "finite_horizon_terminal":True,
            "code_sha256":{p.relative_to(Path(__file__).parents[1]).as_posix():fingerprint(p)
                           for p in [*sorted(Path(__file__).parent.glob("*.py")),
                                     Path(__file__).parents[1]/"agents.py"]}}
    (args.output/"manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report["validation"],indent=2))

if __name__=="__main__":
    main()
