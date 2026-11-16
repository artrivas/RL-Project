"""Prepare or execute the predeclared comparison matrix on a remote checkout."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from .deep_config import DeepConfig

EXPERIMENTS = {
    "V0": {"algorithm":"ddqn"},
    "P0": {"algorithm":"ppo"},
    "S0": {"algorithm":"ddqn","reward_mode":"local"},
    "V1": {"algorithm":"ddqn","shaping":"pbrs"},
    "V2": {"algorithm":"ddqn","dyna_steps":1},
    "V3": {"algorithm":"ddqn","shaping":"pbrs","dyna_steps":1},
    "N0": {"algorithm":"ddqn","shaping":"naive"},
}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo",action="store_true")
    source.add_argument("--data-dir",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--item-id")
    parser.add_argument("--settings",type=Path)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true",help="run jobs sequentially; omission writes a plan only")
    args=parser.parse_args(argv)
    if len(args.seeds)<3 or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        parser.error("provide at least three distinct nonnegative seeds")
    if args.episodes<1:
        parser.error("episodes must be positive")
    if args.demo:
        if args.config or args.item_id:
            parser.error("demo has fixed assumptions and item")
        source_args=["--demo"]
    else:
        if args.config is None or args.item_id is None:
            parser.error("M5 suite requires a fixed --config and --item-id")
        source_args=["--data-dir",str(args.data_dir.resolve()),"--config",str(args.config.resolve()),"--item-id",args.item_id]
    base=DeepConfig(**json.loads(args.settings.read_text(encoding="utf-8"))) if args.settings else DeepConfig()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    settings_dir=root/"settings"
    settings_dir.mkdir()
    jobs=[]
    jobs.append({"id":"H0","seed":None,"command":[sys.executable,"-m","src.retail.rollout",*source_args,
                "--policy","h0","--output",str(root/"H0")]})
    for seed in args.seeds:
        jobs.append({"id":"T0","seed":seed,"command":[sys.executable,"-m","src.retail.train_tabular",*source_args,
                     "--seed",str(seed),"--episodes",str(args.episodes),"--horizon",str(base.horizon),
                     "--gamma",str(base.gamma),"--reward-scale",str(base.reward_scale),"--output",str(root/f"T0_seed{seed}")]})
        for name,variant in EXPERIMENTS.items():
            settings=replace(base,seed=seed,algorithm="ddqn",shaping="none",reward_mode="regional",dyna_steps=0)
            settings=replace(settings,**variant)
            path=settings_dir/f"{name}_seed{seed}.json"
            path.write_text(json.dumps(settings.as_dict(),indent=2)+"\n",encoding="utf-8")
            jobs.append({"id":name,"seed":seed,"command":[sys.executable,"-m","src.retail.train_deep",*source_args,
                         "--settings",str(path),"--episodes",str(args.episodes),
                         "--device",args.device,"--output",str(root/f"{name}_seed{seed}")]})
    plan={"schema_version":1,"real_steps_per_trained_run":args.episodes*base.horizon,"jobs":jobs,
          "test_included":False,"source":"synthetic" if args.demo else "M5"}
    (root/"suite.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    print(f"Prepared {len(jobs)} jobs at {root}")
    if args.execute:
        for job in jobs:
            subprocess.run(job["command"],check=True,cwd=Path(__file__).resolve().parents[2])

if __name__=="__main__":
    main()
