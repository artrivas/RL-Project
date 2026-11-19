"""Tiny GPU/CPU integration jobs and checkpoint resume, never M5."""
import argparse
from dataclasses import replace
from pathlib import Path
import torch
from .deep_config import DeepConfig
from .rollout import demo_data,demo_config
from .train_deep import run
from .checkpoint import load_checkpoint

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    base=DeepConfig(seed=3,horizon=4,hidden=16,batch_size=2,warmup_steps=2,
                    model_warmup=2,ppo_epochs=1,rollout_steps=2)
    for name,changes in (("ddqn",{}),("ppo",{"algorithm":"ppo"}),
                         ("dyna",{"dyna_steps":1,"shaping":"pbrs"})):
        settings=replace(base,**changes)
        output=args.output/name
        run(demo_data(),demo_config(),settings,output,1,eval_every=1,device=args.device)
        result=run(demo_data(),demo_config(),settings,output,2,eval_every=1,
                   device=args.device,resume=output/"checkpoint.pt")
        if result["real_steps"]!=8 or result["test_evaluated"]:
            raise RuntimeError("invalid smoke outcome")
        checkpoint=load_checkpoint(output/"checkpoint.pt")
        if name=="dyna" and checkpoint["model"]["synthetic_steps"]==0:
            raise RuntimeError("Dyna did not execute synthetic transitions")
        print(name,result["validation"])
    print("Tiny integration smoke passed; this is not a performance benchmark.")

if __name__=="__main__":
    main()
