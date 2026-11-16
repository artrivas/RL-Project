"""Evaluate a frozen deep checkpoint, including an explicitly requested test split."""
import argparse
from pathlib import Path
import json
from .checkpoint import load_checkpoint
from .config import RetailConfig
from .deep_config import DeepConfig
from .data import load_m5, SPLITS
from .rollout import demo_data
from .features import FeatureEncoder
from .deep import DDQN,PPO
from .train_deep import evaluate_policy,atomic_json,canonical,code_fingerprint

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--split",choices=("validation","test"),required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    checkpoint=load_checkpoint(args.checkpoint,args.device)
    meta=checkpoint["metadata"]
    if meta["code_sha256"]!=code_fingerprint():
        raise ValueError("evaluation must use the checkpoint code version")
    synthetic=meta["data"].get("source")=="synthetic_test_fixture"
    if synthetic:
        if args.split=="test":
            parser.error("synthetic demo has no held-out test split")
        data,bounds=demo_data(),(61,90)
    else:
        if args.data_dir is None:
            parser.error("M5 checkpoint requires --data-dir")
        data=load_m5(args.data_dir,meta["data"]["item_id"])
        # Auto vs explicit selection does not change the content identity.
        if data.manifest["sha256"]!=meta["data"]["sha256"]:
            raise ValueError("dataset hash mismatch")
        bounds=SPLITS[args.split]
    config=RetailConfig(**meta["retail_config"])
    settings=DeepConfig(**meta["settings"])
    encoder=FeatureEncoder(config,meta["encoder"]["price_scale"])
    klass=DDQN if settings.algorithm=="ddqn" else PPO
    agent=klass(encoder.size,len(config.actions),settings,args.device)
    agent.load_state_dict(checkpoint["agent"])
    metrics=evaluate_policy(agent,encoder,data,config,bounds,settings.algorithm)
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/"evaluation.json",{"schema_version":1,"split":args.split,
                "metrics":metrics,"settings":settings.as_dict(),"data":data.manifest,
                "retail_config":config.as_dict(),"code_sha256":meta["code_sha256"],
                "completed_episodes":checkpoint["completed_episodes"],
                "real_steps":checkpoint["real_steps"],"learning_updates":0,
                "checkpoint":str(args.checkpoint.resolve())})
    print(json.dumps(metrics,indent=2))

if __name__=="__main__":
    main()
