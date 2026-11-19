"""Frozen T0 evaluation from a table checkpoint; no learning updates."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import RetailConfig
from .data import load_m5,SPLITS,fingerprint
from .rollout import demo_data
from .tabular import CooperativeQLearning
from .env import RetailEnv
from .metrics import summarize

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--split",choices=("validation","test"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    report=json.loads((args.run/"manifest.json").read_text(encoding="utf-8"))
    for relative,digest in report["code_sha256"].items():
        if fingerprint(Path(__file__).parents[1]/relative)!=digest:
            raise ValueError("checkpoint code version mismatch")
    synthetic=report["data"].get("source")=="synthetic_test_fixture"
    if synthetic:
        if args.split=="test":
            parser.error("synthetic demo has no held-out test split")
        data,bounds=demo_data(),(61,90)
    else:
        if args.data_dir is None:
            parser.error("M5 evaluation requires --data-dir")
        data=load_m5(args.data_dir,report["data"]["item_id"])
        if data.manifest["sha256"]!=report["data"]["sha256"]:
            raise ValueError("dataset hash mismatch")
        bounds=SPLITS[args.split]
    config=RetailConfig(**report["config"])
    agent=CooperativeQLearning(config,report["seed"],report["alpha"],report["gamma"])
    with np.load(args.run/"q_tables.npz",allow_pickle=False) as tables:
        for i,learner in enumerate(agent.agents):
            q=tables[f"store_{i}"]
            if q.shape!=learner.Q.shape or not np.isfinite(q).all():
                raise ValueError("invalid Q table")
            learner.Q[:]=q
    env=RetailEnv(data,config,*bounds)
    obs,rows=env.reset(),[]
    while obs is not None:
        obs,_,_,_,info=env.step(agent.act(obs))
        rows.append(info)
    args.output.mkdir(parents=True,exist_ok=False)
    result={"split":args.split,"metrics":summarize(rows),"learning_updates":0,
            "data":data.manifest,"config":config.as_dict(),"seed":report["seed"],
            "algorithm":report["algorithm"],"code_sha256":report["code_sha256"]}
    (args.output/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["metrics"],indent=2))

if __name__=="__main__":
    main()
