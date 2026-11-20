#!/usr/bin/env python3
"""Prepare a bounded M5 development pilot; --execute runs it on the Pod."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retail.config import RetailConfig
from src.retail.data import load_m5, SPLITS
from src.retail.deep_config import DeepConfig

def assumptions(data):
    end = SPLITS["train"][1]
    means = [statistics.mean(series[:end]) for series in data.sales]
    price = statistics.median(p for series in data.prices for p in series[:end])
    return RetailConfig(
        store_capacity=math.ceil(7 * max(means)),
        regional_capacity=math.ceil(7 * sum(means)),
        regional_supply=math.ceil(sum(means)),
        initial_store_stock=math.ceil(3 * max(means)),
        initial_regional_stock=math.ceil(3 * sum(means)),
        holding_cost=0.01 * price,
        shortage_cost=price,
        elasticities=(1.5, 1.5, 1.5),
    )

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text())
    manifest = audit["manifest"]
    if manifest["item_id"] != "FOODS_3_090" or manifest["splits"] != {k:list(v) for k,v in SPLITS.items()}:
        parser.error("pilot requires the audited FOODS_3_090 item and unchanged temporal splits")
    print("Loading and verifying M5 files...", flush=True)
    data = load_m5(args.data_dir, manifest["item_id"])
    if data.manifest["sha256"] != manifest["sha256"]:
        parser.error("data differs from the supplied audit")
    config = assumptions(data)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "assumptions.json", config.as_dict())
    write_json(output / "source-audit.json", audit)
    base = DeepConfig(seed=3, horizon=365, epsilon_decay_steps=2920)
    source = ["--data-dir", str(args.data_dir.resolve()), "--item-id", data.item_id,
              "--config", str(output / "assumptions.json")]
    jobs = [
        {"id":"H0", "module":"src.retail.rollout",
         "args":[*source, "--policy","h0","--split","validation","--output",str(output/"H0")]},
        {"id":"T0", "module":"src.retail.train_tabular",
         "args":[*source,"--seed","3","--episodes","8","--horizon","365","--eval-every","4",
                 "--gamma",str(base.gamma),"--reward-scale",str(base.reward_scale),
                 "--output",str(output/"T0")]},
    ]
    for name, changes in (("V0",{}), ("P0",{"algorithm":"ppo"}),
                          ("V3",{"shaping":"pbrs","dyna_steps":1})):
        settings = replace(base, **changes)
        settings_path = output / (name + "-settings.json")
        write_json(settings_path, settings.as_dict())
        jobs.append({"id":name,"module":"src.retail.train_deep",
                     "args":[*source,"--settings",str(settings_path),"--episodes","8",
                             "--eval-every","4","--device","cuda","--output",str(output/name)]})
    protocol = {
        "purpose":"development runtime and stability pilot, not a ranking",
        "seed":3,"episodes":8,"horizon":365,"real_steps_per_learner":2920,
        "test_evaluated":False,"jobs":jobs,
        "assumptions_basis":{
            "capacities":"7 days of maximum store mean / summed means, train only",
            "initial_stock":"3 days of maximum store mean / summed means, train only",
            "regional_supply":"ceiling of summed mean train sales",
            "costs":"holding=1% of median train price per unit-day; shortage=median train price per unit",
            "elasticities":"assumed 1.5 in every store; not estimated from M5",
            "limitations":[
                "equal absolute local capacities and starting stocks imply unequal coverage",
                "historical sales are a demand proxy; zero sales do not identify stockouts",
                "costs and elasticities require sensitivity analysis before final conclusions",
                "T0 exploration decays by episode; DDQN exploration decays by real step"
            ]
        }
    }
    write_json(output/"pilot.json", protocol)
    print("Prepared pilot:", output, flush=True)
    print(json.dumps(config.as_dict(), indent=2), flush=True)
    if not args.execute:
        return 0
    summaries = []
    for job in jobs:
        print("Starting", job["id"], flush=True)
        start = time.perf_counter()
        log_path = output / (job["id"] + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable,"-u","-m",job["module"],*job["args"]],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=1800)
        if result.returncode:
            print(log_path.read_text(errors="replace")[-6000:], flush=True)
            raise RuntimeError("Pilot failed at " + job["id"] + "; see " + str(log_path))
        report = json.loads((output/job["id"]/"manifest.json").read_text())
        summary = {"id":job["id"],"wall_seconds":time.perf_counter()-start,
                   "metrics":report.get("validation",report.get("metrics")),
                   "real_steps":report.get("real_steps",0),
                   "synthetic_steps":report.get("synthetic_steps",0)}
        summaries.append(summary)
        write_json(output/"summary.json", summaries)
        print(json.dumps(summary, indent=2), flush=True)
    print("PILOT COMPLETE:", output, flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
