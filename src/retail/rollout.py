"""Reproducible H0/random rollouts; synthetic mode is explicitly not M5 evidence."""
import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from .config import RetailConfig
from .data import RetailData, SPLITS, load_m5, fingerprint
from .env import RetailEnv
from .baselines import HistoricalPriceBaseStock, RandomPolicy
from .metrics import summarize

def demo_data():
    n = 90
    calendar = []
    for day in range(n):
        dt = date(2020, 1, 1) + timedelta(days=day)
        calendar.append({"date": str(dt), "weekday": dt.weekday(), "month": dt.month,
                         "snap": 0, "event": 0})
    return RetailData("SYNTHETIC_ONLY", tuple(tuple(4+i+(d%7==0)*3 for d in range(n)) for i in range(3)),
                      tuple((2.0+i*0.5,)*n for i in range(3)), tuple(calendar),
                      {"source": "synthetic_test_fixture", "not_m5_results": True})

def demo_config():
    return RetailConfig(80, 150, 12, 20, 80, 0.02, 0.5, (1.0, 1.0, 1.0))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--item-id")
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--policy", choices=("h0", "random"), default="h0")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.demo and args.config:
        parser.error("--demo has fixed synthetic assumptions; omit --config")
    if not args.demo and args.config is None:
        parser.error("M5 requires explicit simulator assumptions in --config")
    if args.demo:
        if args.split == "test":
            parser.error("synthetic demo has no held-out test split")
        data, config = demo_data(), demo_config()
        bounds = (29, 60) if args.split == "train" else (61, 90)
    else:
        config = RetailConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data = load_m5(args.data_dir, args.item_id)
        bounds = SPLITS[args.split]
    env = RetailEnv(data, config, *bounds)
    policy = HistoricalPriceBaseStock(config) if args.policy == "h0" else RandomPolicy(config, args.seed)
    observations, rows = env.reset(), []
    while observations is not None:
        actions = policy.act(observations)
        observations, _, _, _, info = env.step(actions)
        info["actions"] = actions
        rows.append(info)
    # Fail rather than overwrite a previous run.
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).parent
    code_hashes = {p.name: fingerprint(p) for p in sorted(source_root.glob("*.py"))}
    report = {"schema_version": 1, "policy": policy.name, "seed": args.seed,
              "evaluation_only": True, "training_implemented": False,
              "split": "synthetic" if args.demo else args.split,
              "range": bounds, "config": config.as_dict(), "data": data.manifest,
              "code_sha256": code_hashes, "metrics": summarize(rows)}
    serialized = json.dumps(report, sort_keys=True)
    report["run_fingerprint"] = hashlib.sha256(serialized.encode()).hexdigest()
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (args.output/"transitions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row)+"\n")
    with (args.output/"daily_store.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = ("day", "store", "actions", "stock_before", "arrivals", "requests", "dispatched",
                   "demand", "sales", "lost", "stock_after", "transit", "local_rewards")
        writer.writerow(columns)
        for row in rows:
            for i, store in enumerate(("CA_1", "CA_2", "CA_3")):
                writer.writerow([row["day"], store] + [row[key][i] for key in columns[2:]])
    print(json.dumps(report["metrics"], indent=2))

if __name__ == "__main__":
    main()
