#!/usr/bin/env python3
"""Close the completed M5 matrix with frozen CPU evaluation, reports and demo."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.retail.analysis import aggregate, summary
from src.retail.config import RetailConfig
from src.retail.data import load_m5, SPLITS, fingerprint
from src.retail.env import RetailEnv
from src.retail.metrics import summarize
from src.retail.shaping import potential, shaped_rewards

NAMES = ("H0", "T0", "V0", "P0", "S0", "V1", "V2", "V3", "N0")
SEEDS = (3, 4, 5)
STEPS = 109500

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def csv_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run_name(job):
    return job["id"] if job["seed"] is None else f'{job["id"]}_seed{job["seed"]}'

def validate_report_config(actual, expected, run):
    """Compare validated simulator values, not JSON spellings such as 1/1.0.

    No tolerance is applied: any genuinely different numeric value or action
    ordering remains an error. The frozen inputs are never rewritten.
    """
    def contains_bool(value):
        if isinstance(value, dict):
            return any(contains_bool(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_bool(v) for v in value)
        return isinstance(value, bool)

    if contains_bool(actual) or contains_bool(expected):
        raise ValueError(f"{run}: boolean values are not simulator parameters")
    try:
        observed = RetailConfig(**actual).as_dict()
        reference = RetailConfig(**expected).as_dict()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{run}: invalid simulator configuration: {error}") from error
    changed = {key:{"observed":observed[key], "expected":reference[key]}
               for key in reference if observed[key] != reference[key]}
    if changed:
        raise ValueError(f"{run}: simulator assumptions differ from the frozen report: "
                         + canonical(changed))

def audit(suite):
    plan = read(suite / "suite.json")
    expected = {"H0"} | {f"{name}_seed{seed}" for name in NAMES[1:] for seed in SEEDS}
    names = [run_name(job) for job in plan["jobs"]]
    if len(names) != 25 or set(names) != expected or plan["real_steps_per_trained_run"] != STEPS:
        raise ValueError("Expected the complete fixed matrix: H0 and eight methods x seeds 3,4,5.")
    report, curves = aggregate(suite)
    if not report["complete"]:
        raise ValueError("Incomplete validation matrix: " + str(report["missing_runs"]))
    records = []
    for job in plan["jobs"]:
        name = run_name(job)
        folder = suite / name
        manifest = read(folder / "manifest.json")
        config = manifest.get("retail_config", manifest.get("config"))
        validate_report_config(config, read(ROOT/"docker/m5_final_assumptions.json"), name)
        if manifest.get("validation_range",manifest.get("range")) != [1212,1576]:
            raise ValueError("Unexpected validation interval.")
        if manifest["data"].get("source") != "M5":
            raise ValueError("Final report requires audited M5, not fixtures.")
        for relative, digest in manifest["code_sha256"].items():
            # T0 records paths relative to src (including bare agents.py).
            # H0 and deep runners record basenames relative to src/retail.
            source_root = ROOT / ("src" if job["id"] == "T0" else "src/retail")
            code = source_root / relative
            if not code.is_file():
                raise ValueError(f"Missing fingerprinted source: {name}: {relative} -> {code}")
            if fingerprint(code) != digest:
                raise ValueError(f"Code fingerprint mismatch: {name}: {relative}")
        record = {"id":job["id"], "seed":job["seed"], "name":name,
                  "manifest":manifest, "manifest_sha256":fingerprint(folder / "manifest.json")}
        if name != "H0":
            history = read(folder / "training.json")
            if (len(history) != 300 or history[-1]["real_steps"] != STEPS or
                [r["episode"] for r in history] != list(range(1,301))):
                raise ValueError(f"Incomplete episode history: {name}")
            record["history_sha256"] = fingerprint(folder / "training.json")
            checkpoint = folder / ("q_tables.npz" if job["id"] == "T0" else "checkpoint.pt")
            record["checkpoint_sha256"] = fingerprint(checkpoint)
            if job["id"] != "T0":
                if canonical(manifest["settings"]) != canonical(read(suite / "settings" / (name + ".json"))):
                    raise ValueError(f"Settings differ from the prepared matrix: {name}")
                if manifest.get("completed_episodes") != 300:
                    raise ValueError(f"Incomplete checkpoint manifest: {name}")
        records.append(record)
    return records, report, curves

def verify_checkpoint(checkpoint, manifest):
    metadata = checkpoint["metadata"]
    for key in metadata:
        if canonical(metadata[key]) != canonical(manifest[key]):
            raise ValueError(f"Checkpoint metadata differs from final manifest: {key}")
    if checkpoint["completed_episodes"] != 300 or checkpoint["real_steps"] != STEPS:
        raise ValueError("Checkpoint is not the final 300-episode policy.")
    if canonical(checkpoint["history"][-1]["validation"]) != canonical(manifest["validation"]):
        raise ValueError("Checkpoint and final validation disagree.")

def build_policy(record, suite):
    import numpy as np
    from src.retail.baselines import HistoricalPriceBaseStock
    m = record["manifest"]
    config = RetailConfig(**m.get("retail_config", m.get("config")))
    if record["id"] == "H0":
        agent = HistoricalPriceBaseStock(config)
        return config, agent.act, None, .99
    if record["id"] == "T0":
        from src.retail.tabular import CooperativeQLearning
        agent = CooperativeQLearning(config, m["seed"], m["alpha"], m["gamma"])
        with np.load(suite / record["name"] / "q_tables.npz", allow_pickle=False) as tables:
            for i, learner in enumerate(agent.agents):
                q = tables[f"store_{i}"]
                if q.shape != learner.Q.shape or not np.isfinite(q).all():
                    raise ValueError("Invalid tabular checkpoint")
                learner.Q[:] = q
        return config, agent.act, None, m["gamma"]
    from src.retail.checkpoint import load_checkpoint
    from src.retail.deep_config import DeepConfig
    from src.retail.features import FeatureEncoder
    from src.retail.deep import DDQN, PPO
    from src.retail.world_model import DemandModel
    payload = load_checkpoint(suite / record["name"] / "checkpoint.pt", "cpu")
    verify_checkpoint(payload, m)
    settings = DeepConfig(**m["settings"])
    encoder = FeatureEncoder(config, m["encoder"]["price_scale"])
    agent = (PPO if settings.algorithm == "ppo" else DDQN)(encoder.size, len(config.actions), settings, "cpu")
    # Inference weights only: replay and optimizers are deliberately not restored.
    if settings.algorithm == "ppo":
        agent.actors.load_state_dict(payload["agent"]["actors"])
        agent.critics.load_state_dict(payload["agent"]["critics"])
        agent.actors.eval()
        agent.critics.eval()
        act = lambda obs: agent.act(encoder.encode(obs), deterministic=True)[0]
    else:
        agent.online.load_state_dict(payload["agent"]["online"])
        agent.online.eval()
        act = lambda obs: agent.act(encoder.encode(obs))
    model = None
    if settings.dyna_steps:
        model = DemandModel(encoder, len(config.actions), settings, "cpu")
        model.net.load_state_dict(payload["model"]["net"])
        model.net.eval()
    del payload
    gc.collect()
    return config, act, model, settings.gamma

def rollout(data, config, act, bounds, gamma, model=None):
    env = RetailEnv(data, config, *bounds)
    obs = env.reset()
    initial = potential(obs)
    rows = []
    discounted = bonus_total = 0.0
    errors, naive_errors = [], []
    while obs is not None:
        actions = tuple(int(x) for x in act(obs))
        prediction = model.predict(obs, actions) if model is not None else None
        nxt, reward, terminal, _, info = env.step(actions)
        t = len(rows)
        discounted += gamma**t * reward
        _, bonus = shaped_rewards(reward, info["local_rewards"], obs, nxt, terminal, gamma, "pbrs")
        bonus_total += gamma**t * bonus
        info.update(actions=actions, date=data.calendar[info["day"]-1]["date"],
                    price_multipliers=[config.actions[a][0] for a in actions],
                    prices=[obs[i]["base_price"]*config.actions[a][0] for i,a in enumerate(actions)],
                    forecast=[o["forecast"] for o in obs])
        if prediction is not None:
            baseline = [int(math.floor(obs[i]["forecast"]*config.actions[a][0]**(-config.elasticities[i])+.5))
                        for i,a in enumerate(actions)]
            errors.extend(p-d for p,d in zip(prediction,info["demand"]))
            naive_errors.extend(p-d for p,d in zip(baseline,info["demand"]))
            info.update(predicted_demand=prediction, forecast_demand=baseline)
        rows.append(info)
        obs = nxt
    if not math.isclose(bonus_total, -initial, abs_tol=1e-8):
        raise ValueError("PBRS telescoping check failed on evaluation trajectory.")
    metrics = summarize(rows)
    metrics.update(discounted_base_return=discounted,
                   revenue=sum(sum(p*q for p,q in zip(r["prices"],r["sales"])) for r in rows),
                   holding_cost=sum(sum(r["stock_after"])*config.holding_cost for r in rows),
                   shortage_cost=sum(sum(r["lost"])*config.shortage_cost for r in rows))
    diagnostic = None
    if errors:
        diagnostic = {"n_store_days":len(errors), "demand_mae":statistics.mean(map(abs, errors)),
                      "demand_rmse":math.sqrt(statistics.mean(e*e for e in errors)),
                      "forecast_mae":statistics.mean(map(abs, naive_errors)),
                      "forecast_rmse":math.sqrt(statistics.mean(e*e for e in naive_errors)),
                      "scope":"frozen one-step predictions on this policy trajectory; no fitting"}
    return {"metrics":metrics, "daily":rows, "model":diagnostic,
            "pbrs_telescoping_error":bonus_total+initial, "learning_updates":0}

def aggregate_test(results):
    grouped = {name:[r for r in results if r["id"] == name] for name in NAMES}
    groups = {name:{key:summary([r["metrics"][key] for r in runs])
                    for key in results[0]["metrics"]} for name,runs in grouped.items()}
    h0 = grouped["H0"][0]["metrics"]
    selfish = {r["seed"]:r["metrics"] for r in grouped["S0"]}
    comparisons = []
    for r in results:
        if r["id"] == "H0":
            continue
        m, s = r["metrics"], selfish[r["seed"]]
        h_reduction = 1-m["stockout_rate"]/h0["stockout_rate"] if h0["stockout_rate"] else None
        s_reduction = 1-m["stockout_rate"]/s["stockout_rate"] if s["stockout_rate"] else None
        comparisons.append({"run":r["name"], "experiment":r["id"], "seed":r["seed"],
                            "profit_gain_vs_H0":m["base_return"]-h0["base_return"],
                            "stockout_reduction_vs_H0":h_reduction,
                            "stockout_reduction_vs_S0":s_reduction,
                            "proxy_joint_target_met":(h_reduction is not None and s_reduction is not None and
                                                     h_reduction >= .30 and s_reduction >= .30 and
                                                     m["base_return"] > h0["base_return"])})
    pairs = []
    for a,b in (("V0","T0"),("P0","V0"),("V1","V0"),("V2","V0"),("V3","V1"),("V3","V2"),("V0","S0"),("N0","V1")):
        for seed in SEEDS:
            ma = next(r["metrics"] for r in grouped[a] if r["seed"]==seed)
            mb = next(r["metrics"] for r in grouped[b] if r["seed"]==seed)
            pairs.append({"contrast":a+" minus "+b, "seed":seed,
                          "return_difference":ma["base_return"]-mb["base_return"],
                          "stockout_difference":ma["stockout_rate"]-mb["stockout_rate"]})
    return {"complete":True, "test_results":True, "groups":groups, "comparisons":comparisons,
            "paired_ablations":pairs, "dispersion":"sample_std_ddof_1",
            "inference":"descriptive; three development seeds; no significance or convergence claim",
            "historical_Walmart_target_identifiable":False}

def learning_diagnostics(records, suite, curves, reference):
    threshold = .8 * reference
    rows = []
    for r in records:
        if r["id"] == "H0":
            continue
        history = read(suite / r["name"] / "training.json")
        validations = [h for h in history if "validation" in h]
        reached = next((h["real_steps"] for h in validations if h["validation"]["base_return"] >= threshold), None)
        rows.append({"run":r["name"], "experiment":r["id"], "seed":r["seed"],
                     "steps_to_80pct":reached, "right_censored":reached is None,
                     "budget":STEPS, "threshold":threshold,
                     "tail_30_train_return_std":statistics.stdev(h["base_return"] for h in history[-30:]),
                     "tail_10_validation_return_std":statistics.stdev(h["validation"]["base_return"] for h in validations[-10:]),
                     "synthetic_steps":history[-1]["synthetic_steps"],
                     "training_seconds":sum(h["training_seconds"] for h in history)})
    return rows

def hacking_example():
    def obs(stock):
        return tuple({"stock":stock,"transit":0,"forecast":1} for _ in range(3))
    initial = obs(0)
    rows = []
    for name, middle, reward in (("serve",obs(0),1.0),("hoard",obs(14),.9)):
        returns = {}
        for mode in ("none","naive","pbrs"):
            _, b0 = shaped_rewards(0,(0,0,0),initial,middle,False,.99,mode)
            _, b1 = shaped_rewards(reward,(reward,)*3,middle,None,True,.99,mode)
            returns[mode] = b0+.99*(reward+b1)
        rows.append({"action":name, **returns})
    assert rows[0]["none"] > rows[1]["none"]
    assert rows[0]["naive"] < rows[1]["naive"]
    assert rows[0]["pbrs"] > rows[1]["pbrs"]
    return {"scope":"constructed two-step counterexample, not an M5 outcome", "gamma":.99, "returns":rows}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=Path,default=Path("/workspace/rl-project-validation/m5-night-01"))
    parser.add_argument("--data-dir",type=Path,default=Path("/workspace/data/m5/raw"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out / ".finalize.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            finish(args, out)
        except Exception as error:
            save(out / "status.json", {"status":"failed","error":str(error),
                                      "action":"fix cause then rerun the same command; completed evaluations are preserved"})
            raise

def finish(args, out):
    import torch
    import numpy as np
    from m5_delivery import deliver
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    suite = args.batch.resolve() / "suite"
    print("Auditing 25 completed runs, source and checkpoint hashes...", flush=True)
    records, validation, curves = audit(suite)
    data_identity = records[0]["manifest"]["data"]
    data = load_m5(args.data_dir, data_identity["item_id"])
    if data.manifest["sha256"] != data_identity["sha256"]:
        raise ValueError("Data differ from the training inputs.")
    reference = max([validation["groups"]["H0"]["base_return"]["mean"]] + [r["base_return"] for r in curves])
    if reference <= 0:
        raise ValueError("Positive empirical validation reference required for the literal 80% threshold.")
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    protocol = {"schema_version":1, "source_commit":commit,
                "data":data.manifest, "seeds":list(SEEDS), "test_range":list(SPLITS["test"]),
                "checkpoint_rule":"last completed episode 300; no selection or training on test",
                "amendment":"close descriptively with development seeds 3,4,5; new confirmatory seeds 0,1,2 not run",
                "inference":"three optimization seeds on one shared temporal test year, not three independent datasets",
                "evaluation_device":"cpu", "torch_version":str(torch.__version__),
                "python_version":sys.version, "numpy_version":str(np.__version__),
                "threshold_reference":reference, "threshold_rule":"80% of empirical maximum validation return, including H0",
                "threshold_status":"post-hoc descriptive definition fixed before test; not an optimum estimate",
                "demo_rule":"P0_seed3 predetermined from validation family choice and smallest seed",
                "stability_rule":"last 30 training episodes and last 10 fixed-window validation checkpoints",
                "closure_sources_sha256":{p.name:fingerprint(p) for p in
                                        [Path(__file__), ROOT/"scripts/m5_delivery.py", ROOT/"informe/p2_ieee_template.tex"]},
                "runs":records}
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and canonical(read(protocol_path)) != canonical(protocol):
        raise ValueError("Existing finalization uses a different protocol, source or checkpoint. Refusing to mix test results.")
    if not protocol_path.exists() and any((out/"test").glob("*.json")):
        raise ValueError("Test results exist without their freeze protocol.")
    save(protocol_path, protocol)  # BEFORE any test rollout.
    freeze = fingerprint(protocol_path)
    # Validate every checkpoint before opening test, including final episode identity.
    from src.retail.checkpoint import load_checkpoint
    import numpy as np
    for r in records:
        if r["id"] in ("H0","T0"):
            if r["id"] == "T0":
                build_policy(r,suite)
            continue
        payload = load_checkpoint(suite/r["name"]/"checkpoint.pt","cpu")
        verify_checkpoint(payload,r["manifest"])
        if canonical(payload["history"]) != canonical(read(suite/r["name"]/"training.json")):
            raise ValueError("Training history differs from its authoritative checkpoint.")
        del payload
        gc.collect()
    save(out/"validation_summary.json",validation)
    csv_rows(out/"validation_curves.csv",curves)
    diagnostics = learning_diagnostics(records,suite,curves,reference)
    csv_rows(out/"learning_diagnostics.csv",diagnostics)
    save(out/"hacking_example.json",hacking_example())
    results = []
    for i,r in enumerate(records,1):
        path = out/"test"/(r["name"]+".json")
        save(out/"status.json",{"status":"evaluating","run":r["name"],"index":i,"total":25})
        if path.exists():
            result = read(path)
            if result.get("freeze_sha256") != freeze:
                raise ValueError("Saved evaluation belongs to another freeze protocol.")
            print(f"[{i}/25] Reusing {r['name']}",flush=True)
        else:
            print(f"[{i}/25] Frozen CPU evaluation: {r['name']}",flush=True)
            config, act, model, gamma = build_policy(r,suite)
            with torch.inference_mode():
                # Documents CPU/GPU greedy-policy numerical differences without selecting or changing a policy.
                validation_cpu = rollout(data,config,act,SPLITS["validation"],gamma)["metrics"]
                result = rollout(data,config,act,SPLITS["test"],gamma,model)
            original = r["manifest"].get("validation",r["manifest"].get("metrics"))
            result.update(id=r["id"],seed=r["seed"],name=r["name"],split="test",
                          freeze_sha256=freeze, validation_cpu=validation_cpu,
                          cpu_validation_return_delta=validation_cpu["base_return"]-original["base_return"],
                          checkpoint_sha256=r.get("checkpoint_sha256"), device="cpu")
            save(path,result)
            del act,model
            gc.collect()
        if (result.get("name") != r["name"] or result.get("id") != r["id"] or
            result.get("seed") != r["seed"] or result.get("split") != "test" or
            result.get("learning_updates") != 0 or result["metrics"]["days"] != 365):
            raise ValueError("Invalid saved evaluation identity or horizon.")
        results.append(result)
    # Confirm read-only evaluation did not change the artifacts.
    for r in records:
        folder = suite/r["name"]
        if fingerprint(folder/"manifest.json") != r["manifest_sha256"]:
            raise ValueError("Training artifacts changed during evaluation.")
        if r["id"] != "H0":
            checkpoint = folder/("q_tables.npz" if r["id"]=="T0" else "checkpoint.pt")
            if fingerprint(checkpoint) != r["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during evaluation.")
    test = aggregate_test(results)
    save(out/"test_summary.json",test)
    csv_rows(out/"paired_ablations.csv",test["paired_ablations"])
    csv_rows(out/"target_checks.csv",test["comparisons"])
    print("Generating tables, figures, LaTeX report, notebook and demo...",flush=True)
    deliver(out, suite, records, validation, test, curves, diagnostics, results, protocol)
    save(out/"status.json",{"status":"completed","evaluations":25,"test_results":True,
                           "new_training_steps":0, "pdf_generated":(out/"informe_p2.pdf").exists(),
                           "scientific_target":"see target_checks.csv; never inferred from process success",
                           "submission_ready":False, "pending":"human review of PDF, authors and oral presentation"})
    files = {p.relative_to(out).as_posix():fingerprint(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name not in (".finalize.lock","SHA256SUMS.json")}
    save(out/"SHA256SUMS.json",files)
    archive = out.parent/(out.name+".zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as package:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != ".finalize.lock":
                package.write(p,Path(out.name)/p.relative_to(out))
    temporary.replace(archive)
    print("CIERRE COMPLETO:",out,flush=True)
    print("PAQUETE:",archive,flush=True)
    print("No se ha entrenado ni modificado ninguna política. Revisar el informe antes de Canvas.",flush=True)

if __name__ == "__main__":
    main()
