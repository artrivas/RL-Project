"""Aggregate validation runs. This is descriptive analysis, not a significance test."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

METRICS=("base_return","stockout_rate","fill_rate","lost_units")

def summary(values):
    valid=[v for v in values if v is not None]
    return {"n":len(valid),"mean":statistics.mean(valid) if valid else None,
            "std":statistics.stdev(valid) if len(valid)>1 else None}

def time_to_threshold(history,baseline_return,reference_return):
    """80% of improvement over a fixed baseline, valid even for negative returns."""
    if not math.isfinite(reference_return) or not math.isfinite(baseline_return) or reference_return<=baseline_return:
        raise ValueError("reference must improve on the baseline")
    threshold=baseline_return+0.8*(reference_return-baseline_return)
    for row in history:
        if row.get("validation",{}).get("base_return",float("-inf"))>=threshold:
            return row["real_steps"]
    return None  # right-censored, not zero and not silently discarded

def aggregate(root,baseline_return=None,reference_return=None):
    root=Path(root)
    plan=json.loads((root/"suite.json").read_text(encoding="utf-8"))
    groups,curves,missing={},[],[]
    expected_identity=None
    for job in plan["jobs"]:
        directory=root/(job["id"] if job["seed"] is None else f'{job["id"]}_seed{job["seed"]}')
        path=directory/"manifest.json"
        if not path.is_file():
            missing.append(directory.name)
            continue
        manifest=json.loads(path.read_text(encoding="utf-8"))
        identity={"data":manifest["data"].get("sha256",manifest["data"].get("source")),
                  "item_id":manifest["data"].get("item_id"),
                  "config":manifest.get("retail_config",manifest.get("config")),
                  "range":manifest.get("validation_range",manifest.get("range"))}
        identity=json.dumps(identity,sort_keys=True)
        if expected_identity is None:
            expected_identity=identity
        elif identity!=expected_identity:
            raise ValueError("cannot compare different data, SKU, environment assumptions or evaluation ranges")
        if job["seed"] is not None:
            seed=manifest.get("seed",manifest.get("settings",{}).get("seed"))
            if seed!=job["seed"]:
                raise ValueError("run seed disagrees with suite plan")
            if manifest["real_steps"]!=plan["real_steps_per_trained_run"]:
                raise ValueError("training budgets differ from the suite plan")
        metrics=manifest.get("validation",manifest.get("metrics"))
        if metrics is None:
            raise ValueError(f"missing metrics in {path}")
        if manifest.get("test_evaluated",False) or manifest.get("split")=="test":
            raise ValueError("validation report cannot include test results")
        groups.setdefault(job["id"],[]).append((job["seed"],metrics))
        training=directory/"training.json"
        if training.exists():
            history=json.loads(training.read_text(encoding="utf-8"))
            for row in history:
                if "validation" in row:
                    curves.append({"experiment":job["id"],"seed":job["seed"],"real_steps":row["real_steps"],
                                   "base_return":row["validation"]["base_return"],
                                   "stockout_rate":row["validation"]["stockout_rate"],
                                   "synthetic_steps":row["synthetic_steps"]})
    report={"complete":not missing,"missing_runs":missing,"groups":{},"comparisons":{},
            "dispersion":"sample_std_ddof_1","test_results":False}
    for name,rows in groups.items():
        report["groups"][name]={metric:summary([r[metric] for _,r in rows]) for metric in METRICS}
        if name=="H0":
            report["groups"][name]["note"]="deterministic baseline; no seed dispersion"
    if "H0" in groups:
        baseline=groups["H0"][0][1]
        selfish=dict(groups.get("S0",[]))
        for name,rows in groups.items():
            if name=="H0":
                continue
            per_seed=[]
            for seed,metrics in rows:
                h=baseline["stockout_rate"]
                s=selfish.get(seed,{}).get("stockout_rate")
                c=metrics["stockout_rate"]
                per_seed.append({"seed":seed,"profit_gain_vs_H0":metrics["base_return"]-baseline["base_return"],
                                 "stockout_reduction_vs_H0":1-c/h if c is not None and h else None,
                                 "stockout_reduction_vs_S0":1-c/s if c is not None and s else None})
            report["comparisons"][name]=per_seed
    if (baseline_return is None)!=(reference_return is None):
        raise ValueError("threshold requires both baseline and fixed reference")
    if reference_return is not None:
        threshold_rows=[]
        for job in plan["jobs"]:
            if job["seed"] is None:
                continue
            path=root/f'{job["id"]}_seed{job["seed"]}'/"training.json"
            if path.exists():
                t=time_to_threshold(json.loads(path.read_text()),baseline_return,reference_return)
                threshold_rows.append({"experiment":job["id"],"seed":job["seed"],"real_steps_to_80pct":t,
                                       "not_reached":t is None})
        report["sample_efficiency"]={"baseline":baseline_return,"reference":reference_return,
                                     "runs":threshold_rows}
    return report,curves

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline-return",type=float)
    parser.add_argument("--reference-return",type=float)
    parser.add_argument("--plot",action="store_true")
    args=parser.parse_args(argv)
    report,curves=aggregate(args.suite,args.baseline_return,args.reference_return)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (args.output/"validation_curves.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("experiment","seed","real_steps","base_return","stockout_rate","synthetic_steps"))
        writer.writeheader()
        writer.writerows(curves)
    loss_rows=[]
    for path in sorted(args.suite.glob("*/training.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            values=dict(row.get("losses",{}))
            values.update({k:row[k] for k in ("model_loss","synthetic_td_loss") if row.get(k) is not None})
            for metric,value in values.items():
                loss_rows.append({"run":path.parent.name,"real_steps":row["real_steps"],"metric":metric,"value":value})
    with (args.output/"loss_curves.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("run","real_steps","metric","value"))
        writer.writeheader()
        writer.writerows(loss_rows)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for experiment in sorted({r["experiment"] for r in curves}):
            points={}
            for row in curves:
                if row["experiment"]==experiment:
                    points.setdefault(row["real_steps"],[]).append(row["base_return"])
            x=sorted(points)
            means=[statistics.mean(points[t]) for t in x]
            std=[statistics.stdev(points[t]) if len(points[t])>1 else 0 for t in x]
            ax.plot(x,means,label=experiment)
            ax.fill_between(x,[m-s for m,s in zip(means,std)],[m+s for m,s in zip(means,std)],alpha=.15)
        ax.set(xlabel="Real regional simulator steps",ylabel="Validation base return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output/"validation_returns.png",dpi=180)
        plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
