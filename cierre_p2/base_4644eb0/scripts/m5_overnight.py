#!/usr/bin/env python3
"""Prepare/run the overnight development matrix. Launch with nohup on the Pod."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retail.config import RetailConfig
from src.retail.data import FILES, fingerprint
from src.retail.deep_config import DeepConfig
from src.retail.suite import main as prepare_suite

def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def run_command(command, log_path, timeout):
    """Bound the process group; never leave a timed-out trainer in the background."""
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return {"exit_code":process.wait(timeout=timeout), "timed_out":False}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            return {"exit_code":process.returncode, "timed_out":True}

def execute_jobs(jobs, output, max_seconds, job_seconds=3600):
    deadline = time.monotonic() + max_seconds
    state = {"status":"running","jobs":[],"test_evaluated":False}
    save(output/"status.json", state)
    for index, job in enumerate(jobs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        name = job["id"] if job["seed"] is None else f'{job["id"]}_seed{job["seed"]}'
        state["active_job"] = name
        save(output/"status.json", state)
        print(f"[{index+1}/{len(jobs)}] Starting {name}", flush=True)
        started = time.monotonic()
        result = run_command(job["command"], output/"logs"/(name+".log"),
                             min(job_seconds, remaining))
        result.update(name=name, wall_seconds=time.monotonic()-started)
        state["jobs"].append(result)
        save(output/"status.json", state)
        print(json.dumps(result), flush=True)
    state.pop("active_job", None)
    state["pending_jobs"] = len(jobs)-len(state["jobs"])
    state["status"] = ("budget_exhausted" if state["pending_jobs"] or time.monotonic() >= deadline
                       else "completed_with_errors" if any(j["exit_code"] != 0 or j["timed_out"] for j in state["jobs"])
                       else "completed")
    save(output/"status.json", state)
    return state

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data/m5/raw"))
    parser.add_argument("--pilot", type=Path, default=Path("/workspace/rl-project-validation/m5-pilot-01"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--execute", action="store_true")
    args=parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 24:
        parser.error("max-hours must be finite, positive, and at most 24")
    if os.name != "posix":
        parser.error("run this orchestrator on Ubuntu/Linux")
    output=args.output.resolve()
    if output.exists():
        parser.error("output already exists; do not duplicate or overwrite an existing batch")
    audit=json.loads((args.pilot/"source-audit.json").read_text())
    if audit["manifest"]["item_id"] != "FOODS_3_090":
        parser.error("unexpected pilot SKU")
    print("Verifying data hashes...", flush=True)
    if {name:fingerprint(args.data_dir/name) for name in FILES} != audit["manifest"]["sha256"]:
        parser.error("M5 files changed since the pilot")
    config=RetailConfig(**json.loads((args.pilot/"assumptions.json").read_text()))
    # The new budget is a new development run, not a resume of the eight-episode pilot.
    settings=DeepConfig(horizon=365, epsilon_decay_steps=50000)
    output.mkdir(parents=True)
    (output/"logs").mkdir()
    save(output/"assumptions.json", config.as_dict())
    save(output/"settings.json", settings.as_dict())
    save(output/"source-audit.json", audit)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    save(output/"protocol.json", {
        "purpose":"overnight development; not confirmatory",
        "commit":commit,"seeds":[3,4,5],"episodes":300,"horizon":365,
        "real_steps_per_learner":109500,"total_real_training_steps":2628000,
        "test_evaluated":False,"max_hours":args.max_hours,"job_timeout_seconds":3600,
        "epsilon_schedule_ddqn":"1.0 to 0.05 over 50000 real steps, then fixed",
        "epsilon_schedule_tabular":"1.0 to 0.05 over 300 episodes",
        "assumptions":"unchanged from m5-pilot-01; no post-hoc weakening of H0",
        "checkpoint_policy":"deep: latest completed episode; tabular: final tables only",
        "pod_lifecycle":"not stopped or terminated by this script"
    })
    prepare_suite(["--data-dir",str(args.data_dir.resolve()),
                   "--config",str(output/"assumptions.json"),
                   "--item-id","FOODS_3_090","--settings",str(output/"settings.json"),
                   "--seeds","3","4","5","--episodes","300","--device","cuda",
                   "--output",str(output/"suite")])
    plan=json.loads((output/"suite"/"suite.json").read_text())
    if not args.execute:
        print("Plan prepared. No training started.", flush=True)
        return 0
    env_started=time.monotonic()
    result=run_command([sys.executable,"-m","src.retail.preflight","--require-cuda",
                        "--report",str(output/"preflight.json")],
                       output/"logs"/"preflight.log",180)
    if result["exit_code"] != 0:
        save(output/"status.json", {"status":"preflight_failed",**result})
        print("Preflight failed; see logs/preflight.log", flush=True)
        return 1
    state=execute_jobs(plan["jobs"], output,
                       max(0,args.max_hours*3600-(time.monotonic()-env_started)))
    print("Building validation summary...", flush=True)
    analysis=run_command([sys.executable,"-m","src.retail.analysis",
                          "--suite",str(output/"suite"),"--output",str(output/"analysis")],
                         output/"logs"/"analysis.log",180)
    state["analysis_exit_code"]=analysis["exit_code"]
    if analysis["exit_code"] == 0:
        report=json.loads((output/"analysis"/"summary.json").read_text())
        state["complete_results"]=report["complete"]
    save(output/"status.json", state)
    print("BATCH FINISHED:", json.dumps(state), flush=True)
    print("Results:", output, flush=True)
    return 0 if state["status"]=="completed" and analysis["exit_code"]==0 else 1

if __name__ == "__main__":
    sys.exit(main())
