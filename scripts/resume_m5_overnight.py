#!/usr/bin/env python3
"""Continue only unfinished jobs in the existing overnight suite."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m5_overnight import save, execute_jobs, run_command
from src.retail.data import FILES, fingerprint
from src.retail.analysis import aggregate

def option(command, name):
    return command[command.index(name)+1]

def job_name(job):
    return job["id"] if job["seed"] is None else f'{job["id"]}_seed{job["seed"]}'

def checkpoint_progress(path):
    from src.retail.checkpoint import load_checkpoint
    checkpoint=load_checkpoint(path,"cpu")
    return {"completed_episodes":checkpoint["completed_episodes"],
            "real_steps":checkpoint["real_steps"],
            "horizon":checkpoint["metadata"]["settings"]["horizon"]}

def recovery_plan(batch, plan, read_checkpoint=checkpoint_progress):
    pending, records, completed=[],[],[]
    for job in plan["jobs"]:
        name=job_name(job)
        directory=batch/"suite"/name
        command=list(job["command"])
        if Path(option(command,"--output")).resolve()!=directory.resolve():
            raise ValueError("suite output path mismatch: "+name)
        if (directory/"manifest.json").is_file():
            completed.append(name)
            continue
        if "src.retail.train_deep" not in command:
            raise ValueError("unfinished non-deep job requires separate review: "+name)
        if "--resume" in command:
            raise ValueError("original suite unexpectedly contains a resume command")
        record={"name":name,"action":"start","completed_episodes":0}
        if directory.exists():
            checkpoint=directory/"checkpoint.pt"
            if not checkpoint.is_file():
                raise ValueError("existing output lacks a checkpoint; refusing overwrite: "+name)
            progress=read_checkpoint(checkpoint)
            target=int(option(command,"--episodes"))
            done=progress["completed_episodes"]
            if not 0<done<target:
                raise ValueError("checkpoint needs review before resume: "+name)
            if progress["real_steps"]!=done*progress["horizon"]:
                raise ValueError("checkpoint step count is inconsistent: "+name)
            command.extend(["--resume",str(checkpoint)])
            record.update(progress,action="resume")
        command[0]=sys.executable
        pending.append({**job,"command":command})
        records.append(record)
    return pending,records,completed

def assert_idle(batch, proc=Path("/proc")):
    wanted={str(batch),str(batch/"suite")}
    for path in proc.glob("[0-9]*/cmdline"):
        if path.parent.name==str(os.getpid()):
            continue
        try:
            args=path.read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
        related=any(a in wanted or a.startswith(str(batch/"suite")+os.sep) for a in args)
        worker=any(a in ("src.retail.train_deep","src.retail.train_tabular",
                         "src.retail.rollout") or a.endswith("/m5_overnight.py")
                         for a in args)
        if related and worker:
            raise RuntimeError("existing batch worker is still active, PID "+path.parent.name)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-hours",type=float,default=4)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    if os.name!="posix":
        parser.error("run on Ubuntu/Linux")
    if not math.isfinite(args.max_hours) or not 0<args.max_hours<=24:
        parser.error("invalid max-hours")
    batch=args.batch.resolve();output=args.output.resolve()
    if output.exists():
        parser.error("recovery output already exists; inspect its status before retrying")
    # Cooperating recovery launches hold this lock throughout execution.
    import fcntl
    with (batch/".recovery.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_idle(batch)
        plan=json.loads((batch/"suite"/"suite.json").read_text())
        # Validate all finished runs before deciding to skip them.
        aggregate(batch/"suite")
        audit=json.loads((batch/"source-audit.json").read_text())
        data_dir=Path(option(plan["jobs"][0]["command"],"--data-dir"))
        print("Verifying unchanged M5 files...",flush=True)
        if {name:fingerprint(data_dir/name) for name in FILES}!=audit["manifest"]["sha256"]:
            raise ValueError("data changed since the overnight batch")
        code={p.name:fingerprint(p) for p in sorted((ROOT/"src"/"retail").glob("*.py"))}
        for job in plan["jobs"]:
            path=batch/"suite"/job_name(job)/"manifest.json"
            if path.exists() and "src.retail.train_deep" in job["command"]:
                if json.loads(path.read_text())["code_sha256"]!=code:
                    raise ValueError("training code changed; cannot combine these runs")
        pending,records,completed=recovery_plan(batch,plan)
        output.mkdir(parents=True)
        (output/"logs").mkdir()
        save(output/"recovery-plan.json",{
            "batch":str(batch),"completed_runs_skipped":completed,"pending":records,
            "max_hours":args.max_hours,"test_evaluated":False,
            "original_status":json.loads((batch/"status.json").read_text()),
            "suite_sha256":fingerprint(batch/"suite"/"suite.json")
        })
        print(json.dumps({"skipped_completed":len(completed),"pending":records},indent=2),flush=True)
        if not args.execute:
            print("Recovery plan only; no training launched.",flush=True)
            return 0
        state=execute_jobs(pending,output,args.max_hours*3600,job_seconds=5400)
        result=run_command([sys.executable,"-m","src.retail.analysis",
                            "--suite",str(batch/"suite"),"--output",str(output/"analysis")],
                           output/"logs"/"analysis.log",180)
        state["analysis_exit_code"]=result["exit_code"]
        if result["exit_code"]==0:
            report=json.loads((output/"analysis"/"summary.json").read_text())
            state["complete_results"]=report["complete"]
            state["missing_runs"]=report["missing_runs"]
        save(output/"status.json",state)
        print("RECOVERY FINISHED:",json.dumps(state),flush=True)
        print("Summary:",output/"analysis"/"summary.json",flush=True)
        return 0 if state.get("complete_results") and state["status"]=="completed" else 1

if __name__=="__main__":
    sys.exit(main())
