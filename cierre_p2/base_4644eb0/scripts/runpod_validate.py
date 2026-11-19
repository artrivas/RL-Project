#!/usr/bin/env python3
"""Run the small P2 validation on an existing Runpod via API v2 (Ubuntu/WSL).

Reads id/api_key or RUNPOD_POD_ID/RUNPOD_API_KEY from a local .env.
Never creates/stops a Pod, installs packages, or starts the M5 campaign.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

COMMIT = "4f399fe5534fe24bdb49b404aa9fa74f570c9c51"
REPOSITORY = "https://github.com/artrivas/RL-Project.git"
REMOTE_ROOT = "/workspace/rl-project-validation"

JOB = r'''
import json, os, pathlib, subprocess, sys, time
root=pathlib.Path(sys.argv[1])
commit=sys.argv[2]
repo=root/"repo"
status={"status":"running","stage":"checkout","commit":commit}
def write_status():
    temporary=root/"status.tmp"
    temporary.write_text(json.dumps(status))
    temporary.replace(root/"status.json")
def run(command,timeout=180,cwd=None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command,check=True,timeout=timeout,cwd=cwd)
try:
    write_status()
    run(["git","init",str(repo)],30)
    run(["git","remote","add","origin","https://github.com/artrivas/RL-Project.git"],30,repo)
    run(["git","fetch","--depth","1","origin",commit],120,repo)
    run(["git","checkout","--detach","FETCH_HEAD"],30,repo)
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if actual!=commit:
        raise RuntimeError("checkout commit mismatch")
    status["stage"]="runtime"
    write_status()
    run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv"],30,repo)
    run([sys.executable,"-c","import torch,numpy; print('torch',torch.__version__,'cuda',torch.version.cuda,'numpy',numpy.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'"],60,repo)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    os.environ["MPLBACKEND"]="Agg"
    status["stage"]="preflight"
    write_status()
    run([sys.executable,"-m","src.retail.preflight","--require-cuda","--report",str(root/"preflight.json")],240,repo)
    status["stage"]="smoke"
    write_status()
    run([sys.executable,"-m","src.retail.smoke_deep","--device","cuda","--output",str(root/"smoke")],240,repo)
    status.update(status="passed",stage="complete")
except Exception as error:
    status.update(status="failed",error=type(error).__name__+": "+str(error))
    print(status["error"],flush=True)
finally:
    status["finished_unix"]=time.time()
    write_status()
'''

def credentials(path):
    values={}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line=line.strip()
        if not line or line.startswith("#"):
            continue
        name,sep,value=line.removeprefix("export ").partition("=")
        if sep:
            values[name.strip()]=value.strip().strip("\"'")
    pod_id=values.get("RUNPOD_POD_ID",values.get("id",""))
    key=values.get("RUNPOD_API_KEY",values.get("api_key",""))
    if not re.fullmatch(r"[A-Za-z0-9_-]+",pod_id) or not key:
        raise ValueError(".env must contain nonempty id and api_key")
    return pod_id,key

class APIError(RuntimeError):
    def __init__(self,status,message):
        self.status=status
        super().__init__(message)


def error_detail(error,key):
    # Only show documented-style error fields, never headers or request payloads.
    try:
        body=json.loads(error.read(8192))
    except (ValueError,OSError):
        return ""
    if not isinstance(body,dict):
        return ""
    fields=[]
    for name in ("title","detail","message","error","code"):
        value=body.get(name)
        if isinstance(value,(str,int)):
            value=str(value).replace(key,"[redacted]")
            value=re.sub(r"(?i)Bearer\s+\S+","Bearer [redacted]",value)
            fields.append(name+"="+value[:1000])
    return "; ".join(fields)


class Client:
    def __init__(self,pod_id,key):
        self.pod_id,self.key=pod_id,key

    def request(self,suffix="",payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        request=Request("https://api.runpod.io/v2/pods/"+quote(self.pod_id,safe="")+suffix,
                        data=data,headers={"Authorization":"Bearer "+self.key,
                                           "Content-Type":"application/json"})
        try:
            with urlopen(request,timeout=40) as response:
                return json.load(response)
        except HTTPError as error:
            hints={401:"authentication rejected",403:"operation forbidden",
                   404:"Pod not found for this key",409:"Pod not running",
                   503:"Pod host may not support API exec yet"}
            detail=error_detail(error,self.key)
            message=f"HTTP {error.code}: {hints.get(error.code,error.reason)}"
            if detail:
                message+="; server: "+detail
            raise APIError(error.code,message) from None
        except URLError as error:
            raise RuntimeError("Network error: "+str(error.reason).replace(self.key,"[redacted]")) from None

    def execute(self,code):
        if len(code)>4096:
            raise ValueError("command argument exceeds Runpod exec limit")
        result=self.request("/exec",{"command":["python3","-c",code],"timeoutSeconds":10})
        if result.get("timedOut"):
            raise RuntimeError("API exec wait expired; command may still run. Do not relaunch automatically.")
        if result.get("exitCode")!=0:
            raise RuntimeError("Remote command failed: "+str(result.get("stderr",""))[:1500])
        return json.loads(result["stdout"])

def launcher():
    return ("import json,pathlib,subprocess,sys,tempfile\n"
            f"base=pathlib.Path({REMOTE_ROOT!r}); base.mkdir(parents=True,exist_ok=True)\n"
            "root=pathlib.Path(tempfile.mkdtemp(prefix='p2-',dir=base))\n"
            "log=(root/'validation.log').open('ab',buffering=0)\n"
            f"process=subprocess.Popen([sys.executable,'-u','-c',{JOB!r},str(root),{COMMIT!r}],"
            "stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
            "print(json.dumps({'directory':str(root),'pid':process.pid}))\n")

def poll_code(directory):
    path=PurePosixPath(directory)
    if path.parent!=PurePosixPath(REMOTE_ROOT) or not path.name.startswith("p2-"):
        raise ValueError("unexpected remote validation directory")
    return ("import json,pathlib\n"
            f"root=pathlib.Path({directory!r})\n"
            "status=root/'status.json'\n"
            "result=json.loads(status.read_text()) if status.exists() else {'status':'starting'}\n"
            "if result.get('status') in ('passed','failed'):\n"
            "    result['log']=(root/'validation.log').read_text(errors='replace')[-60000:]\n"
            "    preflight=root/'preflight.json'\n"
            "    result['preflight']=json.loads(preflight.read_text()) if preflight.exists() else None\n"
            "print(json.dumps(result))\n")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file",type=Path,default=Path(".env"))
    parser.add_argument("--state-file",type=Path,default=Path("notebooks/artifacts/p2_via_b/setup/runpod-validation.json"))
    parser.add_argument("--resume",action="store_true",help="poll an existing job, never launch another")
    parser.add_argument("--check-exec",action="store_true",help="check exec access with a harmless Python command; do not launch validation")
    args=parser.parse_args(argv)
    if args.check_exec and args.resume:
        parser.error("--check-exec and --resume cannot be combined")
    pod_id,key=credentials(args.env_file)
    client=Client(pod_id,key)
    pod=client.request()
    print(json.dumps({k:pod.get(k) for k in ("id","name","status","image","gpu")},indent=2))
    if pod.get("status")!="RUNNING":
        raise RuntimeError("Pod must already be RUNNING; this script will not start it")
    if args.check_exec:
        result=client.execute("import json,sys; print(json.dumps({'exec_access': True, 'python': sys.version.split()[0]}))")
        print(json.dumps(result,indent=2))
        return 0
    if args.resume:
        state=json.loads(args.state_file.read_text())
        if state["pod_id"]!=pod_id:
            raise ValueError("saved job belongs to another Pod")
    else:
        if args.state_file.exists():
            raise ValueError("saved job exists: use --resume, or choose a new --state-file")
        # Reserve the local state path before launching to avoid concurrent duplicates.
        args.state_file.parent.mkdir(parents=True,exist_ok=True)
        with args.state_file.open("x",encoding="utf-8") as handle:
            json.dump({"pod_id":pod_id,"status":"launch_pending"},handle)
        try:
            state=client.execute(launcher())
        except APIError as error:
            if error.status in (401,403):
                args.state_file.write_text(json.dumps({"pod_id":pod_id,"status":"launch_rejected",
                    "http_status":error.status},indent=2)+"\n",encoding="utf-8")
            raise
        state.update(pod_id=pod_id,commit=COMMIT)
        args.state_file.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    if state.get("status")=="launch_rejected":
        raise RuntimeError("launch was rejected: correct authorization, then use a new --state-file")
    if "directory" not in state:
        raise RuntimeError("launch outcome was not recorded; inspect the Pod before starting a new job")
    print("Remote validation directory:",state["directory"],flush=True)
    previous=None
    for _ in range(120):
        result=client.execute(poll_code(state["directory"]))
        stage=(result.get("status"),result.get("stage"))
        if stage!=previous:
            print("Status:",stage,flush=True)
            previous=stage
        if result.get("status") in ("passed","failed"):
            args.state_file.with_suffix(".result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
            print(result.pop("log",""))
            print(json.dumps(result,indent=2))
            return 0 if result["status"]=="passed" else 1
        time.sleep(5)
    print("Still running. Poll again with --resume; the remote job is not stopped.")
    return 2

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError,ValueError,OSError,KeyError) as error:
        raise SystemExit(str(error))
