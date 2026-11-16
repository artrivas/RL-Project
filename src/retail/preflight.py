"""Run numerical acceptance tests before starting a remote training campaign."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import unittest

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda",action="store_true")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv)
    missing=[p for p in ("numpy","torch") if importlib.util.find_spec(p) is None]
    if missing:
        parser.exit(2,"Missing dependencies: "+", ".join(missing)+". Deep validation is NOT complete.\n")
    import torch
    device="cuda" if args.require_cuda else "cpu"
    if args.require_cuda and not torch.cuda.is_available():
        parser.exit(2,"CUDA is required but unavailable.\n")
    # Exercise actual allocation, autodiff, optimizer and checkpoint-related runtime.
    x=torch.ones(3,device=device,requires_grad=True)
    optimizer=torch.optim.Adam([x],lr=.01)
    optimizer.zero_grad()
    x.square().sum().backward()
    optimizer.step()
    if not torch.isfinite(x).all():
        raise RuntimeError("nonfinite device smoke test")
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/"tests"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    passed=result.wasSuccessful() and not result.skipped
    report={"passed":passed,"tests_run":result.testsRun,"skipped":len(result.skipped),
            "failures":len(result.failures),"errors":len(result.errors),"python":platform.python_version(),
            "torch":str(torch.__version__),"device":device,"cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if passed else 1

if __name__=="__main__":
    sys.exit(main())
