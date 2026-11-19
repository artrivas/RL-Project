"""Atomic episode-boundary checkpoints, including optimizer, replay and RNG."""
import os
from pathlib import Path
import random
import torch

def rng_state():
    return {"python":random.getstate(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

def save_checkpoint(path,payload):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,temporary)
    os.replace(temporary,path)

def load_checkpoint(path,device="cpu"):
    # Only tensor/primitive containers produced by this project are accepted.
    result=torch.load(Path(path),map_location=device,weights_only=True)
    if result.get("schema_version")!=1:
        raise ValueError("unsupported checkpoint schema")
    return result
