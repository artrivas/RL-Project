"""Focused G0 smoke: synthetic demo verifies discounted BASE return and TD metrics."""
import json
import tempfile
from pathlib import Path
from src.retail.metrics import discounted_return
from src.retail.rollout import demo_data, demo_config

def check_unit():
    assert abs(discounted_return([1, 1, 1], 0.9) - 2.71) < 1e-6
    assert abs(discounted_return([1, 2, 3], 0.5) - 2.75) < 1e-6
    assert discounted_return([], 0.9) == 0.0
    print("unit G0 OK: [1,1,1]@0.9=2.71 [1,2,3]@0.5=2.75")

def check_tabular(tmp):
    from src.retail.train_tabular import main
    out = Path(tmp) / "tab"
    main(["--demo", "--seed", "3", "--episodes", "2", "--horizon", "2",
          "--gamma", "1.0", "--output", str(out)])
    rows = [json.loads(l) for l in (out / "training.jsonl").read_text().splitlines()]
    for r in rows:
        assert "base_G0" in r and "td_loss" in r, r.keys()
        assert abs(r["base_G0"] - r["base_return"]) < 1e-9, r
        assert r["td_loss"] is None or r["td_loss"] >= 0
    print(f"tabular OK: episodes=2 base_return={[r['base_return'] for r in rows]} "
          f"base_G0={[r['base_G0'] for r in rows]} td_loss={[r['td_loss'] for r in rows]}")
    return rows

def check_deep(tmp):
    import torch
    torch.set_num_threads(1)
    from src.retail.deep_config import DeepConfig
    from src.retail.train_deep import run
    data, config = demo_data(), demo_config()
    settings = DeepConfig(seed=3, horizon=2, hidden=8, batch_size=2,
                          warmup_steps=2, gamma=1.0)
    out = Path(tmp) / "deep"
    report = run(data, config, settings, out, 1, eval_every=1, device="cpu")
    hist = json.loads((out / "training.json").read_text())
    r = hist[0]
    assert "base_G0" in r, r.keys()
    assert abs(r["base_G0"] - r["base_return"]) < 1e-9, r
    print(f"deep OK: base_return={r['base_return']:.3f} base_G0={r['base_G0']:.3f} "
          f"losses={r.get('losses')} validation={report['validation']['base_return']:.3f}")
    return r

def main():
    import tempfile
    check_unit()
    with tempfile.TemporaryDirectory() as tmp:
        check_tabular(tmp)
        try:
            check_deep(tmp)
        except ImportError as e:
            print(f"deep skipped: {e}")
    print("smoke_g0 passed: G0 and TD metrics present in demo logs.")

if __name__ == "__main__":
    main()
