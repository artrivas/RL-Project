"""Check P2 dependencies and M5 file headers, without training a retail policy."""

import argparse
import csv
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


SCHEMAS = {
    "sales_train_evaluation.csv": {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"},
    "sell_prices.csv": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
    "calendar.csv": {"date", "d", "wm_yr_wk", "weekday", "month", "year", "snap_CA", "event_name_1", "event_name_2"},
}
DEPENDENCIES = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib",
    "scikit-learn": "sklearn", "pyarrow": "pyarrow", "torch": "torch",
    "gymnasium": "gymnasium", "stable-baselines3": "stable_baselines3",
    "jupyterlab": "jupyterlab", "pytest": "pytest",
}


def inspect_data(directory):
    """Read only headers and one row. This is not the full data-quality audit."""
    directory = Path(directory)
    results = []
    for name, required in SCHEMAS.items():
        path = directory / name
        result = {"file": name, "status": "ok", "errors": []}
        if not path.is_file():
            result["status"] = "missing"
            if name == "sales_train_evaluation.csv" and (directory / "sales_train_validation.csv").exists():
                result["errors"].append("The 1913-day validation file does not replace the required 1941-day evaluation file.")
            results.append(result)
            continue
        try:
            result["size_bytes"] = path.stat().st_size
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                first_row = next(reader, None)
            if len(header) != len(set(header)):
                result["errors"].append("Duplicate column names.")
            missing = sorted(required - set(header))
            if missing:
                result["errors"].append("Missing columns: " + ", ".join(missing))
            if first_row is None or len(first_row) != len(header):
                result["errors"].append("Missing or malformed first data row.")
            if name == "sales_train_evaluation.csv":
                days = {column for column in header if column.startswith("d_")}
                expected = {f"d_{day}" for day in range(1, 1942)}
                if days != expected:
                    result["errors"].append("Expected exactly daily columns d_1 through d_1941; do not silently change the temporal split.")
        except (OSError, UnicodeError, csv.Error) as exc:
            result["errors"].append(str(exc))
        if result["errors"]:
            result["status"] = "error"
        results.append(result)
    return results


def inspect_dependencies():
    results = []
    for distribution, module in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
            results.append({"package": distribution, "status": "ok", "version": metadata.version(distribution)})
        except Exception as exc:
            results.append({"package": distribution, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return results


def smoke_check():
    """Tiny CPU integration check on CartPole. No M5 data or project results."""
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from stable_baselines3 import DQN, PPO

    torch.set_num_threads(1)
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    loss = layer(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite tensor loss.")
    for algorithm in (DQN, PPO):
        env = gym.make("CartPole-v1")
        try:
            kwargs = {"n_steps": 32, "batch_size": 16, "n_epochs": 1} if algorithm is PPO else {
                "buffer_size": 128, "learning_starts": 16, "batch_size": 16, "train_freq": 4, "gradient_steps": 1,
            }
            model = algorithm("MlpPolicy", env, seed=0, device="cpu", verbose=0, **kwargs)
            model.learn(total_timesteps=64)
            observation, _ = env.reset(seed=123)
            action, _ = model.predict(observation, deterministic=True)
            if not env.action_space.contains(int(action)):
                raise RuntimeError(f"Invalid {algorithm.__name__} action.")
            if not all(torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
                raise RuntimeError(f"Non-finite {algorithm.__name__} parameters.")
        finally:
            env.close()
    with tempfile.TemporaryDirectory() as temporary:
        figure, axis = plt.subplots()
        try:
            axis.plot(np.arange(3), [0, 1, 2])
            figure.savefig(Path(temporary) / "smoke.png")
        finally:
            plt.close(figure)
    return {"status": "ok", "environment": "CartPole-v1", "steps_per_algorithm": 64,
            "algorithms": ["DQN", "PPO"], "device": "cpu", "is_retail_experiment": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parents[2] / "notebooks" / "data" / "m5" / "raw"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("M5_DATA_DIR", default_data)))
    parser.add_argument("--require-data", action="store_true", help="Fail if any M5 input is missing.")
    parser.add_argument("--skip-dependencies", action="store_true", help="Only check data headers with standard-library Python.")
    parser.add_argument("--smoke", action="store_true", help="Run tiny torch/DQN/PPO/plot integration checks.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.smoke and args.skip_dependencies:
        parser.error("--smoke cannot be combined with --skip-dependencies")

    dependencies = [] if args.skip_dependencies else inspect_dependencies()
    data = inspect_data(args.data_dir)
    errors = [entry for entry in dependencies if entry["status"] == "error"]
    errors += [entry for entry in data if entry["status"] == "error" or (args.require_data and entry["status"] == "missing")]
    smoke = {"status": "not_run"}
    if args.smoke:
        if any(entry["status"] == "error" for entry in dependencies):
            smoke = {"status": "error", "error": "Dependencies are incomplete; smoke check cannot run."}
        else:
            try:
                smoke = smoke_check()
            except Exception as exc:
                smoke = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if smoke["status"] == "error":
            errors.append(smoke)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "status": "error" if errors else "ok",
        "dependencies_checked": not args.skip_dependencies,
        "dependencies": dependencies, "data_directory": str(args.data_dir), "data": data,
        "data_headers_ready": all(entry["status"] == "ok" for entry in data),
        "full_data_audit_completed": False,
        "retail_training_implemented": True,
        "retail_training_algorithms": ["qlearning", "ddqn", "ppo"],
        "retail_deep_training_implemented": True,
        "retail_deep_numerical_validation": "run_preflight_required",
        "smoke": smoke,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
