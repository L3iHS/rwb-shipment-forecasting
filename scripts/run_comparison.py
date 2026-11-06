"""Reproduce a versioned comparison on research or reserved historical origins."""

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb_forecasting.candidates import configured_comparison
from rwb_forecasting.data import ForecastData
from rwb_forecasting.experiments import compare_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/comparison.json")
    parser.add_argument("--split", choices=["research", "control"], default="research")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--all-research-models", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    thread_count = config.get("compute_threads", 2)
    if type(thread_count) is not int or thread_count < 1:
        parser.error("compute_threads must be a positive integer")
    if args.all_research_models and (args.models or args.split != "research"):
        parser.error("--all-research-models requires research and cannot be combined with --models")
    comparison_models = config.get("comparison_models", [config["baseline"], config["selected_model"]])
    names = (list(config["models"]) if args.all_research_models else args.models
             or comparison_models)
    if len(set(names)) != len(names) or any(name not in config["models"] for name in names):
        parser.error("Select unique known models")
    if args.split == "control" and set(names) != set(comparison_models):
        parser.error("Control comparison is restricted to the frozen finalist and declared baselines")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; select a fresh path")
    predictors = configured_comparison({name: config["models"][name] for name in names})
    data_path = ROOT / "data/raw/train_solo_track.parquet"
    started = time.perf_counter()
    data = ForecastData.from_frame(pd.read_parquet(data_path))
    origins = list(pd.to_datetime(config[f"{args.split}_origins"]))
    with threadpool_limits(limits=thread_count):
        results = compare_models(data, origins, predictors, lambda message: print(message, flush=True))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["summary", "folds", "by_horizon", "predictions"]:
        results[name].to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "training_audit.json").write_text(json.dumps(results["audits"], indent=2) + "\n")
    source_paths = [*sorted((ROOT / "src/rwb_forecasting").glob("*.py")), Path(__file__).resolve(), ROOT / "requirements.txt"]
    metadata = {
        "split": args.split, "config": config, "evaluated_models": names,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "native_thread_limit": thread_count,
        "shared_component_cache": "identical model specs reused only within the same history object",
        "libraries": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "primary_comparison": "arithmetic mean of per-origin WAPE + absolute relative bias",
        "elapsed_seconds": time.perf_counter() - started, "platform_submission": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(results["summary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
