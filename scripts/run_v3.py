"""Backtest explicit origins or create a checked local V3 CSV; never uploads."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb_forecasting import data as data_module, experiments, metrics, submission as submission_module
from rwb_forecasting.data import ForecastData
from rwb_forecasting.experiments import compare_models
from rwb_forecasting.submission import make_submission
from rwb_forecasting import v3_features, v3_boosting
from rwb_forecasting.v3_features import fixed_recipe
from rwb_forecasting.v3_boosting import CANDIDATE, RuntimeConfig, native_parameters, predict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    expected_keys = {"model", "recipe", "runtime", "native_parameters", "lightgbm_version", "attribution"}
    if set(config) != expected_keys:
        raise ValueError("Unknown or missing fixed config fields")
    runtime = RuntimeConfig(**config["runtime"])
    attribution = dict(author="Vlado", commit=v3_features.SOURCE_COMMIT,
        git_blob=v3_features.SOURCE_GIT_BLOB, source_sha256=v3_features.SOURCE_SHA256,
        url=v3_features.SOURCE_URL)
    if (config["model"] != CANDIDATE or config["recipe"] != fixed_recipe()
            or runtime.native_threads != 2 or config["native_parameters"] != native_parameters(runtime)
            or config["lightgbm_version"] != "4.7.0" or config["attribution"] != attribution):
        raise ValueError("Config must describe the fixed attributed T_h1s600 recipe at two threads")
    return config, runtime


def ensure_new_output(path, *, directory=False):
    path = Path(path)
    metadata = path / "metadata.json" if directory else path.with_suffix(".metadata.json")
    if path.exists() or metadata.exists():
        raise FileExistsError("Output already exists; choose a new path")


def source_snapshot(config_path):
    paths = [Path(__file__), Path(config_path), Path(data_module.__file__), Path(experiments.__file__), Path(metrics.__file__),
             Path(submission_module.__file__), Path(v3_features.__file__), Path(v3_boosting.__file__)]
    if (ROOT / "requirements.txt").exists():
        paths.append(ROOT / "requirements.txt")
    return {str(path.resolve()): digest(path) for path in paths}


def predictor(runtime, work_directory):
    return lambda history: predict(history, work_directory, runtime)


def backtest_explicit(data, origins, forecast, progress=None):
    """Existing common evaluator supplies a separate history prefix at every origin."""
    dates = [pd.Timestamp(origin) for origin in origins]
    return compare_models(data, dates, {CANDIDATE: forecast}, progress=progress)


def local_submission(data, test, forecast):
    """Reuse the project's route/timestamp-to-ID contract; no positional mapping."""
    return make_submission(data, test, predictor=forecast)


def write_backtest(result, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name in ["summary", "folds", "by_horizon", "predictions"]:
        result[name].to_csv(directory / f"{name}.csv", index=False, mode="x")
    with (directory / "training_audit.json").open("x") as output:
        json.dump(result["audits"], output, indent=2)
    with (directory / "metadata.json").open("x") as output:
        json.dump(metadata, output, indent=2)


def write_submission(frame, path, metadata):
    path = Path(path)
    ensure_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x")
    metadata = {**metadata, "csv_sha256": digest(path), "rows": len(frame)}
    with path.with_suffix(".metadata.json").open("x") as output:
        json.dump(metadata, output, indent=2)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["backtest", "submit"]:
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--work-dir", type=Path)
        if name == "backtest":
            command.add_argument("--origins", nargs="+", required=True)
            command.add_argument("--output-dir", type=Path, required=True)
        else:
            command.add_argument("--test", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = argument_parser(); args = parser.parse_args(argv)
    destination = args.output_dir if args.command == "backtest" else args.output
    ensure_new_output(destination, directory=args.command == "backtest")
    output_paths = [destination.resolve()]
    if args.command == "submit":
        output_paths.append(destination.with_suffix(".metadata.json").resolve())
    if args.work_dir is not None and any(
        args.work_dir.resolve().is_relative_to(path) for path in output_paths
    ):
        raise ValueError("Work directory cannot be inside an output destination")
    config, runtime = load_config(args.config)
    if version("lightgbm") != config["lightgbm_version"]:
        raise ValueError("The fixed native recipe requires LightGBM4.7.0")
    sources = source_snapshot(args.config)
    inputs = {str(args.train.resolve()): digest(args.train)}
    if args.command == "submit":
        inputs[str(args.test.resolve())] = digest(args.test)
    data = ForecastData.from_frame(pd.read_parquet(args.train))
    started = time.perf_counter()
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rwb_v3_", dir=args.work_dir) as temporary:
        forecast = predictor(runtime, Path(temporary))
        with threadpool_limits(limits=runtime.native_threads):
            if args.command == "backtest":
                result = backtest_explicit(data, args.origins, forecast,
                    progress=lambda message: print(message, flush=True))
            else:
                result, audit = local_submission(data, pd.read_parquet(args.test), forecast)
    if source_snapshot(args.config) != sources or any(digest(path) != expected for path, expected in inputs.items()):
        raise RuntimeError("Input or source files changed during execution")
    metadata = {"model": CANDIDATE, "command": args.command, "config": config,
        "source_sha256": sources, "input_sha256": inputs, "python": platform.python_version(),
        "libraries": {name: version(name) for name in ["numpy", "pandas", "scipy", "lightgbm", "pyarrow"]},
        "elapsed_seconds": time.perf_counter() - started, "platform_submission": False}
    if args.command == "backtest":
        metadata.update(origins=[str(pd.Timestamp(origin)) for origin in args.origins],
            primary_metric="arithmetic mean of per-origin WAPE + absolute relative bias")
        write_backtest(result, destination, metadata)
        print(result["summary"].to_string(index=False))
    else:
        metadata["training_audit"] = audit
        write_submission(result, destination, metadata)
        print(f"Created local CSV: {destination} ({len(result)} rows)")


if __name__ == "__main__":
    main()
