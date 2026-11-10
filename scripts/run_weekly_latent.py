#!/usr/bin/env python3
"""Regenerate a local weekly latent CSV from raw data and a keyed stage-flow anchor."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb_forecasting.data import ForecastData
from rwb_forecasting.mlp_submission import csv_forecast
from rwb_forecasting.submission import make_submission
from rwb_forecasting.v3_boosting import RuntimeConfig, native_parameters
from rwb_forecasting import v3_boosting, v3_features, weekly_latent
from rwb_forecasting.weekly_latent import (
    CANDIDATE,
    file_sha256,
    fit_weekly,
    fixed_weekly_recipe,
    load_checkpoint,
    predict_weekly,
    save_checkpoint,
)


def load_config(path):
    config = json.loads(Path(path).read_text())
    expected = dict(
        model=CANDIDATE,
        recipe=fixed_weekly_recipe(),
        runtime=dict(native_threads=2),
        native_parameters=native_parameters(RuntimeConfig(2)),
        lightgbm_version="4.7.0",
    )
    if config != expected:
        raise ValueError("Config must describe the exact selected weekly latent recipe at2threads")
    return config, RuntimeConfig(2)


def test_origin(test):
    if test.empty or not {"id", "route_id", "timestamp"}.issubset(test):
        raise ValueError("Original test grid required")
    times = pd.DatetimeIndex(pd.to_datetime(test.timestamp).unique()).sort_values().as_unit("ns")
    if (
        len(times) != 8
        or times.tz is not None
        or not times.equals(pd.date_range(times[0], periods=8, freq="30min").as_unit("ns"))
    ):
        raise ValueError("Exactly8 consecutive half-hour timestamps required")
    origin = times[0] - pd.Timedelta(minutes=30)
    if origin.hour != 10 or origin.minute != 30:
        raise ValueError("The selected weekly recipe forecasts at10:30")
    return origin


def publish_outputs(staged, final):
    """Publish the CSV last and keep existing outputs intact on collisions."""
    linked = []
    try:
        for name in ("model", "fit", "metadata", "csv"):
            os.link(staged[name], final[name])
            linked.append(name)
    except OSError:
        for name in reversed(linked):
            if final[name].exists() and os.path.samefile(staged[name], final[name]):
                final[name].unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train", "test", "anchor", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-in", type=Path)
    parser.add_argument("--fit-audit-in", type=Path)
    args = parser.parse_args(argv)
    if (args.model_in is None) != (args.fit_audit_in is None):
        parser.error("--model-in and --fit-audit-in must be provided together")
    paths = dict(
        csv=args.output,
        metadata=args.output.with_suffix(".metadata.json"),
        model=args.output.with_suffix(".model.txt"),
        fit=args.output.with_suffix(".fit.json"),
    )
    if any(path.exists() for path in paths.values()) or not args.output.parent.is_dir():
        parser.error("Choose a new output path in an existing parent directory")
    if len({path.resolve() for path in paths.values()}) != len(paths):
        parser.error("Companion output paths collide")
    config, runtime = load_config(args.config)
    if version("lightgbm") != config["lightgbm_version"]:
        raise ValueError("Exact native reproduction requires LightGBM4.7.0")
    inputs = {
        name: file_sha256(getattr(args, name)) for name in ("train", "test", "anchor", "config")
    }
    if args.model_in is not None:
        inputs.update(model=file_sha256(args.model_in), fit=file_sha256(args.fit_audit_in))
    sources = {
        "weekly_latent.py": file_sha256(weekly_latent.__file__),
        "v3_features.py": file_sha256(v3_features.__file__),
        "v3_boosting.py": file_sha256(v3_boosting.__file__),
        "run_weekly_latent.py": file_sha256(Path(__file__)),
    }
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        test = pd.read_parquet(
            args.test, columns=["id", "route_id", "timestamp"], use_threads=False
        )
        origin = test_origin(test)
        # File may contain later observations; predictors receive only the cut prefix.
        data = ForecastData.from_frame(
            pd.read_parquet(
                args.train, filters=[("timestamp", "<=", origin.to_pydatetime())], use_threads=False
            )
        )
        if data.timestamps[-1] != origin:
            raise ValueError("Train does not contain the test's forecast origin")
        anchor = csv_forecast(data, test, pd.read_csv(args.anchor, float_precision="round_trip"))
        if args.model_in is None:
            fitted = origin - pd.Timedelta(days=(origin.day_of_week - 5) % 7)
            if args.work_dir is not None:
                args.work_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="weekly_latent_", dir=args.work_dir
            ) as temporary:
                booster, training = fit_weekly(data.history(fitted), Path(temporary), runtime)
        else:
            booster, training = load_checkpoint(args.model_in, args.fit_audit_in)
        prediction, audit = predict_weekly(data, booster, training, anchor, runtime)
        result, audit = make_submission(data, test, predictor=lambda history: (prediction, audit))
    with tempfile.TemporaryDirectory(prefix=".weekly_latent_", dir=args.output.parent) as directory:
        staged = {name: Path(directory) / path.name for name, path in paths.items()}
        # Portable native checkpoint and cutoff metadata are emitted in both modes.
        save_checkpoint(booster, training, staged["model"], staged["fit"])
        result.to_csv(staged["csv"], index=False, mode="x", float_format="%.17g")
        restored = pd.read_csv(staged["csv"], float_precision="round_trip")
        np.testing.assert_array_equal(restored.id, result.id)
        np.testing.assert_array_equal(
            restored.y_pred.to_numpy().view("u8"), result.y_pred.to_numpy().view("u8")
        )
        input_paths = {
            "train": args.train,
            "test": args.test,
            "anchor": args.anchor,
            "config": args.config,
        }
        if args.model_in is not None:
            input_paths.update(model=args.model_in, fit=args.fit_audit_in)
        if any(file_sha256(path) != inputs[name] for name, path in input_paths.items()):
            raise RuntimeError("Input changed during execution")
        record = dict(
            model=CANDIDATE,
            rows=len(result),
            audit=audit,
            config=config,
            input_sha256=inputs,
            output_sha256={
                name: file_sha256(path) for name, path in staged.items() if name != "metadata"
            },
            source_sha256=sources,
            runtime=dict(
                python=platform.python_version(),
                libraries={
                    name: version(name) for name in ["numpy", "pandas", "lightgbm", "pyarrow"]
                },
            ),
            checkpoint_reused=args.model_in is not None,
            elapsed_seconds=time.perf_counter() - started,
            platform_submission=False,
        )
        with staged["metadata"].open("x") as output:
            json.dump(record, output, indent=2, allow_nan=False)
            output.write("\n")
        publish_outputs(staged, paths)
    print(
        f"Created local CSV: {args.output} ({len(result)} rows), SHA256 {record['output_sha256']['csv']}"
    )


if __name__ == "__main__":
    main()
