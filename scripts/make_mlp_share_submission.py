"""Regenerate the fixed MLP/share CSV from known train and certified inputs."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb_forecasting.data import ForecastData
from rwb_forecasting import data, experiments, halfhour, latent_volume, linear_candidates, mass_preserving, submission
from rwb_forecasting import mlp_features, mlp_model, mlp_spec, mlp_submission
from rwb_forecasting.mlp_model import predict_latent_mlp
from rwb_forecasting.mlp_spec import statistical_spec, specification_sha
from rwb_forecasting.mlp_submission import CANDIDATE, make_mlp_share_submission


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    expected = dict(model=CANDIDATE, latent_mlp=statistical_spec(), specification_sha256=specification_sha(),
        torch_version="2.15.0",
        composition=["mass_preserving_blend(MAE25,raw_mlp_hourly,.5)", "mass_preserving_blend(V3,final_mlp,.5)"],
        runtime=dict(native_threads=2, interop_threads=2, device="cpu"))
    if config != expected:
        raise ValueError("Configuration must describe the fixed original MLP and two-layer50% composition")
    return config


def snapshot(config_path):
    paths = [Path(__file__), Path(config_path), ROOT / "src/rwb_forecasting/mlp_torch.py"]
    paths += [Path(module.__file__) for module in (data, experiments, halfhour, latent_volume,
        linear_candidates, mass_preserving, submission, mlp_features, mlp_model, mlp_spec, mlp_submission)]
    return {str(path.resolve()):digest(path) for path in paths}


def validate_paths(args):
    """Reject conflicting file names before reading the panel or fitting."""
    inputs = [args.train, args.test, args.config, args.mae25_csv, args.v3_csv]
    outputs = [args.output, args.output.with_suffix(".metadata.json")]
    if args.resume and args.checkpoint is None:
        raise ValueError("--resume requires --checkpoint")
    if args.checkpoint is not None:
        cp = args.checkpoint
        temporary = cp.with_name(cp.name + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError("An unfinished checkpoint temporary file exists")
        outputs += [cp, cp.with_suffix(".preprocessing.npz"), temporary]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved) or set(resolved) & {p.resolve() for p in inputs}:
        raise ValueError("Output, checkpoint and inputs must use distinct paths")
    if any(path.is_symlink() for path in outputs):
        raise ValueError("Output and checkpoint paths must not be symbolic links")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3_totals_mlp_shares50.json")
    for name in ("train", "test", "mae25-csv", "v3-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mae25-sha256", required=True)
    parser.add_argument("--v3-sha256", required=True)
    parser.add_argument("--forecast-origin", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    validate_paths(args)
    metadata_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("Output exists; select a new path")
    config = load_config(args.config)
    for path, expected in ((args.mae25_csv, args.mae25_sha256), (args.v3_csv, args.v3_sha256)):
        if len(expected) != 64 or digest(path) != expected:
            raise ValueError("Certified input CSV SHA256 mismatch")
    inputs = {name:dict(path=str(path), sha256=digest(path)) for name, path in
        (("train", args.train), ("test", args.test), ("mae25_csv", args.mae25_csv), ("v3_csv", args.v3_csv))}
    sources = snapshot(args.config)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        known = ForecastData.from_frame(pd.read_parquet(args.train, use_threads=False))
        if known.timestamps[-1] != pd.Timestamp(args.forecast_origin):
            raise ValueError("Certified CSV forecast origin must equal the known train endpoint")
        test = pd.read_parquet(args.test, use_threads=False)
        mae25 = pd.read_csv(args.mae25_csv, float_precision="round_trip")
        v3 = pd.read_csv(args.v3_csv, float_precision="round_trip")
        forecast = lambda history: predict_latent_mlp(history, checkpoint_path=args.checkpoint, resume=args.resume)
        result, audit = make_mlp_share_submission(known, test, mae25, v3, raw_predictor=forecast)
    if snapshot(args.config) != sources or any(digest(item["path"]) != item["sha256"] for item in inputs.values()):
        raise RuntimeError("Input or source identity changed during the fit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, mode="x", float_format="%.17g")
    checked = pd.read_csv(args.output, float_precision="round_trip", dtype={"y_pred":"float64"})
    pd.testing.assert_frame_equal(checked, result)
    metadata = dict(model=CANDIDATE, config=config, training_audit=audit,
        rows=len(result), csv_sha256=digest(args.output), input_sha256=inputs,
        source_sha256=sources, python=platform.python_version(),
        libraries={name:version(name) for name in ("numpy", "pandas", "scikit-learn", "torch")},
        elapsed_seconds=time.perf_counter() - started, platform_submission=False)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2); stream.write("\n")
    print(f"Created local CSV: {args.output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
