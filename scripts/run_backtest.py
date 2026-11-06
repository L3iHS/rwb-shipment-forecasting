"""Compare fixed-origin forecasts without reading competition test targets."""

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

from rwb_forecasting.data import ForecastData
from rwb_forecasting.validation import backtest, daily_origins


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-count", type=int, default=7)
    parser.add_argument("--origin-time", default="10:30")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/runs/temporal_baselines")
    args = parser.parse_args()
    if args.train_days < 1 or args.alpha <= 0:
        parser.error("train-days and alpha must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train_path = ROOT / "data/raw/train_solo_track.parquet"
    data = ForecastData.from_frame(pd.read_parquet(train_path))
    origins = daily_origins(data, args.origin_count, args.origin_time)
    print(f"Loaded {len(data.routes)} routes; validation origins: {origins[0]} to {origins[-1]}", flush=True)
    with threadpool_limits(limits=2):
        result = backtest(data, origins, args.train_days, args.alpha, lambda text: print(text, flush=True))
    for name in ["summary", "folds", "by_horizon", "predictions"]:
        result[name].to_csv(output / f"{name}.csv", index=False)
    (output / "training_audit.json").write_text(json.dumps(result["audits"], indent=2) + "\n")
    source_paths = [*sorted((ROOT / "src/rwb_forecasting").glob("*.py")), Path(__file__).resolve(), ROOT / "requirements.txt"]
    metadata = {
        "dataset_sha256": digest(train_path),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]},
        "origins": [str(origin) for origin in origins],
        "forecast_points": 8,
        "step_minutes": 30,
        "routes": len(data.routes),
        "points_per_fold": len(data.routes) * 8,
        "primary_comparison": "arithmetic mean of official score across fixed-origin folds",
        "pooled_note": "Pooled bias can cancel across folds; pooled score is secondary.",
        "ridge": {
            "features": "six statuses at source timestamp and one-hot route_id",
            "source_window_days": args.train_days,
            "alpha": args.alpha, "solver": "lsqr", "tol": 1e-6,
            "negative_predictions": "clipped to zero",
            "sampling": "none; all available sources in the declared window",
        },
        "leakage_contract": "Predictors receive only observations through origin; all training labels end at or before origin.",
        "elapsed_seconds": time.perf_counter() - started,
        "platform_submission": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(result["summary"].to_string(index=False), flush=True)
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
