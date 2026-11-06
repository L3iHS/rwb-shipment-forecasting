"""Fit the frozen experiment finalist and save a checked local competition CSV."""

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb_forecasting.candidates import configured_predictor
from rwb_forecasting.data import ForecastData
from rwb_forecasting.submission import make_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/comparison.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    name = config["selected_model"]
    spec = config["models"][name]
    output = args.output or ROOT / f"submissions/{name}.csv"
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        parser.error("Output already exists; select a new --output path")
    train = ROOT / "data/raw/train_solo_track.parquet"
    test = ROOT / "data/raw/test_solo_track.parquet"
    data = ForecastData.from_frame(pd.read_parquet(train))
    with threadpool_limits(limits=2):
        submission, audit = make_submission(data, pd.read_parquet(test), predictor=configured_predictor(spec))
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, mode="x")
    source_paths = [*sorted((ROOT / "src/rwb_forecasting").glob("*.py")), Path(__file__).resolve(), ROOT / "requirements.txt"]
    metadata = {
        "model": name, "spec": spec, "rows": len(submission), "training_audit": audit,
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "platform_submission": False,
    }
    with metadata_path.open("x") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    print(f"Ready CSV: {output}\nRows: {len(submission)}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
