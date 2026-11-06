"""Create a checked local CSV for manual upload to the competition."""

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

from rwb_forecasting.data import ForecastData
from rwb_forecasting.features import FEATURE_SETS, feature_spec
from rwb_forecasting.submission import make_submission


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="calendar")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.train_days < 1 or args.alpha <= 0:
        parser.error("train-days and alpha must be positive")
    output = (args.output or ROOT / f"submissions/ridge_{args.feature_set}.csv").resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        parser.error("Output already exists; select a different --output path")

    train_path = ROOT / "data/raw/train_solo_track.parquet"
    test_path = ROOT / "data/raw/test_solo_track.parquet"
    data = ForecastData.from_frame(pd.read_parquet(train_path))
    test = pd.read_parquet(test_path)
    with threadpool_limits(limits=2):
        submission, audit = make_submission(data, test, args.train_days, args.alpha, args.feature_set)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, mode="x")
    source_paths = [*sorted((ROOT / "src/rwb_forecasting").glob("*.py")), Path(__file__).resolve(), ROOT / "requirements.txt"]
    metadata = {
        "model": f"ridge_{args.feature_set}", "rows": len(submission),
        "csv_sha256": digest(output),
        "train_sha256": digest(train_path), "test_sha256": digest(test_path),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        "feature_spec": feature_spec(args.feature_set),
        "ridge": {"train_days": args.train_days, "alpha": args.alpha, "solver": "lsqr", "tol": 1e-6},
        "training_audit": audit,
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "platform_submission": False,
    }
    with metadata_path.open("x") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    print(f"Ready CSV: {output}\nRows: {len(submission)}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
