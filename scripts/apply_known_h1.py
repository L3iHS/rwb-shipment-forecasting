"""Create a local H1-refined submission from the fixed V3/MLP forecast."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwb_forecasting.known_offset import CANDIDATE, replace_h1_offset

ANCHOR_SHA = "11466e7371a79a8fda5b22e0d65fba9e76d21da5d350800816f96ce7355edd69"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=ROOT / "data/raw/train_solo_track.parquet")
    p.add_argument("--test", type=Path, default=ROOT / "data/raw/test_solo_track.parquet")
    p.add_argument("--base", type=Path, default=ROOT / "submissions/v3_totals_mlp_shares50.csv")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sidecar = a.output.with_suffix(".metadata.json")
    if a.output.exists() or sidecar.exists() or sidecar.resolve() == a.output.resolve():
        raise FileExistsError("Choose a new CSV and metadata path")
    if hashlib.sha256(a.base.read_bytes()).hexdigest() != ANCHOR_SHA:
        raise ValueError("Expected the reproduced V3/MLP submission for 2025-11-01 10:30")
    train = pd.read_parquet(a.train, columns=["route_id", "timestamp", "target_1h"], use_threads=False)
    g = train.pivot(index="timestamp", columns="route_id", values="target_1h")
    origin = pd.Timestamp("2025-11-01 10:30")
    if (not g.index.equals(pd.date_range("2025-07-28", origin, freq="30min"))
            or not np.array_equal(g.columns, np.arange(1000))):
        raise ValueError("Original route panel and complete known history are required")
    test = pd.read_parquet(a.test, columns=["id", "route_id", "timestamp"], use_threads=False)
    future = pd.date_range(origin + pd.Timedelta(minutes=30), periods=8, freq="30min")
    grid = pd.MultiIndex.from_product([g.columns, future], names=["route_id", "timestamp"])
    b = pd.read_csv(a.base, float_precision="round_trip")
    if (list(b) != ["id", "y_pred"] or len(b) != len(test) or len(test) != len(grid)
            or not test.id.is_unique or set(test.id) != set(b.id) or not b.id.is_unique
            or test.duplicated(["route_id", "timestamp"]).any()):
        raise ValueError("Incomplete or duplicate test/forecast IDs")
    rows = test.merge(b, on="id", validate="one_to_one").set_index(["route_id", "timestamp"]).reindex(grid)
    if rows.isna().any().any():
        raise ValueError("Missing route/time mapping")
    pred = replace_h1_offset(g.to_numpy(), rows.y_pred.to_numpy().reshape(-1, 8))
    result = pd.DataFrame({"id": rows.id.to_numpy(), "y_pred": pred.ravel()}).sort_values("id")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False, mode="x", float_format="%.17g")
    record = dict(model=CANDIDATE, base_sha256=ANCHOR_SHA, origin=str(origin), rows=len(result),
                  csv_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), platform_submission=False)
    with sidecar.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    print(a.output, record["csv_sha256"])


if __name__ == "__main__":
    main()
