#!/usr/bin/env python3
"""Reproduce the fixed stage-flow adjustment from raw train/test and an anchor CSV."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb_forecasting.stage_flow import load_stage_flow_panel, predict_stage_flow


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def align_inputs(test, anchor):
    """Validate the complete eight-step grid and align predictions by original ID."""
    if (not {"id", "route_id", "timestamp"}.issubset(test.columns)
            or list(anchor.columns) != ["id", "y_pred"] or test.empty
            or test[["id", "route_id", "timestamp"]].isna().any().any()
            or anchor.isna().any().any() or test.id.duplicated().any() or anchor.id.duplicated().any()
            or test.duplicated(["route_id", "timestamp"]).any()
            or len(test) != len(anchor) or set(test.id) != set(anchor.id)):
        raise ValueError("Invalid test grid or anchor IDs/columns")
    test = test.sort_values(["route_id", "timestamp"]).copy()
    test["timestamp"] = pd.to_datetime(test.timestamp)
    times = pd.DatetimeIndex(test.timestamp.unique()).sort_values().as_unit("ns")
    routes = np.sort(test.route_id.unique())
    if (len(times) != 8 or times.tz is not None
            or not times.equals(pd.date_range(times[0], periods=8, freq="30min").as_unit("ns"))
            or times[0] != times[0].floor("30min") or len(test) != len(routes) * 8
            or not np.array_equal(test.route_id.to_numpy(), np.repeat(routes, 8))
            or not np.array_equal(test.timestamp.to_numpy(dtype="datetime64[ns]"), np.tile(times.to_numpy(), len(routes)))):
        raise ValueError("Test must contain every route at eight consecutive half-hour timestamps")
    origin = times[0] - pd.Timedelta(minutes=30)
    if origin.hour != 10 or origin.minute != 30:
        raise ValueError("The fixed stage-flow recipe requires a 10:30 origin")
    values = anchor.set_index("id").loc[test.id, "y_pred"].to_numpy(dtype=np.float64).reshape(len(routes), 8)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Anchor predictions must be finite and nonnegative")
    return test, routes, origin, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sidecar = args.output.with_suffix(".metadata.json")
    if args.output.exists() or sidecar.exists() or args.output.resolve() == sidecar.resolve():
        parser.error("Output CSV or metadata already exists; choose a fresh output path")
    if not args.output.parent.is_dir():
        parser.error("Output parent directory must already exist")
    with threadpool_limits(limits=1):
        test, routes, origin, base = align_inputs(
            pd.read_parquet(args.test, columns=["id", "route_id", "timestamp"], use_threads=False),
            pd.read_csv(args.anchor, float_precision="round_trip"))
        panel = load_stage_flow_panel(args.train, origin, routes)
        prediction, audit = predict_stage_flow(panel, base)
        result = pd.DataFrame({"id": test.id.to_numpy(), "y_pred": prediction.ravel()}).sort_values("id")
        result.to_csv(args.output, index=False, mode="x", float_format="%.17g")
        restored = pd.read_csv(args.output, float_precision="round_trip")
        np.testing.assert_array_equal(restored.id, result.id)
        np.testing.assert_array_equal(restored.y_pred.to_numpy().view("u8"), result.y_pred.to_numpy().view("u8"))
        metadata = dict(rows=len(result), csv_sha256=_sha(args.output),
                        inputs={name: _sha(getattr(args, name)) for name in ("train", "test", "anchor")},
                        audit=audit, platform_submission=False)
        with sidecar.open("x") as stream:
            stream.write(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
