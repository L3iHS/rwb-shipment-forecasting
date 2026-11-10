#!/usr/bin/env python3
"""Fit the fixed weekly route-window model from raw train/test and an anchor CSV."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb_forecasting.data import ForecastData
from rwb_forecasting.route_window import (fit_weekly_route_window, load_route_window_panel,
                                          predict_weekly_route_window)
from rwb_forecasting.submission import make_submission


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def align_inputs(test, anchor):
    if (not {"id", "route_id", "timestamp"}.issubset(test.columns)
            or list(anchor.columns) != ["id", "y_pred"] or test.empty
            or test[["id", "route_id", "timestamp"]].isna().any().any()
            or anchor.isna().any().any() or test.id.duplicated().any() or anchor.id.duplicated().any()
            or test.duplicated(["route_id", "timestamp"]).any()
            or len(test) != len(anchor) or set(test.id) != set(anchor.id)):
        raise ValueError("Invalid original test grid or anchor IDs/columns")
    ordered = test.sort_values(["route_id", "timestamp"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered.timestamp)
    times = pd.DatetimeIndex(ordered.timestamp.unique()).sort_values().as_unit("ns")
    routes = np.sort(ordered.route_id.unique())
    if (len(times) != 8 or times.tz is not None or times[0] != times[0].floor("30min")
            or not times.equals(pd.date_range(times[0], periods=8, freq="30min").as_unit("ns"))
            or len(ordered) != len(routes) * 8
            or not np.array_equal(ordered.route_id, np.repeat(routes, 8))
            or not np.array_equal(ordered.timestamp.to_numpy(dtype="datetime64[ns]"), np.tile(times.to_numpy(), len(routes)))):
        raise ValueError("Test must contain every route at the next eight consecutive half-hour timestamps")
    origin = times[0] - pd.Timedelta(minutes=30)
    if origin.hour != 10 or origin.minute != 30:
        raise ValueError("The fixed weekly recipe requires a10:30 origin")
    values = anchor.set_index("id").loc[ordered.id, "y_pred"].to_numpy(dtype=np.float64).reshape(len(routes), 8)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Anchor forecasts must be finite and nonnegative")
    return ordered, routes, origin, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "test", "anchor", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    artifacts = dict(csv=args.output, metadata=args.output.with_suffix(".metadata.json"),
                     model=args.output.with_suffix(".model.txt"), route_totals=args.output.with_suffix(".route_totals.npy"))
    if len({path.resolve() for path in artifacts.values()}) != len(artifacts) or any(path.exists() for path in artifacts.values()):
        parser.error("CSV or companion artifact already exists; choose a fresh output path")
    if not args.output.parent.is_dir():
        parser.error("Output parent directory must already exist")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        test, routes, origin, anchor = align_inputs(
            pd.read_parquet(args.test, columns=["id", "route_id", "timestamp"], use_threads=False),
            pd.read_csv(args.anchor, float_precision="round_trip"))
        panel = load_route_window_panel(args.train, origin, routes)
        fit_origin = origin - pd.Timedelta(days=(origin.weekday() - 5) % 7)
        model, fit_audit = fit_weekly_route_window(panel.prefix(fit_origin))
        prediction, route_totals, decision = predict_weekly_route_window(panel, model, fit_audit, anchor)
        # Existing public mapper validates causal audit, the exact future grid and original IDs.
        data = ForecastData(panel.times, panel.route_ids, panel.raw[:, :, 1:], panel.raw[:, :, 0])
        audit = dict(model="weekly_route_window_l1", fit=fit_audit, reconciliation=decision,
                     latest_feature_timestamp=str(origin), platform_submission=False)
        result, audit = make_submission(data, test, predictor=lambda history: (prediction, audit))
        np.testing.assert_array_equal(prediction[:, 0].view("u8"), anchor[:, 0].view("u8"))
        np.testing.assert_allclose(prediction[:, 1:].sum(0), anchor[:, 1:].sum(0), rtol=2e-10, atol=5e-4)
        result.to_csv(artifacts["csv"], index=False, mode="x", float_format="%.17g")
        restored = pd.read_csv(artifacts["csv"], float_precision="round_trip")
        np.testing.assert_array_equal(restored.id, result.id)
        np.testing.assert_array_equal(restored.y_pred.to_numpy().view("u8"), result.y_pred.to_numpy().view("u8"))
        model.save_model(str(artifacts["model"]))
        with artifacts["route_totals"].open("xb") as stream:
            np.save(stream, route_totals, allow_pickle=False)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = dict(candidate="weekly_route_window_l1", rows=len(result), audit=audit,
                      runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                   pandas=pd.__version__, lightgbm=version("lightgbm")),
                      inputs={name: _sha(getattr(args, name)) for name in ("train", "test", "anchor")},
                      outputs={name: _sha(path) for name, path in artifacts.items() if name != "metadata"},
                      elapsed_seconds=time.perf_counter() - started,
                      peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
                      platform_submission=False)
        with artifacts["metadata"].open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
