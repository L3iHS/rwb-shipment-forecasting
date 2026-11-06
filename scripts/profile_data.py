"""Audit the original competition data without training a forecasting model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parents[1]
STEP = pd.Timedelta(minutes=30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_profile(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("route_id", sort=False)
    sizes = grouped.size()
    intervals = grouped["timestamp"].diff().dropna().value_counts().sort_index()
    numeric = frame.select_dtypes(include="number").drop(columns=["id", "route_id"], errors="ignore")
    return {
        "rows": len(frame),
        "columns": frame.dtypes.astype(str).to_dict(),
        "missing": {key: int(value) for key, value in frame.isna().sum().items()},
        "routes": int(frame["route_id"].nunique()),
        "unique_timestamps": int(frame["timestamp"].nunique()),
        "timestamp_min": str(frame["timestamp"].min()),
        "timestamp_max": str(frame["timestamp"].max()),
        "timestamp_timezone": str(frame["timestamp"].dt.tz),
        "rows_per_route_min": int(sizes.min()),
        "rows_per_route_max": int(sizes.max()),
        "duplicate_route_timestamp": int(frame.duplicated(["route_id", "timestamp"]).sum()),
        "interval_counts_in_file_order": {str(key): int(value) for key, value in intervals.items()},
        "numeric_summary": numeric.describe(percentiles=[0.01, 0.5, 0.9, 0.99]).to_dict() if len(numeric.columns) else {},
    }


def main() -> None:
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Original file changed: {item['path']}")

    train = pd.read_parquet(ROOT / "data/raw/train_solo_track.parquet")
    test = pd.read_parquet(ROOT / "data/raw/test_solo_track.parquet")
    sample = pd.read_csv(ROOT / "data/reference/sample_submission.csv")
    status_cols = [f"status_{index}" for index in range(1, 7)]

    train_profile = table_profile(train)
    test_profile = table_profile(test)
    train_routes = set(train["route_id"])
    test_routes = set(test["route_id"])
    last = train.groupby("route_id")["timestamp"].max()
    test_with_origin = test.join(last.rename("origin"), on="route_id")
    steps = (test_with_origin["timestamp"] - test_with_origin["origin"]) / STEP
    same_routes = train_routes == test_routes
    regular = all(
        profile["duplicate_route_timestamp"] == 0
        and set(profile["interval_counts_in_file_order"]) == {str(STEP)}
        for profile in [train_profile, test_profile]
    )
    sample_valid = (
        list(sample.columns) == ["id", "y_pred"]
        and len(sample) == len(test)
        and sample["id"].is_unique
        and set(sample["id"]) == set(test["id"])
        and bool(np.isfinite(sample["y_pred"]).all())
        and bool(sample["y_pred"].ge(0).all())
    )
    horizon_by_route = test_with_origin.assign(step=steps).groupby("route_id")["step"].agg(list)
    eight_future_points = all(sorted(values) == list(range(1, 9)) for values in horizon_by_route)
    checks = {
        "original_file_checksums_match": True,
        "same_train_test_routes": same_routes,
        "regular_30_minute_grid_in_file_order": regular,
        "eight_next_points_per_route": eight_future_points,
        "test_id_unique": bool(test["id"].is_unique),
        "sample_submission_valid": sample_valid,
        "no_missing_train_test_values": not bool(train.isna().any().any() or test.isna().any().any()),
        "nonnegative_statuses_and_target": bool(train[status_cols + ["target_1h"]].ge(0).all().all()),
    }

    target = train["target_1h"]
    route_totals = train.groupby("route_id")["target_1h"].sum().sort_values(ascending=False)
    target_q99 = float(target.quantile(0.99))
    row_counts = train.groupby("route_id").size()
    daily = train.groupby(train["timestamp"].dt.normalize())["target_1h"].agg(
        target_points="size", target_mean="mean", sum_of_hourly_window_values="sum"
    )
    total_by_time = train.groupby("timestamp")["target_1h"].sum().rename("total_target_1h")
    total_by_time.index.name = "timestamp"

    # Quantify label overlap in the organizer's source-timestamp split.
    # Future labels cover 4 hours, so a simple split of origins is insufficient.
    source_max = train["timestamp"].max() - 8 * STEP
    recent_origins = train.loc[
        (train["timestamp"] >= source_max - pd.Timedelta(days=14))
        & (train["timestamp"] <= source_max),
        "timestamp",
    ]
    split = recent_origins.quantile(0.8)
    label_crossing = (recent_origins <= split) & (recent_origins + 8 * STEP > split)

    result = {
        "libraries": {"pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "train": train_profile,
        "test": test_profile,
        "checks": checks,
        "target": {
            "zero_count": int(target.eq(0).sum()),
            "zero_fraction": float(target.eq(0).mean()),
            "q99": target_q99,
            "share_of_sum_above_q99": float(target[target > target_q99].sum() / target.sum()),
            "top_10_routes_share": float(route_totals.iloc[:10].sum() / route_totals.sum()),
            "route_mean_min": float((route_totals / row_counts).min()),
            "route_mean_max": float((route_totals / row_counts).max()),
            "contemporaneous_correlations": train[status_cols + ["target_1h"]].corr()["target_1h"].to_dict(),
            "zero_count_by_column": {column: int(train[column].eq(0).sum()) for column in status_cols},
        },
        "horizon": {
            "step_counts": {str(key): int(value) for key, value in steps.value_counts().sort_index().items()},
            "forecast_origin_min": str(last.min()),
            "forecast_origin_max": str(last.max()),
            "train_only_routes": sorted(train_routes - test_routes),
            "test_only_routes": sorted(test_routes - train_routes),
        },
        "sample_submission": {
            "rows": len(sample),
            "columns": list(sample.columns),
            "id_min": int(sample["id"].min()),
            "id_max": int(sample["id"].max()),
            "y_pred_summary": sample["y_pred"].describe().to_dict(),
        },
        "organizer_validation_review": {
            "latest_supervised_origin": str(source_max),
            "split_point": str(split),
            "fit_origins_with_label_after_split": int(label_crossing.sum()),
            "note": "Count before optional row sampling. Labels extend beyond the source timestamp split; use a cutoff on the latest label timestamp.",
        },
    }
    reports = ROOT / ".local/reports/data_audit"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    daily.to_csv(reports / "daily_target_summary.csv", index_label="date")
    total_by_time.to_csv(reports / "target_by_timestamp.csv")
    route_totals.rename("sum_of_hourly_window_values").to_csv(reports / "route_target_totals.csv", index_label="route_id")
    print(json.dumps({"checks": checks, "train": {key: value for key, value in train_profile.items() if key != "numeric_summary"}, "test": test_profile, "target": result["target"], "validation": result["organizer_validation_review"]}, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("Data contract check failed; inspect .local/reports/data_audit/data_profile.json")


if __name__ == "__main__":
    main()
