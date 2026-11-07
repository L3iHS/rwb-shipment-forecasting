"""Weekly V3 Tweedie half-hour forecast with known-H1 residual redistribution.

Reuses the attributed V3 feature implementation and exact native600 recipe.
Only the half-hour context/labels, weekly fit policy and redistribution differ.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .halfhour import halfhour_bounds
from .v3_boosting import RuntimeConfig, native_parameters, training_design
from .v3_features import V3Context, fixed_recipe

CANDIDATE = "weekly_v3_latent_half_shares"


def fixed_weekly_recipe():
    return dict(
        base_v3_recipe=fixed_recipe(),
        fit="Saturday10:30; reuse for less than7days",
        half_features="Full forward-only feasible midpoint/width; hourly observation tolerance0.5",
        label="Eight disjoint future half-hour midpoints/source V3 level",
        native_rounds=600,
        sample_weight="Source V3 level, repeated over8horizons",
        decode="Nonnegative half predictions, adjacent sums, current inferred half added toH1",
        composition="Fixed0.5 route-share redistribution of anchor residual at all8horizons; inferred current half reserved inH1",
    )


def full_context(history):
    context = V3Context.from_history(history)
    low, high = halfhour_bounds(context.raw_target, rounding_tolerance=0.5)
    context.half_midpoint = (low + high) / 2
    context.half_width = high - low
    return context


def fit_weekly(
    history,
    work_directory,
    runtime=RuntimeConfig(),
    *,
    dataset_factory=None,
    trainer=None,
    design_factory=training_design,
):
    """Fit one600-tree model on a Saturday prefix; no evaluation set or search."""
    origin = pd.Timestamp(history.timestamps[-1])
    if origin.weekday() != 5 or origin.hour != 10 or origin.minute != 30:
        raise ValueError("Weekly fitting requires Saturday10:30")
    if dataset_factory is None or trainer is None:
        import lightgbm as lgb

        dataset_factory = dataset_factory or lgb.Dataset
        trainer = trainer or lgb.train
    context = full_context(history)
    with design_factory(context, work_directory) as design:
        latest_label = context.times[design.sources[-1] + 8]
        if latest_label > origin:
            raise ValueError("Future fitting labels")
        block_rows = 8 * len(context.routes)
        for number, source in enumerate(design.sources):
            section = slice(number * block_rows, (number + 1) * block_rows)
            # Source-major, then horizon-major, then route. This buffer replaces
            # the original hourly/H1-residual labels without changing features.
            design.normalized_labels[section] = (
                context.half_midpoint[source + 1 : source + 9].reshape(-1) / design.scales[section]
            )
        design.normalized_labels.flush()
        dataset = dataset_factory(
            design.model_matrix,
            label=design.normalized_labels,
            weight=design.scales,
            feature_name=design.model_columns,
            categorical_feature=[],
            free_raw_data=True,
        )
        booster = trainer(native_parameters(runtime), dataset, num_boost_round=600)
        if booster.current_iteration() != 600 or booster.feature_name() != design.model_columns:
            raise ValueError("Incomplete weekly fit or different feature order")
        del dataset
        audit = dict(
            fit_origin=str(origin),
            first_source=str(context.times[design.sources[0]]),
            last_source=str(context.times[design.sources[-1]]),
            latest_label_timestamp=str(latest_label),
            training_sources=len(design.sources),
            rows=len(design.features),
        )
    return booster, audit


def hourly_from_native(prediction, scale, known):
    scale, known = np.asarray(scale, dtype=np.float64), np.asarray(known, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        scale.ndim != 1
        or not len(scale)
        or known.shape != scale.shape
        or prediction.shape != (8 * len(scale),)
        or not np.isfinite(prediction).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.isfinite(known).all()
        or (known < 0).any()
    ):
        raise ValueError(
            "Finite aligned native predictions, positive scales and known halves required"
        )
    half = np.maximum(prediction.reshape(8, -1).T, 0) * scale[:, None]
    hourly = half.copy()
    hourly[:, 1:] += half[:, :-1]
    hourly[:, 0] += known
    return hourly


def redistribute_known_h1(anchor, alternative, known):
    """Preserve each horizon total, reserving each route's inferred H1 part.

    H1's unknown residual is redistributed too; individual H1 forecasts may
    change. The reserved known half is never redistributed.
    """
    anchor, alternative, known = [
        np.asarray(value, dtype=np.float64) for value in (anchor, alternative, known)
    ]
    if (
        anchor.ndim != 2
        or not len(anchor)
        or anchor.shape[1] != 8
        or anchor.shape != alternative.shape
        or known.shape != (len(anchor),)
    ):
        raise ValueError("Unaligned route/horizon grid")
    if any(
        not np.isfinite(value).all() or (value < 0).any() for value in (anchor, alternative, known)
    ):
        raise ValueError("Finite nonnegative predictions required")
    floor = np.zeros_like(anchor)
    floor[:, 0] = known
    a, b = anchor - floor, alternative - floor
    if min(a.min(), b.min()) < -1e-6:
        raise ValueError("Known H1 part exceeds its forecast")
    a, b = np.maximum(a, 0), np.maximum(b, 0)
    total_a, total_b = a.sum(0), b.sum(0)
    valid = total_b > 0
    adjusted = a.copy()
    adjusted[:, valid] = 0.5 * a[:, valid] + 0.5 * (b[:, valid] / total_b[valid]) * total_a[valid]
    result = adjusted + floor
    np.testing.assert_allclose(result.sum(0), anchor.sum(0), rtol=5e-14, atol=1e-6)
    return result


def validate_week(fit_audit, origin):
    origin = pd.Timestamp(origin)
    fitted = pd.Timestamp(fit_audit["fit_origin"])
    if fitted.weekday() != 5 or fitted.hour != 10 or fitted.minute != 30:
        raise ValueError("Checkpoint must have a Saturday10:30 fit origin")
    if not fitted <= origin < fitted + pd.Timedelta(days=7):
        raise ValueError("Wrong Saturday's booster")
    latest = pd.Timestamp(fit_audit["latest_label_timestamp"])
    if latest > fitted or latest != pd.Timestamp(fit_audit["last_source"]) + pd.Timedelta(hours=4):
        raise ValueError("Invalid training label cutoff")


def predict_weekly(history, booster, fit_audit, anchor, runtime=RuntimeConfig()):
    """Predict from the caller's source prefix and one same-week native booster."""
    origin = pd.Timestamp(history.timestamps[-1])
    validate_week(fit_audit, origin)
    context = full_context(history)
    features, scale = context.features(len(context.times) - 1)
    columns = [name for name in features if name != "route_id"]
    if booster.current_iteration() != 600 or booster.feature_name() != columns:
        raise ValueError("Expected selected600-tree booster with exact123 feature columns")
    known = context.half_midpoint[-1]
    raw = booster.predict(features[columns], num_iteration=600, num_threads=runtime.native_threads)
    alternative = hourly_from_native(raw, scale, known)
    result = redistribute_known_h1(anchor, alternative, known)
    audit = dict(
        model=CANDIDATE,
        fit=fit_audit,
        latest_feature_timestamp=str(origin),
        native_threads=runtime.native_threads,
        known_h1_reserved=True,
        maximum_horizon_total_drift=float(abs(result.sum(0) - np.asarray(anchor).sum(0)).max()),
        platform_submission=False,
    )
    return result, audit


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(booster, fit_audit, model_path, audit_path):
    model_path, audit_path = Path(model_path), Path(audit_path)
    if model_path.exists() or audit_path.exists():
        raise FileExistsError("Checkpoint artifacts already exist")
    booster.save_model(str(model_path), num_iteration=600)
    record = dict(
        candidate=CANDIDATE,
        model_sha256=file_sha256(model_path),
        training_audit=fit_audit,
        recipe=fixed_weekly_recipe(),
    )
    with audit_path.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")


def load_checkpoint(model_path, audit_path, *, booster_factory=None):
    """Load portable metadata or the existing native {'audit','boosters'} format."""
    record = json.loads(Path(audit_path).read_text())
    expected = record.get("model_sha256", record.get("boosters", {}).get(CANDIDATE))
    if expected is None or file_sha256(model_path) != expected:
        raise ValueError("Native checkpoint hash mismatch")
    if "recipe" in record and record["recipe"] != fixed_weekly_recipe():
        raise ValueError("Different weekly recipe")
    audit = record.get("training_audit", record.get("audit"))
    if audit is None:
        raise ValueError("Training cutoff metadata required")
    validate_week(audit, audit["fit_origin"])
    if booster_factory is None:
        import lightgbm as lgb

        booster_factory = lgb.Booster
    booster = booster_factory(model_file=str(model_path))
    if booster.current_iteration() != 600:
        raise ValueError("Expected600 native trees")
    return booster, audit
