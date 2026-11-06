import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData
from rwb_forecasting.features import FEATURE_SETS, feature_frame
from rwb_forecasting.models import predict_ridge, training_samples
from rwb_forecasting.validation import backtest, daily_origins


@pytest.fixture
def panel():
    times = pd.date_range("2025-10-01", periods=21 * 48, freq="30min")
    index = np.repeat(np.arange(len(times)), 2)
    routes = np.tile([19, 47], len(times))
    frame = pd.DataFrame({
        "timestamp": np.repeat(times, 2), "route_id": routes,
        "target_1h": index**2 + routes,
        **{f"status_{k}": (index + k) % 17 + routes for k in range(1, 7)},
    })
    return frame


def test_lags_and_inclusive_rolling_means_align_with_route_and_source(panel):
    history = ForecastData.from_frame(panel)
    features = feature_frame(history, np.array([400, 500]), feature_set="calendar")
    assert features["route_id"].tolist() == [19, 47, 19, 47]
    for row, (source, route) in enumerate([(400, 19), (400, 47), (500, 19), (500, 47)]):
        for lag in [0, 1, 2, 3, 6, 12, 48, 336]:
            assert features.loc[row, f"target_lag_{lag}"] == (source - lag)**2 + route
        for window in [6, 12, 48]:
            expected = np.mean(np.arange(source - window + 1, source + 1)**2 + route)
            assert features.loc[row, f"target_mean_{window}"] == pytest.approx(expected)
    assert features.loc[0, "slot_of_day"] == 16  # 2025-10-09 08:00
    assert features.loc[0, "day_of_week"] == 3  # Thursday


@pytest.mark.parametrize("feature_set,columns", [("status", 7), ("lags", 15), ("rolling", 18), ("calendar", 20)])
def test_feature_sets_add_only_the_declared_blocks(panel, feature_set, columns):
    history = ForecastData.from_frame(panel)
    result = feature_frame(history, np.array([400]), feature_set=feature_set)
    assert result.shape == (2, columns)
    assert not result.isna().any().any()


@pytest.mark.parametrize("indices", [[], [-1], [335], [1008], [400.5], [[400]]])
def test_feature_builder_rejects_unavailable_or_invalid_sources(panel, indices):
    with pytest.raises(ValueError):
        feature_frame(ForecastData.from_frame(panel), np.asarray(indices), feature_set="calendar")


def test_unknown_feature_set_is_rejected(panel):
    with pytest.raises(ValueError, match="feature"):
        feature_frame(ForecastData.from_frame(panel), np.array([400]), feature_set="unknown")


def test_training_features_cannot_see_later_history_even_before_validation_cutoff(panel):
    source = pd.Timestamp("2025-10-09 08:00")
    reference = ForecastData.from_frame(panel)
    poisoned = panel.copy()
    columns = ["target_1h", *[f"status_{k}" for k in range(1, 7)]]
    poisoned.loc[poisoned["timestamp"] > source, columns] = 10**12
    hidden = ForecastData.from_frame(poisoned)
    for feature_set in FEATURE_SETS:
        pd.testing.assert_frame_equal(
            feature_frame(reference, np.array([400]), feature_set),
            feature_frame(hidden, np.array([400]), feature_set),
        )


def test_enriched_training_preserves_label_boundary_and_skips_missing_lag_history(panel):
    data = ForecastData.from_frame(panel)
    history = data.history(data.timestamps[360])
    features, labels, audit = training_samples(history, train_days=14, feature_set="calendar")
    assert audit["first_source_timestamp"] == str(data.timestamps[336])
    assert audit["last_source_timestamp"] == str(data.timestamps[352])
    assert audit["latest_label_timestamp"] == str(data.timestamps[360])
    assert audit["training_rows"] == 17 * 2
    assert audit["earliest_feature_timestamp"] == str(data.timestamps[0])
    np.testing.assert_array_equal(labels[-2:], history.target[-8:].T)
    np.testing.assert_array_equal(features["target_lag_336"].to_numpy()[:2], history.target[0])
    with pytest.raises(ValueError, match="history"):
        training_samples(data.history(data.timestamps[340]), feature_set="calendar")


@pytest.mark.parametrize("feature_set", ["lags", "rolling", "calendar"])
def test_enriched_predictions_do_not_change_when_validation_future_is_poisoned(panel, feature_set):
    origin = pd.Timestamp("2025-10-15 10:30")
    reference = ForecastData.from_frame(panel).history(origin)
    poisoned = panel.copy()
    columns = ["target_1h", *[f"status_{k}" for k in range(1, 7)]]
    poisoned.loc[poisoned["timestamp"] > origin, columns] = 10**12
    hidden = ForecastData.from_frame(poisoned).history(origin)
    first, _ = predict_ridge(reference, train_days=1, feature_set=feature_set)
    second, _ = predict_ridge(hidden, train_days=1, feature_set=feature_set)
    np.testing.assert_allclose(first, second, atol=0, rtol=0)
    assert first.shape == (2, 8)
    assert np.isfinite(first).all() and (first >= 0).all()


def test_comparison_retains_existing_forecasts_and_evaluation_points(panel):
    data = ForecastData.from_frame(panel)
    origins = daily_origins(data, count=1)
    original = backtest(data, origins, train_days=1)
    extended = backtest(data, origins, train_days=1, ridge_feature_sets=FEATURE_SETS)
    assert len(extended["summary"]) == 8
    assert len(extended["folds"]) == 8
    assert len(extended["by_horizon"]) == 8 * 8
    assert len(extended["audits"]) == 4
    pd.testing.assert_frame_equal(original["predictions"], extended["predictions"][original["predictions"].columns])
    assert len(set(audit["training_rows"] for audit in extended["audits"])) == 1


@pytest.mark.parametrize("sets", [[], ["unknown"], ["status", "status"]])
def test_backtest_rejects_invalid_feature_set_selection(panel, sets):
    data = ForecastData.from_frame(panel)
    with pytest.raises(ValueError, match="feature"):
        backtest(data, daily_origins(data, count=1), train_days=1, ridge_feature_sets=sets)
