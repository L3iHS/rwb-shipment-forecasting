import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData
from rwb_forecasting.linear_candidates import aligned_features, normalize_forecast_totals, predict_linear_candidate, source_indices


@pytest.fixture
def panel():
    times = pd.date_range("2025-09-01", periods=30 * 48, freq="30min")
    x = np.arange(len(times))[:, None]
    target = x ** 1.3 + np.array([[19, 47]])
    statuses = np.stack([(x % (k + 3)) + np.array([[19, 47]]) for k in range(6)], axis=-1)
    return ForecastData(times, np.array([19, 47]), statuses.astype(float), target)


def test_aligned_lags_known_expected_values(panel):
    rows = aligned_features(panel, np.array([400, 500]))
    for row, (t, r) in enumerate([(400, 0), (400, 1), (500, 0), (500, 1)]):
        for p in (48, 336):
            for h in range(1, 9):
                assert rows.loc[row, f"target_aligned_{p}_{h}"] == panel.target[t + h - p, r]


def test_features_ignore_future_even_inside_known_history(panel):
    poisoned_target, poisoned_status = panel.target.copy(), panel.statuses.copy()
    poisoned_target[401:] = 1e12
    poisoned_status[401:] = 1e12
    poisoned = ForecastData(panel.timestamps, panel.routes, poisoned_status, poisoned_target)
    pd.testing.assert_frame_equal(aligned_features(panel, np.array([400])), aligned_features(poisoned, np.array([400])))
    pd.testing.assert_frame_equal(aligned_features(panel, np.array([400]), True), aligned_features(poisoned, np.array([400]), True))


@pytest.mark.parametrize("family", ["global", "per_route"])
@pytest.mark.parametrize("solver", ["lsqr", "block_gram"])
def test_prediction_is_causal_and_training_labels_end_before_origin(panel, family, solver):
    origin = panel.timestamps[1200]
    reference = panel.history(origin)
    target, statuses = panel.target.copy(), panel.statuses.copy()
    target[1201:] = 1e12
    statuses[1201:] = 1e12
    changed = ForecastData(panel.timestamps, panel.routes, statuses, target).history(origin)
    first, audit = predict_linear_candidate(reference, family=family, train_days=2, alpha=20, solver=solver)
    second, _ = predict_linear_candidate(changed, family=family, train_days=2, alpha=20, solver=solver)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 8) and np.isfinite(first).all() and (first >= 0).all()
    assert pd.Timestamp(audit["latest_label_timestamp"]) <= origin


def test_clock_filter_is_circular_and_remains_within_label_cutoff(panel):
    history = panel.history(pd.Timestamp("2025-09-25 00:30"))
    indices = source_indices(history, train_days=3, clock_radius=2)
    slots = history.timestamps[indices].hour * 2 + history.timestamps[indices].minute // 30
    assert set(slots) == {47, 0, 1, 2, 3}
    assert indices[-1] + 8 < len(history.timestamps)


@pytest.mark.parametrize("indices", [[], [335], [-1], [1440], [400.1], [[400]]])
def test_invalid_features_fail_closed(panel, indices):
    with pytest.raises(ValueError):
        aligned_features(panel, np.asarray(indices))


@pytest.mark.parametrize("params", [{"train_days": 0}, {"clock_radius": -1}, {"alpha": 0}, {"family": "unknown"}, {"solver": "unknown"}])
def test_invalid_model_configuration_rejected(panel, params):
    with pytest.raises(ValueError):
        predict_linear_candidate(panel, **params)


def test_route_models_do_not_mix_other_route_observations(panel):
    original, _ = predict_linear_candidate(panel, family="per_route", train_days=2, alpha=20)
    changed_target, changed_statuses = panel.target.copy(), panel.statuses.copy()
    changed_target[:, 1] = 7 * changed_target[:, 1] + 10000
    changed_statuses[:, 1] = changed_statuses[:, 1] ** 2
    changed = ForecastData(panel.timestamps, panel.routes, changed_statuses, changed_target)
    prediction, _ = predict_linear_candidate(changed, family="per_route", train_days=2, alpha=20)
    np.testing.assert_array_equal(original[0], prediction[0])
    assert not np.allclose(original[1], prediction[1])


def test_status_dynamics_exact_alignment_and_inclusive_means(panel):
    rows = aligned_features(panel, np.array([400, 500]), status_dynamics=True)
    for row, (t, r) in enumerate([(400, 0), (400, 1), (500, 0), (500, 1)]):
        for k in range(6):
            for lag in (1, 2, 3, 6, 12):
                assert rows.loc[row, f"status_{k+1}_lag_{lag}"] == panel.statuses[t-lag, r, k]
            for window in (6, 12):
                assert rows.loc[row, f"status_{k+1}_mean_{window}"] == pytest.approx(panel.statuses[t-window+1:t+1, r, k].mean())


@pytest.mark.parametrize("solver", ["lsqr", "block_gram"])
def test_aggregate_calibration_uses_known_data_and_audits_both_models(panel, solver):
    origin = panel.timestamps[1200]
    target, statuses = panel.target.copy(), panel.statuses.copy()
    target[1201:] = 1e12
    statuses[1201:] = 1e12
    poisoned = ForecastData(panel.timestamps, panel.routes, statuses, target)
    before, audit = predict_linear_candidate(panel.history(origin), train_days=2, status_dynamics=True, aggregate_calibration=True, solver=solver)
    after, _ = predict_linear_candidate(poisoned.history(origin), train_days=2, status_dynamics=True, aggregate_calibration=True, solver=solver)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(before.sum(axis=0), audit["aggregate_model"]["forecast_totals"])
    for component in (audit, audit["aggregate_model"]):
        assert pd.Timestamp(component["latest_label_timestamp"]) <= origin
        assert component["solver"] == solver


def test_normalization_preserves_shares_and_allocates_zero_forecasts():
    prediction = np.array([[1., 0, 0], [3, 0, 0]])
    actual = normalize_forecast_totals(prediction, np.array([8., 8., 0]), np.array([1., 3.]))
    np.testing.assert_array_equal(actual, [[2, 2, 0], [6, 6, 0]])
    uniform = normalize_forecast_totals(np.zeros((2, 2)), np.array([10., 20.]), np.zeros(2))
    np.testing.assert_array_equal(uniform, [[5, 10], [5, 10]])


@pytest.mark.parametrize("prediction,totals,weights", [
    ([[np.nan, 1]], [1, 1], [1]),
    ([[1, 1]], [1, np.inf], [1]),
    ([[-1, 1]], [1, 1], [1]),
    ([[1, 1]], [1], [1]),
    ([[1, 1]], [1, 1], [-1]),
    ([[1e308, 1e308], [1e308, 1e308]], [1, 1], [1, 1]),
])
def test_normalization_rejects_bad_inputs(prediction, totals, weights):
    with pytest.raises(ValueError):
        normalize_forecast_totals(np.array(prediction), np.array(totals), np.array(weights))
