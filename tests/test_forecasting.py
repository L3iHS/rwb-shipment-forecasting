import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData
from rwb_forecasting.metrics import evaluate
from rwb_forecasting.models import predict_simple, predict_ridge, training_samples
from rwb_forecasting.validation import backtest, daily_origins


@pytest.fixture
def frame():
    timestamps = pd.date_range("2025-10-01", periods=21 * 48, freq="30min")
    rows = []
    for index, timestamp in enumerate(timestamps):
        for route in [19, 47]:
            row = {"route_id": route, "timestamp": timestamp, "target_1h": 100 + route + index % 48}
            row.update({f"status_{k}": route + k * (index % 7) for k in range(1, 7)})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("predicted,expected", [([10, 20, 0], 0), ([0, 0, 0], 2), ([20, 40, 0], 2)])
def test_metric_exact_cases(predicted, expected):
    assert evaluate(np.array([10, 20, 0]), np.array(predicted))["score"] == pytest.approx(expected)


def test_metric_uses_global_sum_and_keeps_signed_bias():
    truth = np.array([[10, 20], [0, 70]])
    prediction = np.array([[12, 16], [2, 60]])
    result = evaluate(truth, prediction)
    assert result["wape"] == pytest.approx(0.18)
    assert result["signed_bias"] == pytest.approx(-0.10)
    assert result["score"] == pytest.approx(0.28)
    assert evaluate(truth * 10, prediction * 10)["score"] == pytest.approx(0.28)


@pytest.mark.parametrize("truth,prediction", [([], []), ([0], [1]), ([1], [np.nan]), ([np.inf], [1]), ([-1], [1]), ([1], [-1]), ([1, 2], [1])])
def test_metric_rejects_invalid_comparisons(truth, prediction):
    with pytest.raises(ValueError):
        evaluate(np.asarray(truth), np.asarray(prediction))


def test_shuffled_rows_keep_route_time_alignment(frame):
    data = ForecastData.from_frame(frame.sample(frac=1, random_state=4))
    assert data.routes.tolist() == [19, 47]
    np.testing.assert_array_equal(data.target[0], [119, 147])
    np.testing.assert_array_equal(data.target[1], [120, 148])
    assert data.timestamps[0] == pd.Timestamp("2025-10-01")


@pytest.mark.parametrize("damage", ["duplicate", "missing_row", "gap", "nan", "negative", "missing_column", "empty"])
def test_data_contract_rejects_bad_inputs(frame, damage):
    if damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif damage == "missing_row":
        frame = frame.iloc[1:]
    elif damage == "gap":
        frame = frame[frame["timestamp"] != frame["timestamp"].iloc[0]]
        frame.loc[frame["timestamp"] == frame["timestamp"].max(), "timestamp"] += pd.Timedelta(minutes=30)
    elif damage == "nan":
        frame.loc[0, "status_1"] = np.nan
    elif damage == "negative":
        frame.loc[0, "target_1h"] = -1
    elif damage == "missing_column":
        frame = frame.drop(columns="status_1")
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        ForecastData.from_frame(frame)


def test_history_excludes_every_observation_after_origin(frame):
    data = ForecastData.from_frame(frame)
    history = data.history(pd.Timestamp("2025-10-15 10:30"))
    assert history.timestamps[-1] == pd.Timestamp("2025-10-15 10:30")
    assert len(history.target) == len(history.statuses) == len(history.timestamps)
    with pytest.raises(ValueError, match="origin"):
        data.history(pd.Timestamp("2025-10-15 10:15"))


def test_simple_forecasts_use_only_correct_historical_points(frame):
    data = ForecastData.from_frame(frame)
    history = data.history(pd.Timestamp("2025-10-15 10:30"))
    last = predict_simple(history, "last_value")
    np.testing.assert_array_equal(last, np.repeat(history.target[-1, :, None], 8, axis=1))
    np.testing.assert_allclose(predict_simple(history, "mean_last_3h")[:, 0], history.target[-6:].mean(axis=0))
    np.testing.assert_array_equal(predict_simple(history, "seasonal_day"), history.target[-48:-40].T)
    np.testing.assert_array_equal(predict_simple(history, "seasonal_week"), history.target[-336:-328].T)
    with pytest.raises(ValueError, match="Unknown"):
        predict_simple(history, "invented")
    with pytest.raises(ValueError, match="history"):
        predict_simple(data.history(data.timestamps[3]), "seasonal_week")


def test_training_labels_end_by_forecast_origin(frame):
    data = ForecastData.from_frame(frame)
    history = data.history(pd.Timestamp("2025-10-15 10:30"))
    x, y, audit = training_samples(history, train_days=1)
    assert audit["latest_label_timestamp"] == "2025-10-15 10:30:00"
    assert audit["last_source_timestamp"] == "2025-10-15 06:30:00"
    assert audit["training_rows"] == 49 * 2
    np.testing.assert_array_equal(y[-2:], history.target[-8:].T)
    assert x["route_id"].tolist()[-2:] == [19, 47]
    with pytest.raises(ValueError, match="history"):
        training_samples(data.history(data.timestamps[3]), train_days=1)


def test_future_poison_cannot_change_any_prediction(frame):
    origin = pd.Timestamp("2025-10-15 10:30")
    reference = ForecastData.from_frame(frame).history(origin)
    poisoned = frame.copy()
    future = poisoned["timestamp"] > origin
    poisoned.loc[future, ["target_1h", *[f"status_{k}" for k in range(1, 7)]]] = 10**12
    hidden = ForecastData.from_frame(poisoned).history(origin)
    for name in ["last_value", "mean_last_3h", "seasonal_day", "seasonal_week"]:
        np.testing.assert_array_equal(predict_simple(reference, name), predict_simple(hidden, name))
    first, audit = predict_ridge(reference, train_days=1)
    second, _ = predict_ridge(hidden, train_days=1)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    assert np.isfinite(first).all() and (first >= 0).all()
    assert audit["latest_label_timestamp"] == str(origin)


def test_daily_origins_leave_complete_horizon(frame):
    data = ForecastData.from_frame(frame)
    origins = daily_origins(data, count=3)
    assert origins == [pd.Timestamp(f"2025-10-{day} 10:30") for day in [19, 20, 21]]
    assert all(origin + pd.Timedelta(hours=4) <= data.timestamps[-1] for origin in origins)
    with pytest.raises(ValueError):
        daily_origins(data, count=0)
    with pytest.raises(ValueError, match="outside"):
        daily_origins(data, count=99)


def test_daily_origins_shifts_back_when_final_day_is_incomplete(frame):
    frame = frame[frame["timestamp"] <= pd.Timestamp("2025-10-21 10:30")]
    data = ForecastData.from_frame(frame)
    assert daily_origins(data, count=1) == [pd.Timestamp("2025-10-20 10:30")]


@pytest.mark.parametrize("with_progress", [False, True])
def test_backtest_evaluates_same_points_and_reports_training_boundary(frame, with_progress):
    data = ForecastData.from_frame(frame)
    origins = daily_origins(data, count=2)
    progress = []
    result = backtest(data, origins, train_days=1, progress=progress.append if with_progress else None)
    assert len(progress) == (4 if with_progress else 0)
    assert len(result["folds"]) == 2 * 5
    assert len(result["by_horizon"]) == 2 * 5 * 8
    assert len(result["summary"]) == 5
    assert set(result["folds"]["points"]) == {16}
    assert len(result["predictions"]) == 2 * 2 * 8
    assert len(result["audits"]) == 2
    for audit in result["audits"]:
        assert pd.Timestamp(audit["latest_label_timestamp"]) <= pd.Timestamp(audit["origin"])
    assert result["summary"].iloc[0]["mean_score"] == pytest.approx(0)
    with pytest.raises(ValueError):
        backtest(data, [data.timestamps[-1]], train_days=1)
    for invalid in [[], [origins[0], origins[0]]]:
        with pytest.raises(ValueError, match="unique"):
            backtest(data, invalid, train_days=1)
