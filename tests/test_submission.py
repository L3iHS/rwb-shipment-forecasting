import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData, HORIZON, STEP
from rwb_forecasting.submission import make_submission


@pytest.fixture
def submission_inputs():
    times = pd.date_range("2025-10-01", periods=10 * 48, freq="30min")
    index = np.repeat(np.arange(len(times)), 2)
    routes = np.tile([19, 47], len(times))
    frame = pd.DataFrame({
        "timestamp": np.repeat(times, 2), "route_id": routes,
        "target_1h": 100 + index % 48 + routes,
        **{f"status_{k}": (index + k) % 17 + routes for k in range(1, 7)},
    })
    data = ForecastData.from_frame(frame)
    future = pd.date_range(times[-1] + STEP, periods=HORIZON, freq=STEP)
    test = pd.DataFrame({
        "id": np.arange(100, 116),
        "route_id": np.repeat(data.routes, HORIZON),
        "timestamp": np.tile(future, len(data.routes)),
    })
    return data, test


def test_submission_maps_by_route_and_time_instead_of_input_row_position(submission_inputs, monkeypatch):
    data, test = submission_inputs
    prediction = np.array([np.arange(101, 109), np.arange(201, 209)])
    monkeypatch.setattr("rwb_forecasting.submission.predict_ridge", lambda *args, **kwargs: (prediction, {}))
    output, audit = make_submission(data, test.sample(frac=1, random_state=2))
    assert output["id"].tolist() == list(range(100, 116))
    assert output["y_pred"].tolist() == prediction.reshape(-1).tolist()
    assert audit["origin"] == str(data.timestamps[-1])


@pytest.mark.parametrize("damage", ["empty", "missing_column", "duplicate_id", "missing_row", "duplicate_pair", "unknown_route", "wrong_time", "null"])
def test_invalid_test_contract_fails_before_training(submission_inputs, monkeypatch, damage):
    data, test = submission_inputs
    if damage == "empty":
        test = test.iloc[:0]
    elif damage == "missing_column":
        test = test.drop(columns="id")
    elif damage == "duplicate_id":
        test.loc[0, "id"] = test.loc[1, "id"]
    elif damage == "missing_row":
        test = test.iloc[1:]
    elif damage == "duplicate_pair":
        test.loc[0, ["route_id", "timestamp"]] = test.loc[1, ["route_id", "timestamp"]]
    elif damage == "unknown_route":
        test.loc[0, "route_id"] = 9999
    elif damage == "wrong_time":
        test.loc[0, "timestamp"] = data.timestamps[-1]
    else:
        test.loc[0, "id"] = np.nan
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid test input must fail before training")
    monkeypatch.setattr("rwb_forecasting.submission.predict_ridge", unexpected_fit)
    with pytest.raises(ValueError):
        make_submission(data, test)


@pytest.mark.parametrize("prediction", [np.full((2, 8), np.nan), np.full((2, 8), -1), np.ones((2, 7))])
def test_invalid_model_output_cannot_be_exported(submission_inputs, monkeypatch, prediction):
    data, test = submission_inputs
    monkeypatch.setattr("rwb_forecasting.submission.predict_ridge", lambda *args, **kwargs: (prediction, {}))
    with pytest.raises(ValueError, match="prediction"):
        make_submission(data, test)


def test_real_ridge_submission_has_exact_csv_contract(submission_inputs, tmp_path):
    data, test = submission_inputs
    output, audit = make_submission(data, test, train_days=1, feature_set="calendar")
    path = tmp_path / "submission.csv"
    output.to_csv(path, index=False)
    restored = pd.read_csv(path)
    assert list(restored.columns) == ["id", "y_pred"]
    assert len(restored) == len(test)
    assert restored["id"].is_unique
    assert set(restored["id"]) == set(test["id"])
    assert np.isfinite(restored["y_pred"]).all() and restored["y_pred"].ge(0).all()
    assert pd.Timestamp(audit["latest_label_timestamp"]) == data.timestamps[-1]


def test_custom_predictor_receives_only_train_and_preserves_key_alignment(submission_inputs):
    data, test = submission_inputs
    def predictor(history):
        assert history is data
        return np.repeat(history.target[-1, :, None], HORIZON, axis=1), {"model": "last_value"}
    output, audit = make_submission(data, test.sample(frac=1, random_state=19), predictor=predictor)
    np.testing.assert_array_equal(output["y_pred"], np.repeat(data.target[-1], HORIZON))
    assert audit["model"] == "last_value"


def test_invalid_test_never_calls_custom_predictor(submission_inputs):
    data, test = submission_inputs
    def predictor(history):
        pytest.fail("Invalid test must be rejected before fitting any predictor")
    with pytest.raises(ValueError):
        make_submission(data, test.iloc[1:], predictor=predictor)


@pytest.mark.parametrize("nested", [False, True])
def test_submission_rejects_training_labels_after_origin(submission_inputs, nested):
    data, test = submission_inputs
    def predictor(history):
        audit = {"latest_label_timestamp": str(history.timestamps[-1] + STEP)}
        return np.ones((2, HORIZON)), {"aggregate_model": audit} if nested else audit
    with pytest.raises(ValueError, match="labels exceed"):
        make_submission(data, test, predictor=predictor)
