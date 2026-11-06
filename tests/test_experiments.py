import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData
from rwb_forecasting.experiments import compare_models, validate_training_boundary


@pytest.fixture
def panel():
    times = pd.date_range("2025-10-01", periods=30, freq="30min")
    return ForecastData(times, np.array([19, 47]), np.ones((30, 2, 6)),
                        np.repeat(np.arange(1, 31)[:, None], 2, axis=1).astype(float))


def test_fixed_origin_predictor_has_no_access_to_future_and_scores_exact_keys(panel):
    origins = list(panel.timestamps[[15, 20]])
    seen = []
    def predictor(history):
        seen.append(history.timestamps[-1])
        assert len(history.target) == len(history.timestamps)
        pred = np.repeat(history.target[-1, :, None], 8, axis=1)
        return pred, {"latest_label_timestamp": str(history.timestamps[-1])}
    result = compare_models(panel, origins, {"last": predictor})
    assert seen == origins
    frame = result["predictions"]
    assert len(frame) == 32
    assert not frame.duplicated(["origin", "route_id", "timestamp"]).any()
    first = result["folds"].iloc[0]
    assert first["score"] == pytest.approx(2 * 36 / sum(range(17, 25)))
    assert frame.iloc[0]["timestamp"] == panel.timestamps[16]
    assert result["summary"].iloc[0]["mean_score"] == result["folds"]["score"].mean()


@pytest.mark.parametrize("damage", ["shape", "future_label", "negative"])
def test_invalid_prediction_or_training_boundary_is_rejected(panel, damage):
    def predictor(history):
        shape = (2, 7) if damage == "shape" else (2, 8)
        pred = np.full(shape, -1.0 if damage == "negative" else 1.0)
        label = panel.timestamps[-1] if damage == "future_label" else history.timestamps[-1]
        return pred, {"latest_label_timestamp": str(label)}
    with pytest.raises(ValueError):
        compare_models(panel, [panel.timestamps[15]], {"test": predictor})


@pytest.mark.parametrize("indices,names", [([], ["a"]), ([15, 15], ["a"]), ([25], ["a"]),
                                          ([15], []), ([15], ["y_true"]), ([15], [""])])
def test_invalid_comparison_fails_before_fitting(panel, indices, names):
    def predictor(history):
        pytest.fail("Invalid comparison should fail before fitting")
    with pytest.raises(ValueError):
        compare_models(panel, list(panel.timestamps[indices]), {name: predictor for name in names})


def test_unknown_declared_label_timestamp_is_not_treated_as_safe(panel):
    with pytest.raises(ValueError, match="invalid timestamp"):
        validate_training_boundary({"aggregate_model": {"latest_label_timestamp": None}}, panel.timestamps[-1])
