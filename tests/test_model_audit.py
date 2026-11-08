import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.model_audit import check_chronology, check_truth, choose_scale, per_origin, summarize, available_calibration_mask


def test_daily_bias_must_not_cancel_between_days():
    truth = np.ones((2, 1, 1)) * 10
    pred = np.array([5, 15]).reshape(2, 1, 1)
    result = summarize(pred, truth)
    assert result["mean_score"] == pytest.approx(1.0)
    assert result["pooled_score"] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_forecasts_are_rejected(value):
    with pytest.raises(ValueError):
        per_origin(np.full((1, 1, 1), value), np.ones((1, 1, 1)))


def test_zero_day_cannot_hide_behind_positive_total():
    with pytest.raises(ValueError, match="Each origin"):
        summarize(np.ones((2, 1, 1)), np.array([0, 10]).reshape(2, 1, 1))


def test_calibration_uses_earlier_week_and_is_frozen_for_evaluation():
    cal_pred = np.ones((2, 1, 1)) * 10
    scale = choose_scale(cal_pred, cal_pred * 1.1, np.array([0.9, 1.0, 1.1]))
    assert scale == pytest.approx(1.1)
    # Evaluation favours 0.9, but cannot change the already selected 1.1.
    assert summarize(cal_pred * scale, cal_pred * 0.9)["mean_score"] > 0
    check_chronology(np.array([1, 2]), np.array([3, 4]))
    with pytest.raises(ValueError):
        check_chronology(np.array([3, 4]), np.array([1, 2]))
    with pytest.raises(ValueError):
        check_chronology(np.array([1, 3]), np.array([3, 4]))


def test_matching_shape_does_not_prove_matching_route_order():
    actual = np.array([10, 20]).reshape(1, 2, 1)
    with pytest.raises(ValueError, match="raw data ordering"):
        check_truth(actual[:, ::-1], actual, "reordered archive")


def test_score_is_weighted_by_volume_not_mean_of_route_percentages():
    truth = np.array([1, 99]).reshape(1, 2, 1)
    pred = np.array([2, 98]).reshape(1, 2, 1)
    assert per_origin(pred, truth)["score"][0] == pytest.approx(0.02)


def test_calibration_origin_before_evaluation_is_not_enough_for_future_labels():
    # The origin 8 is earlier than 9, but its second target at 10 is still unknown.
    np.testing.assert_array_equal(available_calibration_mask(np.array([5, 6, 7, 8]), 9, 2),
                                  [True, True, True, False])
    with pytest.raises(ValueError):
        available_calibration_mask(np.array([8]), 9, 2)
