import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwb.flows import legacy_cumulative_inflow, reconcile_hourly


def test_known_half_hour_path_is_recovered_without_changing_input():
    rng = np.random.default_rng(12)
    path = rng.uniform(0, 100, (3, 5, 9))
    hourly = path[..., :-1] + path[..., 1:]
    before = hourly.copy()
    result = reconcile_hourly(hourly, path[..., 0])
    np.testing.assert_allclose(result.half_hour, path[..., 1:], atol=1e-10)
    np.testing.assert_array_equal(hourly, before)
    np.testing.assert_allclose(result.cumulative, np.cumsum(path[..., 1:], axis=-1))


def test_legacy_can_decrease_but_reconciliation_is_physical():
    pred, x0 = np.array([[10., 0., 0., 0., 0., 0., 0., 0.]]), np.array([0.])
    assert (np.diff(legacy_cumulative_inflow(pred, x0)) < 0).any()
    result = reconcile_hourly(pred, x0)
    assert (result.half_hour >= 0).all()
    assert (np.diff(result.cumulative) >= 0).all()
    assert np.any(result.hourly_adjustment != 0)


def test_impossible_first_hour_reports_residual_instead_of_negative_flow():
    result = reconcile_hourly(np.array([[0., 2.]]), np.array([3.]))
    np.testing.assert_allclose(result.half_hour, [[0., 2.]])
    np.testing.assert_allclose(result.hourly_adjustment, [[3., 0.]])


def test_solution_satisfies_constrained_least_squares_optimality():
    rng = np.random.default_rng(4)
    pred, x0 = rng.uniform(0, 10, (60, 8)), rng.uniform(0, 20, 60)
    result = reconcile_hourly(pred, x0)
    design = np.eye(8) + np.eye(8, k=-1)
    gradient = result.hourly_adjustment @ design
    assert (gradient >= -1e-8).all()
    np.testing.assert_allclose(gradient * result.half_hour, 0, atol=1e-7)


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_predictions_rejected(value):
    with pytest.raises(ValueError):
        reconcile_hourly(np.full((1, 8), value), np.array([1.]))


def test_zero_flow():
    assert np.array_equal(reconcile_hourly(np.zeros((2, 8)), np.zeros(2)).cumulative, np.zeros((2, 8)))
