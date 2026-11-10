import numpy as np
import pytest

from rwb_forecasting.latent_volume import infer_half_hour, hour_from_half_predictions, project_hourly_forecasts, project_hourly_interval


def test_zero_resets_identify_latent_sequence_exactly():
    latent = np.array([[3, 4], [0, 2], [0, 0], [5, 0], [7, 6], [1, 9]], dtype=float)
    target = latent + np.vstack([np.zeros(2), latent[:-1]])
    inferred, width = infer_half_hour(target, rounding_tolerance=0)
    np.testing.assert_array_equal(inferred[3:], latent[3:])
    np.testing.assert_array_equal(width[3:], np.zeros_like(width[3:]))


def test_reconstruction_is_prefix_causal_without_backward_revision():
    target = np.array([[12, 9], [7, 5], [10, 1], [5, 6], [9, 3]], dtype=float)
    prefix, prefix_width = infer_half_hour(target[:3])
    altered = target.copy()
    altered[3:] = 1e12
    values, widths = infer_half_hour(altered)
    np.testing.assert_array_equal(prefix, values[:3])
    np.testing.assert_array_equal(prefix_width, widths[:3])


def test_ambiguous_initial_intervals_remain_explicit():
    values, widths = infer_half_hour(np.array([[10.], [10.], [10.]]), rounding_tolerance=0)
    np.testing.assert_array_equal(values, np.full((3, 1), 5.))
    np.testing.assert_array_equal(widths, np.full((3, 1), 10.))


def test_hour_forecast_uses_current_known_half_only_once():
    actual = hour_from_half_predictions(np.array([8, 2]), np.array([[1, 4, 5], [3, 6, 7]]))
    np.testing.assert_array_equal(actual, [[9, 5, 9], [5, 9, 13]])


def test_projection_preserves_feasible_profiles_and_handles_zeroes():
    current = np.array([8., 2., 0])
    future = np.array([[1, 4, 5, 0], [3, 0, 7, 1], [0, 0, 0, 0]], dtype=float)
    hourly = hour_from_half_predictions(current, future)
    np.testing.assert_allclose(project_hourly_forecasts(current, hourly), hourly, rtol=1e-12, atol=1e-12)


def test_projection_anchors_known_volume_and_yields_nonnegative_halves():
    current = np.array([8.])
    hourly = project_hourly_forecasts(current, np.array([[1., 2., 1., 2.]]))
    assert hourly[0, 0] >= current[0]
    previous = current[0]
    for value in hourly[0]:
        latent = value - previous
        assert latent >= -1e-12
        previous = latent


@pytest.mark.parametrize("bad", [np.array([]), np.array([1, 2]), np.array([[-1.]]), np.array([[np.nan]])])
def test_reconstruction_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        infer_half_hour(bad)


def test_rounding_uncertainty_is_preserved_and_conflicts_raise():
    _, widths = infer_half_hour(np.array([[0.], [1.], [0.]]))
    assert widths[0, 0] == .5
    with pytest.raises(ValueError, match="No feasible"):
        infer_half_hour(np.array([[0.], [1.], [0.]]), rounding_tolerance=0)
    with pytest.raises(ValueError, match="No feasible"):
        infer_half_hour(np.array([[0.], [100.], [0.]]))


def test_interval_projection_keeps_any_feasible_initial_half_not_only_midpoint():
    lower, upper = np.array([1., 8.]), np.array([5., 8.])
    hourly = hour_from_half_predictions(np.array([4.7, 8.]), np.array([[0., 2., 1.], [2., 1., 0.]]))
    np.testing.assert_allclose(project_hourly_interval(lower, upper, hourly), hourly, rtol=1e-12, atol=1e-12)


def test_interval_projection_corrects_impossible_origin_volume():
    lower, upper = np.array([10.]), np.array([11.])
    result = project_hourly_interval(lower, upper, np.array([[1., 2., 1.]]))
    assert result[0, 0] >= 10
    assert np.isfinite(result).all() and (result >= 0).all()
