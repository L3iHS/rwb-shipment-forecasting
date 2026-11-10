import numpy as np
import pandas as pd
import pytest

from rwb_forecasting import candidates, latent_models
from rwb_forecasting.data import ForecastData


@pytest.fixture
def setup(monkeypatch):
    calls = []
    data = ForecastData(pd.date_range("2025-10-01", periods=2, freq="30min"),
                        np.array([3, 7]), np.ones((2, 2, 6)), np.array([[10., 20.], [12., 22.]]))

    def base(history, **kwargs):
        calls.append(history)
        return np.repeat(history.target[-1, :, None], 8, axis=1), {
            "latest_label_timestamp": str(history.timestamps[-1]), "nested": {"value": 17}}

    def latent(history, *args, **kwargs):
        return np.full((2, 8), 42.), {"latest_label_timestamp": str(history.timestamps[-1])}

    monkeypatch.setattr(candidates, "predict_ridge", base)
    monkeypatch.setattr(latent_models, "predict_latent_ridge", latent)
    spec = {"kind": "ridge", "params": {}}
    models = candidates.configured_comparison({
        "base": spec, "hybrid": {"kind": "latent_hybrid", "params": {}, "incumbent": spec}})
    return data, calls, models


@pytest.mark.parametrize("order", [("base", "hybrid"), ("hybrid", "base")])
def test_shared_fit_runs_once_regardless_of_comparison_order(setup, order):
    data, calls, models = setup
    results = {name: models[name](data)[0] for name in order}
    assert len(calls) == 1
    np.testing.assert_array_equal(results["hybrid"][:, 0], [42., 42.])
    np.testing.assert_array_equal(results["hybrid"][:, 1:], results["base"][:, 1:])


def test_cache_does_not_cross_datasets_with_the_same_origin(setup):
    data, calls, models = setup
    first, _ = models["base"](data)
    changed = ForecastData(data.timestamps, data.routes, data.statuses, data.target + 100.)
    second, _ = models["base"](changed)
    assert len(calls) == 2
    np.testing.assert_array_equal(second, first + 100.)
    models["base"](data.history(data.timestamps[0]))
    assert len(calls) == 3


def test_callers_cannot_mutate_cached_predictions_or_nested_audits(setup):
    data, calls, models = setup
    prediction, audit = models["base"](data)
    prediction[:] = -999.
    audit["nested"]["value"] = -999
    again, audit_again = models["base"](data)
    assert len(calls) == 1 and (again > 0).all()
    assert audit_again["nested"]["value"] == 17


def test_mass_blend_dispatch_and_shared_components(setup):
    data, calls, _ = setup
    base = {"kind": "ridge", "params": {}}
    alternative = {"kind": "latent", "params": {}}
    blend = {"kind": "mass_preserving", "params": {"weight": 0.5},
             "incumbent": base, "alternative": alternative}
    models = candidates.configured_comparison({"blend": blend, "base": base,
                                               "alternative": alternative})
    blended, audit = models["blend"](data)
    original, _ = models["base"](data)
    assert len(calls) == 1
    np.testing.assert_array_equal(blended[:, 0], original[:, 0])
    np.testing.assert_allclose(blended.sum(axis=0), original.sum(axis=0))
    standalone, standalone_audit = candidates.configured_predictor(blend)(data)
    np.testing.assert_array_equal(standalone, blended)
    assert standalone_audit == audit
