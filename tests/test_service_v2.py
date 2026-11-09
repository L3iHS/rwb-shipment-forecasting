from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwb.data import Panel
from rwb.features import MAX_LOOKBACK, cumsums
from rwb.intervals import Conformal
from rwb.monitor import DataUnavailableError
from rwb.service import Service, ServiceConfig, forecast_table


def example():
    n = MAX_LOOKBACK + 100
    panel = Panel(pd.date_range("2025-01-01", periods=n, freq="30min"), np.arange(5),
                  np.full((5, n), 20., dtype=np.float32), np.ones((6, 5, n), dtype=np.float32), pd.DataFrame())
    fake = SimpleNamespace(label_cutoff=0, predict=lambda panel, cums, origins: np.full((len(origins), 5, 8), 20.))
    table = np.broadcast_to(np.array([-0.4, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]), (8, 5, 7)).copy()
    conf = Conformal(table=table, edges=np.array([2., 4., 6., 8.]))
    svc = Service(fake, conf, conf, np.arange(1, 6, dtype=float), ServiceConfig(), panel.routes, conf)
    return svc, panel


def test_point_remains_raw_and_p50_is_calibrated_median():
    svc, panel = example()
    fc = svc.forecast(panel, cumsums(panel), panel.n_times - 1)
    np.testing.assert_array_equal(fc["point_forecast"], np.full((5, 8), 20.))
    np.testing.assert_allclose(fc["p50"], np.full((5, 8), 23.1))
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()
    assert (np.diff(fc["inflow_point"], axis=-1) >= 0).all()
    assert (np.diff(fc["inflow_p50"], axis=-1) >= 0).all()
    table = forecast_table(panel, panel.n_times - 1, fc)
    assert "point_forecast" in table and len(table) == 40


def test_status_gap_uses_fallback_without_claiming_calibrated_intervals():
    svc, panel = example()
    panel.s[0, 0, -1] = np.nan
    fc = svc.forecast(panel, cumsums(panel), panel.n_times - 1)
    assert fc["stale"].tolist() == [True, False, False, False, False]
    assert np.isfinite(fc["point_forecast"]).all()
    assert np.isnan(fc["p50"][0]).all()
    assert not fc["intervals_valid"][0]


def test_missing_target_cannot_be_silently_zero_filled():
    svc, panel = example()
    panel.y[0, -1] = np.nan
    with pytest.raises(DataUnavailableError, match="fallback"):
        svc.forecast(panel, cumsums(panel), panel.n_times - 1)


def test_unknown_routes_and_earlier_origins_rejected():
    svc, panel = example()
    svc.routes = svc.routes + 100
    with pytest.raises(DataUnavailableError, match="Routes"):
        svc.forecast(panel, cumsums(panel), panel.n_times - 1)
    svc.routes = panel.routes
    with pytest.raises(DataUnavailableError, match="history"):
        svc.forecast(panel, cumsums(panel), 10)
    svc.forecaster.label_cutoff = panel.n_times
    with pytest.raises(DataUnavailableError, match="trained after"):
        svc.forecast(panel, cumsums(panel), panel.n_times - 1)


def test_empty_size_groups_and_network_calibration_work():
    pred = np.ones((10, 1, 8))
    conf = Conformal().fit(pred, pred * 2, np.array([1.]))
    np.testing.assert_allclose(conf.quantile(pred, np.array([1.]), .5), pred * 2)


def test_missing_calibration_truth_rejected():
    with pytest.raises(ValueError, match="finite"):
        Conformal().fit(np.ones((2, 1, 8)), np.full((2, 1, 8), np.nan), np.array([1.]))


def test_legacy_bundle_requires_explicit_recalibration(tmp_path):
    (tmp_path / "service.json").write_text('{}')
    with pytest.raises(ValueError, match="prepare_service_v2"):
        Service.load(tmp_path)


def test_all_routes_can_use_fallback_without_calling_model():
    svc, panel = example()
    panel.s[:, :, -1] = np.nan
    svc.forecaster.predict = lambda *args: pytest.fail("model must not receive unusable input")
    fc = svc.forecast(panel, cumsums(panel), panel.n_times - 1)
    assert fc["stale"].all() and np.isfinite(fc["point_forecast"]).all()
