import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.dispatch import inflow_actual, inflow_forecast, simulate
from rwb.intervals import Conformal, coverage


def test_inflow_forecast_is_exact_for_steady_flow():
    c, k = 5.0, 3
    pred = np.full((1, 2, 8), 2 * c)  # y = x[t] + x[t-1] = 2c
    x0 = np.full((1, 2), c)
    assert np.allclose(inflow_forecast(pred, x0, k), c * k)


def test_inflow_actual_sums_the_next_steps():
    x = np.arange(20, dtype=float)[None, :]
    assert inflow_actual(x, np.array([4]), 3)[0, 0] == 5 + 6 + 7


def _steady(R=3, T=200, c=1.0):
    return np.full((R, T), c), np.arange(10, 190)


def test_reactive_trucks_are_late_by_the_lead_time():
    x, origins = _steady()
    cap = np.full(3, 4.0)
    r = simulate(x, origins, np.zeros((len(origins), 3)), cap, lead=2, max_wait=4)
    # A load is ready every 4 steps, the truck is called then and comes 2 steps (1 hour) later.
    assert r.trucks > 0 and r.idle_h == 0
    assert np.isclose(r.late_h / r.trucks, 1.0)


def test_perfect_foresight_removes_waiting():
    x, origins = _steady()
    cap = np.full(3, 4.0)
    oracle = inflow_actual(x, origins, 2)
    r = simulate(x, origins, oracle, cap, lead=2, max_wait=4)
    assert r.late_h == 0 and r.idle_h == 0 and r.partial == 0
    assert np.isclose(r.fill, 1.0)


def test_overlapping_trucks_cover_a_long_lead_time():
    # A load fills in 2 steps but a truck needs 4: two trucks must be on the way at once.
    x, origins = _steady()
    cap = np.full(3, 2.0)
    oracle = inflow_actual(x, origins, 4)
    r = simulate(x, origins, oracle, cap, lead=4, max_wait=4)
    assert r.trucks > 50 and r.idle_h == 0 and r.partial == 0
    # Only the warm-up loads, complete before the first trucks could arrive, wait: a run twice
    # as long has the same waiting.
    half = simulate(x, origins[:90], oracle[:90], cap, lead=4, max_wait=4)
    assert r.late_h == half.late_h > 0


def test_call_plan_counts_trucks_already_coming():
    from rwb.dispatch import call_plan

    inflow = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8]] * 2)
    cap = np.array([4.0, 4.0])
    # Empty dock: a load is complete after 4 steps, so with lead 2 call in 2 steps.
    plan = call_plan(np.zeros(2), inflow, cap, lead=2, inbound=np.array([0, 1]))
    assert plan[0] == 2
    # One truck already coming covers that load; the next is due after 8 steps -> call in 6.
    assert plan[1] == 6


def test_conformal_band_covers_the_nominal_share():
    rng = np.random.default_rng(0)
    level = np.repeat([1.0, 10.0, 100.0, 1000.0, 10000.0], 40)
    mean = np.broadcast_to(level[None, :, None], (300, 200, 8))

    def draw():
        return mean * rng.lognormal(0, 0.3, mean.shape)

    conf = Conformal().fit(np.asarray(mean), draw(), level)
    lo, hi = conf.quantile(np.asarray(mean), level, 0.1), conf.quantile(np.asarray(mean), level, 0.9)
    assert abs(coverage(lo, hi, draw()) - 0.8) < 0.01


def test_forecaster_survives_save_and_load(tmp_path):
    from dataclasses import replace

    import pandas as pd

    from rwb.data import Panel
    from rwb.features import DAY, MAX_LOOKBACK, cumsums
    from rwb.forecaster import LEAN, Forecaster

    rng = np.random.default_rng(1)
    R, T = 6, MAX_LOOKBACK + 3 * DAY
    panel = Panel(pd.date_range("2025-07-28", periods=T, freq="30min"), np.arange(R),
                  rng.gamma(2.0, 100.0, size=(R, T)).astype(np.float32),
                  rng.gamma(2.0, 50.0, size=(6, R, T)).astype(np.float32), pd.DataFrame())
    cums = cumsums(panel)
    cut = T - 10
    fc = Forecaster(replace(LEAN, rounds=5, min_child=5)).fit(panel, cums, cut)
    origins = np.array([cut, cut + 1])
    before = fc.predict(panel, cums, origins)
    fc.save(tmp_path / "bundle")
    after = Forecaster.load(tmp_path / "bundle").predict(panel, cums, origins)
    assert before.shape == (2, R, 8) and np.isfinite(before).all()
    np.testing.assert_allclose(after, before, rtol=1e-6)
    from rwb.service import Service, ServiceConfig

    service = Service.calibrate(fc, cums, panel, origins[:1], before[:1], ServiceConfig())
    service.save(tmp_path / "service_v2")
    loaded = Service.load(tmp_path / "service_v2")
    np.testing.assert_array_equal(loaded.forecast(panel, cums, cut)["point_forecast"], before[0])
    assert loaded.calibration_metadata == service.calibration_metadata
