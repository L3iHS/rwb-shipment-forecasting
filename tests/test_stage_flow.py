"""Generated-data invariants for the fixed stage-flow module and export CLI."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.stage_flow import StageFlowPanel, _fit_pair, load_stage_flow_panel, predict_stage_flow


@pytest.fixture(scope="module")
def arrays():
    rng = np.random.default_rng(1729)
    times = pd.date_range("2025-07-28", periods=25 * 48, freq="30min")
    latent = rng.uniform(800, 2200, (len(times) + 1, 4))
    hourly = latent[:-1] + latent[1:]
    status = rng.uniform(100, 2500, (len(times), 4, 3))
    return times, hourly, status, np.arange(4)


def _panel(arrays, end=None):
    times, hourly, status, routes = arrays
    return StageFlowPanel(times[:end], hourly[:end], status[:end], routes)


def _cli():
    spec = importlib.util.spec_from_file_location("stage_flow_cli", Path(__file__).parents[1] / "scripts/apply_stage_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_prefix_and_future_mutation(arrays):
    full = _panel(arrays)
    cut = 20 * 48 + 21
    prefix = _panel(arrays, cut + 1)
    times, hourly, status, routes = arrays
    changed_y = hourly.copy(); changed_y[cut + 1:] *= 1.01
    changed_s = status.copy(); changed_s[cut + 1:] = 999999
    changed = StageFlowPanel(times, changed_y, changed_s, routes)
    sources = np.array([336, 450, cut])
    for panel in (prefix, changed):
        x, scale = panel.features(sources)
        expected, expected_scale = full.features(sources)
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_array_equal(scale, expected_scale)
        np.testing.assert_array_equal(panel.half_hour[:cut + 1], full.half_hour[:cut + 1])


def test_feature_order_and_finite_47_flow_identity(arrays):
    panel = _panel(arrays)
    t = np.array([400, 650])
    features, scale = panel.features(t)
    assert features.shape == (2, 4, 79)
    np.testing.assert_array_equal(features[:, :, 0], panel.hourly[t] / scale)
    np.testing.assert_array_equal(features[:, :, 8], panel.half_hour[t] / scale)
    unit = np.maximum((panel.c_status[t + 1] - panel.c_status[t + 1 - 336]) / 336, 1)
    np.testing.assert_array_equal(features[:, :, 16:19], panel.statuses[t] / unit)
    flow = panel.statuses[:, :, :2] - panel.statuses[:, :, 1:]
    for source in t:
        direct = sum((47 - lag) * flow[source - lag] for lag in range(47)) / 48
        np.testing.assert_allclose(panel.excess[source], direct, atol=1e-8, rtol=1e-11)
    np.testing.assert_array_equal(features[:, :, 73:75], panel.excess[t] / unit[:, :, :2])
    np.testing.assert_array_equal(features[:, :, 75:77], panel.excess[t - 1] / unit[:, :, :2])
    np.testing.assert_array_equal(features[:, :, 77:79],
                                  (panel.c_excess[t + 1] - panel.c_excess[t - 5]) / (6 * unit[:, :, :2]))


def test_label_maturity_including_h1_anchor(arrays):
    panel = _panel(arrays)
    cutoff = 1000
    mature = np.array([cutoff - 9, cutoff - 8])
    _, scale = panel.features(mature)
    labels = panel.labels(mature, scale, cutoff)
    np.testing.assert_array_equal(labels[:, :, 0],
                                  (panel.hourly[mature + 1] - panel.half_hour[mature]) / scale)
    late = np.array([cutoff - 8, cutoff - 7])
    with pytest.raises(ValueError, match="cutoff"):
        panel.labels(late, scale, cutoff)
    with pytest.raises(ValueError, match="cutoff"):
        _fit_pair(panel, late, cutoff)


def test_prediction_totals_floor_and_source_cutoff(arrays):
    panel = _panel(arrays, 24 * 48 + 22)
    anchor = np.full((4, 8), 3000.)
    anchor[:, 0] += panel.half_hour[-1]
    prediction, audit = predict_stage_flow(panel, anchor)
    assert prediction.shape == anchor.shape and np.isfinite(prediction).all()
    assert (prediction >= 0).all() and (prediction[:, 0] >= panel.half_hour[-1] - 1e-7).all()
    np.testing.assert_allclose(prediction.sum(0), anchor.sum(0), rtol=5e-14, atol=1e-6)
    assert audit["training_sources"] == 154
    assert audit["first_training_source"] == "2025-08-07 08:00:00"
    assert audit["last_training_source"] == "2025-08-20 13:00:00"
    assert audit["latest_label_timestamp"] == "2025-08-20 17:00:00"
    assert audit["origin"] == "2025-08-21 10:30:00"
    bad = anchor.copy(); bad[0, 0] = panel.half_hour[-1, 0] - 1
    with pytest.raises(ValueError, match="midpoint"):
        predict_stage_flow(panel, bad)


def test_zero_candidate_column_falls_back_to_anchor(arrays, monkeypatch):
    import rwb_forecasting.stage_flow as module
    panel = _panel(arrays, 24 * 48 + 22)
    anchor = np.zeros((4, 8)); anchor[:, 0] = panel.half_hour[-1]
    monkeypatch.setattr(module, "_fit_pair", lambda p, s, c: [
        (np.zeros((73, 8)), np.zeros((4, 8))), (np.zeros((79, 8)), np.zeros((4, 8)))])
    prediction, audit = module.predict_stage_flow(panel, anchor)
    np.testing.assert_array_equal(prediction, anchor)
    assert audit["zero_column_fallbacks"] == 8


def test_panel_and_loader_grid_guards(arrays, tmp_path):
    times, hourly, status, routes = arrays
    with pytest.raises(ValueError, match="grid"):
        StageFlowPanel(times.delete(2), hourly[:-1], status[:-1], routes)
    bad = status.copy(); bad[0, 0, 0] = -1
    with pytest.raises(ValueError, match="panel"):
        StageFlowPanel(times, hourly, bad, routes)
    frame = pd.DataFrame({"timestamp": np.repeat(times[:50], 4), "route_id": np.tile(routes, 50),
                          "target_1h": hourly[:50].ravel(),
                          **{f"status_{j + 1}": status[:50, :, j].ravel() for j in range(3)}})
    path = tmp_path / "train.parquet"
    frame.sample(frac=1, random_state=2).to_parquet(path, index=False)
    loaded = load_stage_flow_panel(path, times[39], routes)
    np.testing.assert_array_equal(loaded.hourly, hourly[:40])
    np.testing.assert_array_equal(loaded.statuses, status[:40])
    assert loaded.times[-1] == times[39]
    frame.drop(frame.index[15]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exactly one row"):
        load_stage_flow_panel(path, times[39], routes)


def test_cli_id_alignment_and_grid_guards():
    cli = _cli()
    times = pd.date_range("2025-11-01 11:00", periods=8, freq="30min")
    test = pd.DataFrame({"id": np.arange(16)[::-1], "route_id": np.repeat([0, 1], 8),
                         "timestamp": np.tile(times, 2)}).sample(frac=1, random_state=3)
    anchor = pd.DataFrame({"id": np.arange(16), "y_pred": np.arange(16) + 1.})
    ordered, routes, origin, values = cli.align_inputs(test, anchor)
    np.testing.assert_array_equal(values.ravel(), anchor.set_index("id").loc[ordered.id, "y_pred"])
    assert origin == pd.Timestamp("2025-11-01 10:30")
    np.testing.assert_array_equal(routes, [0, 1])
    for malformed in (test.iloc[:-1], pd.concat([test, test.iloc[:1]])):
        with pytest.raises(ValueError):
            cli.align_inputs(malformed, anchor)
    invalid = anchor.copy(); invalid.loc[0, "y_pred"] = np.nan
    with pytest.raises(ValueError):
        cli.align_inputs(test, invalid)


def test_cli_refuses_overwrite_before_input_access(tmp_path):
    cli = _cli()
    output = tmp_path / "existing.csv"
    output.write_text("keep\n")
    with pytest.raises(SystemExit):
        cli.main(["--train", "absent", "--test", "absent", "--anchor", "absent", "--output", str(output)])
    assert output.read_text() == "keep\n"
