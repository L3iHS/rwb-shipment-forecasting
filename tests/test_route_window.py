import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.halfhour import halfhour_bounds
from rwb_forecasting.route_window import (RouteWindowPanel, fit_weekly_route_window,
                                          load_route_window_panel, predict_weekly_route_window)


@pytest.fixture(scope="module")
def arrays():
    rng = np.random.default_rng(2718)
    times = pd.date_range("2025-07-28", periods=28 * 48, freq="30min")
    latent = rng.integers(100, 3000, (len(times) + 1, 4), dtype=np.int32)
    counts = rng.integers(10, 10000, (len(times), 4, 7), dtype=np.int32)
    counts[:, :, 0] = latent[:-1] + latent[1:]
    return times, counts, np.arange(4)


def _panel(arrays, length=None):
    times, counts, routes = arrays
    return RouteWindowPanel(times[:length], counts[:length], routes)


def _cli():
    spec = importlib.util.spec_from_file_location("weekly_route_cli", Path(__file__).parents[1] / "scripts/apply_weekly_route_window.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_midpoint_inplace_and_feature_prefix_causality(arrays):
    full = _panel(arrays)
    low, high = halfhour_bounds(full.raw[:, :, 0], .5)
    np.testing.assert_array_equal(full.half.view("u4"), ((low + high) / 2).astype(np.float32).view("u4"))
    cut = 25 * 48 + 21
    prefix = full.prefix(full.times[cut])
    rebuilt = _panel(arrays, cut + 1)
    times, counts, routes = arrays
    changed = counts.copy(); changed[cut + 1:, :, 1:] = 123456; changed[-1, :, 0] += 10000
    future_mutated = RouteWindowPanel(times, changed, routes)
    for source in (672, 900, cut):
        expected = full.features(source)
        for other in (prefix, rebuilt, future_mutated):
            actual = other.features(source)
            np.testing.assert_array_equal(actual[0].view("u4"), expected[0].view("u4"))
            np.testing.assert_array_equal(actual[1], expected[1])


def test_feature_order_labels_and_cutoff(arrays):
    panel = _panel(arrays)
    source = 1000
    numeric, scale = panel.features(source)
    assert numeric.shape == (4, 105) and numeric.dtype == np.float32
    np.testing.assert_array_equal(numeric[:, 0], np.log(scale).astype(np.float32))
    np.testing.assert_array_equal(numeric[:, 1], (panel.raw[source, :, 0] / scale).astype(np.float32))
    np.testing.assert_array_equal(numeric[:, -2], np.full(4, panel.slot[source], dtype=np.float32))
    x, target, weight = panel.supervised(np.array([source]))
    np.testing.assert_array_equal(x, numeric)
    expected = (panel.cum[source + 9, :, 0] - panel.cum[source + 2, :, 0]) / (7 * scale)
    np.testing.assert_array_equal(target, expected)
    np.testing.assert_array_equal(weight, scale)
    with pytest.raises(ValueError, match="mature"):
        panel.supervised(np.array([panel.n - 9, panel.n - 8]))
    with pytest.raises(ValueError, match="history"):
        panel.features(671)


def test_fixed_weekly_fit_and_inference_guards(arrays, monkeypatch):
    panel = _panel(arrays).prefix(pd.Timestamp("2025-08-23 10:30"))
    captured = {}

    class Dataset:
        def __init__(self, x, **kwargs):
            captured["shape"] = x.shape
            captured["dataset"] = kwargs

    class Model:
        def current_iteration(self):
            return 400

        def predict(self, x, **kwargs):
            return np.ones(len(x))

    def train(params, dataset, num_boost_round):
        captured.update(params=params, rounds=num_boost_round)
        return Model()

    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(Dataset=Dataset, train=train))
    model, audit = fit_weekly_route_window(panel)
    assert captured["rounds"] == 400 and captured["shape"] == (36 * 4, 105)
    assert captured["params"]["objective"] == "regression_l1" and captured["params"]["num_threads"] == 1
    assert audit["first_source"] == "2025-08-11 09:30:00"
    assert audit["last_source"] == "2025-08-22 11:30:00"
    assert audit["latest_label_timestamp"] == "2025-08-22 15:30:00"
    anchor = np.arange(32, dtype=np.float64).reshape(4, 8) + 1000
    prediction, total, decision = predict_weekly_route_window(panel, model, audit, anchor)
    np.testing.assert_array_equal(prediction[:, 0].view("u8"), anchor[:, 0].view("u8"))
    np.testing.assert_allclose(prediction[:, 1:].sum(0), anchor[:, 1:].sum(0), rtol=1e-10)
    assert decision["converged"] and total.shape == (4,)
    with pytest.raises(ValueError, match="Saturday"):
        fit_weekly_route_window(panel.prefix(pd.Timestamp("2025-08-22 10:30")))
    stale = dict(audit, fit_origin="2025-08-16 10:30", latest_label_timestamp="2025-08-15 15:30")
    with pytest.raises(ValueError, match="cutoff"):
        predict_weekly_route_window(panel, model, stale, anchor)


def test_loader_prefix_grid_and_count_guards(arrays, tmp_path):
    times, counts, routes = arrays
    frame = pd.DataFrame({"timestamp": np.repeat(times[:60], 4), "route_id": np.tile(routes, 60),
                          **{name: counts[:60, :, j].ravel() for j, name in enumerate(
                              ["target_1h", *[f"status_{i}" for i in range(1, 7)]])}})
    path = tmp_path / "train.parquet"
    frame.sample(frac=1, random_state=4).to_parquet(path, index=False)
    panel = load_route_window_panel(path, times[49], routes)
    np.testing.assert_array_equal(panel.raw, counts[:50])
    assert panel.times[-1] == times[49]
    frame.iloc[:-1].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="every requested route"):
        load_route_window_panel(path, times[59], routes)
    with pytest.raises(ValueError, match="int32"):
        RouteWindowPanel(times, counts.astype(float), routes)


def test_cli_id_alignment_and_no_overwrite(tmp_path):
    cli = _cli()
    times = pd.date_range("2025-11-01 11:00", periods=8, freq="30min")
    test = pd.DataFrame({"id": np.arange(16)[::-1], "route_id": np.repeat([0, 1], 8),
                         "timestamp": np.tile(times, 2)}).sample(frac=1, random_state=5)
    anchor = pd.DataFrame({"id": np.arange(16), "y_pred": np.arange(16) + .5})
    ordered, routes, origin, values = cli.align_inputs(test, anchor)
    np.testing.assert_array_equal(values.ravel(), anchor.set_index("id").loc[ordered.id, "y_pred"])
    assert origin == pd.Timestamp("2025-11-01 10:30")
    for bad in (test.iloc[:-1], pd.concat([test, test.iloc[:1]])):
        with pytest.raises(ValueError):
            cli.align_inputs(bad, anchor)
    output = tmp_path / "existing.csv"
    output.write_text("keep\n")
    with pytest.raises(SystemExit):
        cli.main(["--train", "absent", "--test", "absent", "--anchor", "absent", "--output", str(output)])
    assert output.read_text() == "keep\n"
