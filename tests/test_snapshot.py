from dataclasses import replace
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwb.dispatch import DockState
from rwb.features import cumsums
from rwb.models import GBMConfig
from rwb.monitor import DataUnavailableError
from rwb.snapshot import SnapshotStore, route_status, network_summary, find_routes, evidence, what_if
from test_service_v2 import example


def ready_service():
    svc, panel = example()
    svc.forecaster.cfg = GBMConfig()
    svc.forecaster.models = [SimpleNamespace(model_to_string=lambda: "fixed weights")]
    svc.network_inflow_bands = svc.network_bands
    svc.dock_state = lambda *args: DockState(np.zeros(5), np.zeros(5), np.zeros(5))
    return svc, panel


def test_repeated_tools_and_snapshot_do_not_repeat_inference(tmp_path):
    svc, panel = ready_service()
    store, sums = SnapshotStore(tmp_path), cumsums(panel)
    first, hit = store.get_or_create(svc, panel, sums, panel.n_times - 1)
    assert not hit
    svc.forecaster.predict = lambda *args: pytest.fail("cache hit must not rerun ML")
    second, hit = store.get_or_create(svc, panel, sums, panel.n_times - 1)
    assert hit and first == second
    assert route_status(second, 0)["forecast"]["point_forecast"] == [20.] * 8
    assert network_summary(second)["snapshot_id"] == first["snapshot_id"]
    assert find_routes(second)["snapshot_id"] == first["snapshot_id"]
    assert evidence(second)["evidence_id"] == "assumptions"
    with pytest.raises(KeyError):
        route_status(second, 999)


def test_scenario_and_calibration_and_data_invalidate_cache(tmp_path):
    svc, panel = ready_service()
    sums, store = cumsums(panel), SnapshotStore(tmp_path)
    original, _ = store.get_or_create(svc, panel, sums, panel.n_times - 1)
    svc.cfg = replace(svc.cfg, lead=4)
    scenario, hit = store.get_or_create(svc, panel, sums, panel.n_times - 1)
    assert not hit and original["scenario_id"] != scenario["scenario_id"]
    assert original["data_id"] == scenario["data_id"]
    svc.volume_bands.table = svc.volume_bands.table.copy() + .01
    calibrated, hit = store.get_or_create(svc, panel, sums, panel.n_times - 1)
    assert not hit and calibrated["calibration_id"] != scenario["calibration_id"]
    panel.y[0, -1] += 1
    changed, hit = store.get_or_create(svc, panel, cumsums(panel), panel.n_times - 1)
    assert not hit and changed["data_id"] != calibrated["data_id"]


def test_request_for_newer_data_does_not_return_stale_cache(tmp_path):
    svc, panel = ready_service()
    with pytest.raises(DataUnavailableError, match="stale"):
        SnapshotStore(tmp_path).get_or_create(svc, panel, cumsums(panel), panel.n_times - 1,
                                           requested_at=panel.times[-1] + pd.Timedelta(minutes=30))


def test_fallback_is_json_null_not_zero_or_nan(tmp_path):
    svc, panel = ready_service()
    panel.s[0, 0, -1] = np.nan
    dock = DockState(np.zeros(5), np.zeros(5), np.zeros(5))
    snapshot, _ = SnapshotStore(tmp_path).get_or_create(svc, panel, cumsums(panel), panel.n_times - 1, dock=dock)
    row = route_status(snapshot, 0)
    assert row["forecast"]["p50"] == [None] * 8
    assert snapshot["network"]["p90"] == [None] * 8
    assert find_routes(snapshot, "stale")["total"] == 1


def test_what_if_uses_service_and_preserves_original_configuration(tmp_path):
    svc, panel = ready_service()
    result = what_if(SnapshotStore(tmp_path), svc, panel, cumsums(panel), panel.n_times - 1, 120, route_id=0)
    assert result["scenario"]["lead"] == 4
    assert svc.cfg.lead == 2
    assert result["snapshot_id"] == result["result"]["snapshot_id"]
    with pytest.raises(ValueError):
        what_if(SnapshotStore(tmp_path), svc, panel, cumsums(panel), panel.n_times - 1, 45)


def test_tampered_cache_is_not_used(tmp_path):
    svc, panel = ready_service()
    store, sums = SnapshotStore(tmp_path), cumsums(panel)
    first, _ = store.get_or_create(svc, panel, sums, panel.n_times - 1)
    path = tmp_path / (first["snapshot_id"] + ".json")
    corrupted = json.loads(path.read_text())
    corrupted["routes"][0]["forecast"]["point_forecast"][0] = 999999
    path.write_text(json.dumps(corrupted))
    repaired, hit = store.get_or_create(svc, panel, sums, panel.n_times - 1)
    assert not hit and route_status(repaired, 0)["forecast"]["point_forecast"][0] == 20
