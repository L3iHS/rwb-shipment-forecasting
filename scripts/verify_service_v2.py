"""Run the local service twice, verify point exports and check generated board JS."""

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwb.data import load_panel, make_submission
from rwb.snapshot import route_status, network_summary, find_routes, evidence
from compare_service_models import write_json, sha256


def run():
    timings, logs = {}, {}
    for name, command in [("forecast_first", ["forecast", "--at", "2025-11-01 10:30"]),
                          ("forecast_repeat", ["forecast", "--at", "2025-11-01 10:30"]),
                          ("board", ["report", "--at", "2025-11-01 10:30"])]:
        started = time.perf_counter()
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/pipeline.py"), *command], cwd=ROOT,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        timings[name] = time.perf_counter() - started
        logs[name] = completed.stdout + completed.stderr
        print(name, round(timings[name], 3), "seconds", flush=True)
    if "cache_hit=True" not in logs["forecast_repeat"]:
        raise AssertionError("Repeated forecast did not use snapshot cache")
    panel = load_panel()
    expected = make_submission(panel, np.load(ROOT / "models/rolling/lean_test.npy"))
    actual = pd.read_csv(ROOT / "outputs/submission_service.csv")
    np.testing.assert_array_equal(expected.id, actual.id)
    np.testing.assert_allclose(expected.y_pred, actual.y_pred, rtol=1e-12, atol=1e-8)
    snapshot = json.loads((ROOT / "outputs/snapshot_20251101_1030.json").read_text(encoding="utf-8"))
    row = route_status(snapshot, int(panel.routes[0]))
    assert network_summary(snapshot)["snapshot_id"] == row["snapshot_id"]
    assert find_routes(snapshot)["snapshot_id"] == row["snapshot_id"]
    assert evidence(snapshot, "validation")["snapshot_id"] == row["snapshot_id"]
    forecast = pd.read_csv(ROOT / "outputs/forecast_20251101_1030.csv")
    same_route = forecast[forecast.route_id == row["route_id"]]
    np.testing.assert_allclose(same_route.point_forecast, row["forecast"]["point_forecast"])
    np.testing.assert_allclose(same_route.p50, row["forecast"]["p50"])
    flows = np.array([r["forecast"]["inflow_point"] for r in snapshot["routes"]])
    assert not (np.diff(flows, axis=-1) < -1e-8).any()
    np.testing.assert_allclose(flows.sum(axis=0), snapshot["network"]["inflow_point"])
    board_check = subprocess.run(["node", str(ROOT / "scripts/check_board.cjs")], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", check=True)
    result = {"timings_seconds": timings, "pipeline_logs": logs, "snapshot_id": snapshot["snapshot_id"],
        "snapshot_creation_seconds": snapshot["creation_seconds"], "repeated_snapshot_cache_hit": True,
        "submission_rows": len(actual), "point_export_matches_archived_model": True,
        "tools_and_csv_agree": True, "negative_cumulative_steps": 0,
        "board_runtime_check": board_check.stdout.strip(), "visual_browser_qa": False,
        "bundle_provenance_sha256": sha256(ROOT / "models/prod/service_v2/provenance.json")}
    write_json(ROOT / "reports/colhound/service_v2/verification.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "pipeline_logs"}, indent=2))


if __name__ == "__main__":
    run()
