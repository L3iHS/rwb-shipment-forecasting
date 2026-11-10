"""Versioned shared forecast snapshots; reading tools never rerun ML or dock replay."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from rwb.data import Panel
from rwb.dispatch import DockState
from rwb.features import Cumsums
from rwb.flows import FLOW_METHOD
from rwb.monitor import DataUnavailableError
from rwb.service import Service

SNAPSHOT_VERSION = 2


def json_value(value):
    """Strict JSON: missing intervals remain null rather than fabricated zeros."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _digest(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str((part.dtype.str, part.shape)).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(json.dumps(json_value(part), sort_keys=True, allow_nan=False).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def snapshot_identity(service: Service, panel: Panel, cums: Cumsums, origin: int,
                      dock: DockState | None = None) -> dict:
    source_id = _digest(*[(Path(__file__).parent / name).read_text(encoding="utf-8")
                         for name in ("snapshot.py", "service.py", "flows.py", "intervals.py", "features.py", "models.py", "monitor.py", "forecaster.py", "data.py")])
    model_id = _digest(asdict(service.forecaster.cfg), *[m.model_to_string() for m in service.forecaster.models])
    calibration_id = _digest(service.level, *[part for conf in (service.volume_bands, service.inflow_bands,
        service.network_bands, service.network_inflow_bands) for part in (conf.table, conf.edges, conf.probs, conf.eps)], service.calibration_metadata)
    # Include cums: a stale derived feature cache must not reuse an otherwise identical snapshot.
    data_id = _digest(panel.routes, panel.times[:origin + 1].asi8, panel.y[:, :origin + 1],
                      panel.s[:, :, :origin + 1], cums.x[:, :origin + 1],
                      cums.y[:, :origin + 2], cums.y2[:, :origin + 2], cums.s[:, :, :origin + 2], cums.xc[:, :origin + 2])
    state_id = "simulated_empty_dock_24h" if dock is None else _digest(dock.backlog, dock.on_the_way, dock.at_dock)
    scenario_id = _digest(asdict(service.cfg), state_id)
    snapshot_id = _digest(SNAPSHOT_VERSION, FLOW_METHOD, source_id, model_id, calibration_id, data_id, scenario_id,
                          service.forecaster.label_cutoff, str(panel.times[origin]))
    return {"snapshot_id": snapshot_id, "scenario_id": scenario_id, "model_id": model_id,
            "calibration_id": calibration_id, "data_id": data_id, "source_id": source_id}


class SnapshotStore:
    def __init__(self, directory: Path, evidence_path: Path | None = None):
        self.directory = Path(directory)
        self.evidence_path = evidence_path

    def get_or_create(self, service: Service, panel: Panel, cums: Cumsums, origin: int,
                      dock: DockState | None = None, requested_at=None) -> tuple[dict, bool]:
        service.validate_history(panel, origin)
        if origin < service.forecaster.label_cutoff:
            raise DataUnavailableError("The fitted model is newer than the requested snapshot")
        if requested_at is not None and pd.Timestamp(requested_at) != panel.times[origin]:
            raise DataUnavailableError("Requested time has no matching available data; refusing a stale forecast")
        started = time.perf_counter()
        identity = snapshot_identity(service, panel, cums, origin, dock)
        validation = None
        if self.evidence_path is not None:
            content = Path(self.evidence_path).read_bytes()
            identity["evidence_id"] = hashlib.sha256(content).hexdigest()
            validation = json.loads(content)
            if validation.get("flow_method") != FLOW_METHOD:
                raise ValueError("Validation report describes a different flow method")
            identity["snapshot_id"] = _digest(identity["snapshot_id"], identity["evidence_id"])
        path = self.directory / f"{identity['snapshot_id']}.json"
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                checksum = _digest({k:v for k,v in cached.items() if k != "content_sha256"})
                if (all(cached.get(key) == value for key, value in identity.items())
                        and cached.get("schema_version") == SNAPSHOT_VERSION and cached.get("content_sha256") == checksum):
                    return cached, True
            except (ValueError, OSError):
                pass
        fc = service.forecast(panel, cums, origin)
        network = service.network_forecast(fc)
        provided = dock is not None
        if dock is None:
            dock = service.dock_state(panel, cums, origin)
        plan = service.plan(panel, origin, fc, dock)
        routes = []
        for i, row in enumerate(plan.to_dict(orient="records")):
            row.update({"forecast": {name: values[i] for name, values in fc.items()},
                        "history": panel.y[i, max(0, origin - 15):origin + 1],
                        "evidence_ids": ["forecast", "dispatch_rule", "assumptions"]})
            row["cargo_at_arrival"] = row["backlog"] + row["inflow_during_lead"]
            routes.append(row)
        snapshot = json_value({"schema_version": SNAPSHOT_VERSION, "flow_method": FLOW_METHOD, **identity,
            "as_of": panel.times[origin], "timezone": "unspecified in source data", "step_minutes": 30,
            "forecast_times": [panel.times[origin] + (h + 1) * pd.Timedelta(minutes=30) for h in range(8)],
            "units": "source item counters; not mass or physical truck volume", "scenario": asdict(service.cfg),
            "calibration_metadata": service.calibration_metadata,
            "dock_state_source": "provided_by_caller" if provided else "simulation_from_empty_dock_24h",
            "dock_replay_in_sample": not provided and service.forecaster.label_cutoff > origin - 48,
            "routes": routes, "network": network, "data_quality": {"fallback_routes": int(fc["stale"].sum()),
            "network_intervals_valid": bool(fc["intervals_valid"].all())},
            "evidence": {"forecast": "Saved LightGBM Huber model; point and estimated quantiles are distinct",
                "dispatch_rule": "Call when backlog + risk-adjusted lead-time inflow covers capacity * (1 + inbound trucks)",
                "assumptions": "Capacity is a scenario in source units; reconstructed flow and simulated dock are not observations"},
            "creation_seconds": time.perf_counter() - started})
        if validation is not None:
            snapshot["evidence"]["validation"] = {"point": validation["point"], "comparison": validation["comparison"],
                "limitations": validation["limitations"], "source_hash": identity["evidence_id"]}
        snapshot["content_sha256"] = _digest(snapshot)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False)
            except Exception:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return snapshot, False


def route_status(snapshot: dict, route_id: int) -> dict:
    for row in snapshot["routes"]:
        if row["route_id"] == route_id:
            return {"snapshot_id": snapshot["snapshot_id"], "scenario_id": snapshot["scenario_id"], **row}
    raise KeyError(f"Unknown route {route_id}")


def find_routes(snapshot: dict, kind: str = "call_now", limit: int = 20) -> dict:
    filters = {"call_now": lambda r: r["call_now"], "within_hour": lambda r: r["call_at"] is not None and
               0 <= (pd.Timestamp(r["call_at"]) - pd.Timestamp(snapshot["as_of"])).total_seconds() <= 3600,
               "stale": lambda r: r["stale_data"]}
    if kind not in filters or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Unknown filter or limit outside 1..100")
    rows = [r for r in snapshot["routes"] if filters[kind](r)]
    rows.sort(key=lambda r: (r["call_at"] or "9999", r["route_id"]))
    return {"snapshot_id": snapshot["snapshot_id"], "scenario_id": snapshot["scenario_id"], "total": len(rows),
            "routes": [{k: r[k] for k in ("route_id", "call_at", "call_now", "backlog", "cargo_at_arrival", "next_load_threshold", "evidence_ids")} for r in rows[:limit]]}


def network_summary(snapshot: dict) -> dict:
    return {"snapshot_id": snapshot["snapshot_id"], "scenario_id": snapshot["scenario_id"],
            "as_of": snapshot["as_of"], "forecast_times": snapshot["forecast_times"], **snapshot["network"],
            "evidence_ids": ["forecast", "assumptions"]}


def evidence(snapshot: dict, topic: str = "assumptions") -> dict:
    topic = {"model": "forecast", "dispatch": "dispatch_rule"}.get(topic, topic)
    if topic not in snapshot["evidence"]:
        raise KeyError(f"Unknown evidence topic {topic}")
    return {"snapshot_id": snapshot["snapshot_id"], "evidence_id": topic, "description": snapshot["evidence"][topic]}


def what_if(store: SnapshotStore, service: Service, panel: Panel, cums: Cumsums, origin: int,
            lead_minutes: int, fill_hours: float | None = None, route_id: int | None = None) -> dict:
    """Recompute a simulated scenario through the same service, without retraining."""
    if not isinstance(lead_minutes, int) or lead_minutes % 30 or not 30 <= lead_minutes <= 240:
        raise ValueError("Lead time must be 30..240 minutes in 30-minute steps")
    base, _ = store.get_or_create(service, panel, cums, origin)
    if route_id is not None:
        route_status(base, route_id)  # Reject an unknown ID before the expensive scenario.
    cfg = replace(service.cfg, lead=lead_minutes // 30,
                  fill_hours=service.cfg.fill_hours if fill_hours is None else fill_hours)
    scenario, hit = store.get_or_create(replace(service, cfg=cfg), panel, cums, origin)
    return {"base_snapshot_id": base["snapshot_id"], "snapshot_id": scenario["snapshot_id"],
            "scenario_id": scenario["scenario_id"], "cache_hit": hit,
            "scenario": scenario["scenario"], "dock_state_source": scenario["dock_state_source"],
            "result": route_status(scenario, route_id) if route_id is not None else network_summary(scenario)}
