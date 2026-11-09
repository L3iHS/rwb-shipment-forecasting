"""The forecasting service a dispatcher would use: forecast with bands, dock state, truck calls.

A bundle on disk holds the fitted forecaster, the interval tables and the route parameters,
so the half-hourly run only loads it and reads the data up to "now".
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rwb.data import HORIZON, STEP, Panel
from rwb.dispatch import DockState, call_plan, simulate
from rwb.features import DAY, WEEK, MAX_LOOKBACK, Cumsums, cumsums
from rwb.flows import FLOW_METHOD, reconcile_hourly
from rwb.forecaster import Forecaster
from rwb.intervals import Conformal
from rwb.monitor import DataUnavailableError, with_fallback


@dataclass
class ServiceConfig:
    lead: int = 2  # half-hours from calling a truck to its arrival
    fill_hours: float = 2.0  # truck capacity in hours of the route's mean flow
    max_wait: int = 4  # half-hours a truck waits before leaving under-filled
    risk: float = 0.5  # quantile of the expected inflow used by the call rule

    def __post_init__(self):
        if not isinstance(self.lead, int) or not 1 <= self.lead <= HORIZON:
            raise ValueError("lead must be 1..8 half-hours")
        if not np.isfinite(self.fill_hours) or self.fill_hours <= 0 or not isinstance(self.max_wait, int) or self.max_wait < 1:
            raise ValueError("Capacity and maximum wait must be positive")
        if self.risk not in Conformal().probs:
            raise ValueError("Risk must be one of the calibrated quantile probabilities")


def route_level(cums: Cumsums, origin: int) -> np.ndarray:
    """Mean half-hour volume over the 4 weeks up to ``origin``."""
    return cums.x[:, origin - 4 * WEEK + 1 : origin + 1].mean(axis=1)


def cumulative_inflow(pred: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """(n_o, R, H): expected volume over the next 1..H half-hours."""
    return reconcile_hourly(pred, x0).cumulative


def _conf_to_json(c: Conformal) -> dict:
    return {"probs": list(c.probs), "eps": c.eps, "table": c.table.tolist(), "edges": c.edges.tolist()}


def _conf_from_json(d: dict) -> Conformal:
    return Conformal(tuple(d["probs"]), d["eps"], np.array(d["table"]), np.array(d["edges"]))


@dataclass
class Service:
    forecaster: Forecaster
    volume_bands: Conformal  # hourly target_1h
    inflow_bands: Conformal  # cumulative half-hour volume, the truck decision input
    level: np.ndarray  # (R,) route level the bands and capacities were set with
    cfg: ServiceConfig
    routes: np.ndarray | None = None
    network_bands: Conformal | None = None
    network_inflow_bands: Conformal | None = None
    calibration_metadata: dict | None = None

    @property
    def capacity(self) -> np.ndarray:
        return np.maximum(self.cfg.fill_hours * 2 * self.level, 1.0)

    @classmethod
    def calibrate(cls, forecaster: Forecaster, cums: Cumsums, panel: Panel, origins: np.ndarray,
                  pred: np.ndarray, cfg: ServiceConfig) -> "Service":
        """Interval tables from rolling forecasts of a model fitted before ``origins``."""
        level = route_level(cums, int(origins[0]))
        if not len(origins) or origins.min() < MAX_LOOKBACK or origins.max() + HORIZON >= panel.n_times:
            raise ValueError("Calibration targets must be fully inside the observed panel")
        if pred.shape != (len(origins), panel.n_routes, HORIZON):
            raise ValueError("Calibration forecast shape does not match the panel")
        truth = np.stack([panel.y[:, o + 1 : o + 1 + HORIZON] for o in origins])
        x = cums.x
        act = np.stack([np.cumsum(x[:, o + 1 : o + 1 + HORIZON], axis=1) for o in origins])
        cumulative = cumulative_inflow(pred, x[:, origins].T)
        return cls(forecaster, Conformal().fit(pred, truth, level),
                   Conformal().fit(cumulative, act, level), level, cfg,
                   panel.routes.copy(), Conformal().fit(pred.sum(axis=1, keepdims=True), truth.sum(axis=1, keepdims=True), np.array([level.sum()])),
                   Conformal().fit(cumulative.sum(axis=1, keepdims=True), act.sum(axis=1, keepdims=True), np.array([level.sum()])),
                   {"origins": len(origins), "start": str(panel.times[origins.min()]), "end_origin": str(panel.times[origins.max()]),
                    "last_observed_label": str(panel.times[origins.max() + HORIZON]), "source": "archived rolling predictions"})

    def network_forecast(self, fc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Calibrate network totals directly; never sum marginal route quantiles."""
        point = fc["point_forecast"].sum(axis=0)
        inflow = fc["inflow_point"].sum(axis=0)
        level = np.array([self.level.sum()])
        result = {"point_forecast": point, "inflow_point": inflow}
        for p in (.1, .5, .9):
            q = self.network_bands.quantile(point[None, None], level, p)[0, 0]
            iq = self.network_inflow_bands.cumulative_quantile(inflow[None, None], level, p)[0, 0]
            if not fc["intervals_valid"].all():
                q[:], iq[:] = np.nan, np.nan
            result[f"p{int(p * 100)}"], result[f"inflow_p{int(p * 100)}"] = q, iq
        return result

    def validate_history(self, panel: Panel, origin: int) -> None:
        if not 0 <= origin < panel.n_times or origin < MAX_LOOKBACK:
            raise DataUnavailableError("Unknown origin or insufficient feature history")
        if self.routes is None or not np.array_equal(self.routes, panel.routes):
            raise DataUnavailableError("Routes or route ordering differ from the fitted service")
        if panel.s.shape != (6, panel.n_routes, panel.n_times):
            raise DataUnavailableError("Six status series are required")
        if not panel.times[:origin + 1].equals(pd.date_range(panel.times[0], periods=origin + 1, freq=STEP)):
            raise DataUnavailableError("History must use a regular 30-minute grid")

    def forecast(self, panel: Panel, cums: Cumsums, origin: int) -> dict[str, np.ndarray]:
        self.validate_history(panel, origin)
        if origin < self.forecaster.label_cutoff:
            raise DataUnavailableError("Model was trained after this forecast origin")
        fallback, stale = with_fallback(np.zeros((panel.n_routes, HORIZON)), panel, cums, origin)
        model_panel, model_cums = panel, cums
        if stale.any() and not stale.all():
            # Invalid routes are discarded after prediction. Sanitize only their rows so
            # nonfinite inputs cannot crash feature construction before fallback is applied.
            if getattr(getattr(self.forecaster, "cfg", None), "global_feats", False):
                raise DataUnavailableError("Network-wide model features cannot use incomplete route histories")
            model_panel = Panel(panel.times, panel.routes, panel.y.copy(), panel.s.copy(), panel.test)
            model_panel.y[stale], model_panel.s[:, stale] = 0, 0
            model_cums = cumsums(model_panel)
        if stale.all():
            point = fallback
        else:
            point = self.forecaster.predict(model_panel, model_cums, np.array([origin]))[0].astype(np.float64)
            point = np.where(stale[:, None], fallback, point)
        if not np.isfinite(point).all() or (point < 0).any():
            raise DataUnavailableError("Model did not produce finite nonnegative forecasts")
        out = {"point_forecast": point, "stale": stale, "intervals_valid": ~stale}
        flow = reconcile_hourly(point, cums.x[:, origin])
        out.update(half_hour=flow.half_hour, reconciled_hourly=flow.hourly,
                   hourly_adjustment=flow.hourly_adjustment, inflow_point=flow.cumulative)
        for p in (0.1, 0.5, 0.9):
            q = self.volume_bands.quantile(point[None], self.level, p)[0]
            q[stale] = np.nan  # Main-model calibration is not valid for fallback routes.
            out[f"p{int(p * 100)}"] = q
            iq = self.inflow_bands.cumulative_quantile(flow.cumulative[None], self.level, p)[0]
            iq[stale] = np.nan
            out[f"inflow_p{int(p * 100)}"] = iq
        risk = self.inflow_bands.cumulative_quantile(flow.cumulative[None], self.level, self.cfg.risk)[0]
        # Fallback plans use the uncalibrated point, and explicitly carry an invalid-band flag.
        out["inflow_risk"] = np.where(stale[:, None], flow.cumulative, risk)
        return out

    def dock_state(self, panel: Panel, cums: Cumsums, origin: int, hours: int = 24) -> DockState:
        """Dock state at ``origin`` by replaying the last day with the service's own rule.

        A live system reads the backlog from the warehouse system; the data here has none,
        so it is reconstructed from shipments, starting from an empty dock a day earlier.
        """
        origins = np.arange(origin - 2 * hours + 1, origin + 1)
        if origins[0] < MAX_LOOKBACK:
            raise DataUnavailableError("Insufficient history for simulated dock replay")
        self.validate_history(panel, origin)
        pred = self.forecaster.predict(panel, cums, origins).astype(np.float64)
        inflow = cumulative_inflow(pred, cums.x[:, origins].T)
        # Replay requires complete histories; zero-filled gaps must not create fictitious cargo.
        if not np.isfinite(panel.y[:, :origin + 1]).all() or not np.isfinite(panel.s[:, :, :origin + 1]).all():
            raise DataUnavailableError("Dock replay needs complete history; provide a dock state for incomplete data")
        exp = self.inflow_bands.cumulative_quantile(inflow, self.level, self.cfg.risk)[:, :, self.cfg.lead - 1]
        res = simulate(cums.x, origins, exp, self.capacity, self.cfg.lead, self.cfg.max_wait, decide_last=False)
        assert res.state is not None
        return res.state

    def plan(self, panel: Panel, origin: int, fc: dict[str, np.ndarray], dock: DockState) -> pd.DataFrame:
        """One row per route: what is waiting, what is coming and when to call the next truck."""
        cap = self.capacity
        for values in (dock.backlog, dock.on_the_way, dock.at_dock):
            if values.shape != cap.shape or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Dock state must have finite nonnegative values for every route")
        if ((dock.on_the_way % 1) != 0).any() or ((dock.at_dock % 1) != 0).any():
            raise ValueError("Truck counts must be integers")
        wait = call_plan(dock.backlog, fc["inflow_risk"], cap, self.cfg.lead, dock.inbound)
        now = panel.times[origin]
        call_at = [None if w < 0 else now + STEP * int(w) for w in wait]
        # Two loads waiting with no truck is where the dock starts to choke.
        overflow = (dock.backlog + fc["inflow_p90"][:, -1]) / cap
        return pd.DataFrame({
            "route_id": panel.routes, "capacity": cap, "backlog": dock.backlog,
            "backlog_loads": dock.backlog / cap,
            "inflow_4h_point": fc["inflow_point"][:, -1], "inflow_4h_p50": fc["inflow_p50"][:, -1],
            "inflow_4h_p90": fc["inflow_p90"][:, -1],
            "trucks_on_the_way": dock.on_the_way, "trucks_at_dock": dock.at_dock,
            "call_at": call_at, "call_now": [c == now for c in call_at],
            "loads_4h_p90": overflow, "stale_data": fc["stale"], "intervals_valid": fc["intervals_valid"],
            "inflow_during_lead": fc["inflow_risk"][:, self.cfg.lead - 1],
            "next_load_threshold": cap * (1 + dock.inbound),
        })

    def save(self, path: Path) -> None:
        self.forecaster.save(path / "model")
        (path / "service.json").write_text(json.dumps({
            "schema_version": 2, "flow_method": FLOW_METHOD,
            "cfg": asdict(self.cfg), "level": self.level.tolist(),
            "volume_bands": _conf_to_json(self.volume_bands), "inflow_bands": _conf_to_json(self.inflow_bands),
            "routes": self.routes.tolist(), "network_bands": _conf_to_json(self.network_bands),
            "network_inflow_bands": _conf_to_json(self.network_inflow_bands),
            "calibration_metadata": self.calibration_metadata,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Service":
        d = json.loads((path / "service.json").read_text(encoding="utf-8"))
        if d.get("schema_version") != 2 or d.get("flow_method") != FLOW_METHOD:
            raise ValueError("Legacy inflow calibration is incompatible. Run scripts/prepare_service_v2.py first.")
        return cls(Forecaster.load(path / "model"), _conf_from_json(d["volume_bands"]),
                   _conf_from_json(d["inflow_bands"]), np.array(d["level"]), ServiceConfig(**d["cfg"]),
                   np.array(d["routes"]), _conf_from_json(d["network_bands"]), _conf_from_json(d["network_inflow_bands"]),
                   d.get("calibration_metadata"))


def forecast_table(panel: Panel, origin: int, fc: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long table route x time with the band, the format a planning system would ingest."""
    R = panel.n_routes
    times = panel.times[origin] + STEP * np.arange(1, HORIZON + 1)
    return pd.DataFrame({
        "route_id": np.repeat(panel.routes, HORIZON), "timestamp": np.tile(times, R),
        "point_forecast": fc["point_forecast"].ravel(),
        "p10": fc["p10"].ravel(), "p50": fc["p50"].ravel(), "p90": fc["p90"].ravel(),
    })


__all__ = ["DAY", "Service", "ServiceConfig", "cumulative_inflow", "forecast_table", "route_level"]
