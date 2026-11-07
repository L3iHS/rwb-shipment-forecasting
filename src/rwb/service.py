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
from rwb.dispatch import DockState, call_plan, inflow_forecast, simulate
from rwb.features import DAY, WEEK, Cumsums
from rwb.forecaster import Forecaster
from rwb.intervals import Conformal
from rwb.monitor import with_fallback


@dataclass
class ServiceConfig:
    lead: int = 2  # half-hours from calling a truck to its arrival
    fill_hours: float = 2.0  # truck capacity in hours of the route's mean flow
    max_wait: int = 4  # half-hours a truck waits before leaving under-filled
    risk: float = 0.5  # quantile of the expected inflow used by the call rule


def route_level(cums: Cumsums, origin: int) -> np.ndarray:
    """Mean half-hour volume over the 4 weeks up to ``origin``."""
    return cums.x[:, origin - 4 * WEEK + 1 : origin + 1].mean(axis=1)


def cumulative_inflow(pred: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """(n_o, R, H): expected volume over the next 1..H half-hours."""
    return np.stack([inflow_forecast(pred, x0, k) for k in range(1, HORIZON + 1)], axis=2)


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

    @property
    def capacity(self) -> np.ndarray:
        return np.maximum(self.cfg.fill_hours * 2 * self.level, 1.0)

    @classmethod
    def calibrate(cls, forecaster: Forecaster, cums: Cumsums, panel: Panel, origins: np.ndarray,
                  pred: np.ndarray, cfg: ServiceConfig) -> "Service":
        """Interval tables from rolling forecasts of a model fitted before ``origins``."""
        level = route_level(cums, int(origins[0]))
        truth = np.nan_to_num(np.stack([panel.y[:, o + 1 : o + 1 + HORIZON] for o in origins]))
        x = cums.x
        act = np.stack([np.cumsum(x[:, o + 1 : o + 1 + HORIZON], axis=1) for o in origins])
        return cls(forecaster, Conformal().fit(pred, truth, level),
                   Conformal().fit(cumulative_inflow(pred, x[:, origins].T), act, level), level, cfg)

    def forecast(self, panel: Panel, cums: Cumsums, origin: int) -> dict[str, np.ndarray]:
        p50 = self.forecaster.predict(panel, cums, np.array([origin]))[0].astype(np.float64)
        p50, stale = with_fallback(p50, panel, cums, origin)
        out = {"p50": p50, "stale": stale}
        for p in (0.1, 0.9):
            out[f"p{int(p * 100)}"] = self.volume_bands.quantile(p50[None], self.level, p)[0]
        inflow = cumulative_inflow(p50[None], cums.x[:, [origin]].T)
        out["inflow"] = inflow[0]
        out["inflow_risk"] = self.inflow_bands.quantile(inflow, self.level, self.cfg.risk)[0]
        out["inflow_p90"] = self.inflow_bands.quantile(inflow, self.level, 0.9)[0]
        return out

    def dock_state(self, panel: Panel, cums: Cumsums, origin: int, hours: int = 24) -> DockState:
        """Dock state at ``origin`` by replaying the last day with the service's own rule.

        A live system reads the backlog from the warehouse system; the data here has none,
        so it is reconstructed from shipments, starting from an empty dock a day earlier.
        """
        origins = np.arange(origin - 2 * hours + 1, origin + 1)
        pred = self.forecaster.predict(panel, cums, origins).astype(np.float64)
        inflow = cumulative_inflow(pred, cums.x[:, origins].T)
        exp = self.inflow_bands.quantile(inflow, self.level, self.cfg.risk)[:, :, self.cfg.lead - 1]
        res = simulate(cums.x, origins, exp, self.capacity, self.cfg.lead, self.cfg.max_wait, decide_last=False)
        assert res.state is not None
        return res.state

    def plan(self, panel: Panel, origin: int, fc: dict[str, np.ndarray], dock: DockState) -> pd.DataFrame:
        """One row per route: what is waiting, what is coming and when to call the next truck."""
        cap = self.capacity
        wait = call_plan(dock.backlog, fc["inflow_risk"], cap, self.cfg.lead, dock.inbound)
        now = panel.times[origin]
        call_at = [None if w < 0 else now + STEP * int(w) for w in wait]
        # Two loads waiting with no truck is where the dock starts to choke.
        overflow = (dock.backlog + fc["inflow_p90"][:, -1]) / cap
        return pd.DataFrame({
            "route_id": panel.routes, "capacity": cap, "backlog": dock.backlog,
            "backlog_loads": dock.backlog / cap,
            "inflow_4h_p50": fc["inflow"][:, -1], "inflow_4h_p90": fc["inflow_p90"][:, -1],
            "trucks_on_the_way": dock.on_the_way, "trucks_at_dock": dock.at_dock,
            "call_at": call_at, "call_now": [c == now for c in call_at],
            "loads_4h_p90": overflow, "stale_data": fc["stale"],
        })

    def save(self, path: Path) -> None:
        self.forecaster.save(path / "model")
        (path / "service.json").write_text(json.dumps({
            "cfg": asdict(self.cfg), "level": self.level.tolist(),
            "volume_bands": _conf_to_json(self.volume_bands), "inflow_bands": _conf_to_json(self.inflow_bands),
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Service":
        d = json.loads((path / "service.json").read_text(encoding="utf-8"))
        return cls(Forecaster.load(path / "model"), _conf_from_json(d["volume_bands"]),
                   _conf_from_json(d["inflow_bands"]), np.array(d["level"]), ServiceConfig(**d["cfg"]))


def forecast_table(panel: Panel, origin: int, fc: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long table route x time with the band, the format a planning system would ingest."""
    R = panel.n_routes
    times = panel.times[origin] + STEP * np.arange(1, HORIZON + 1)
    return pd.DataFrame({
        "route_id": np.repeat(panel.routes, HORIZON), "timestamp": np.tile(times, R),
        "p10": fc["p10"].ravel(), "p50": fc["p50"].ravel(), "p90": fc["p90"].ravel(),
    })


__all__ = ["DAY", "Service", "ServiceConfig", "cumulative_inflow", "forecast_table", "route_level"]
