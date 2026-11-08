"""Production pipeline: daily training, half-hourly forecast with truck calls, quality control.

    python scripts/pipeline.py train                         # fit on all data, calibrate bands, save bundle
    python scripts/pipeline.py forecast --at "2025-11-01 10:30"
    python scripts/pipeline.py monitor                       # daily metrics and alerts on the forecast log

``train`` takes the interval calibration from the rolling forecasts of the latest validation week
(scripts/rolling.py), i.e. from a model fitted before those days, as a live log would provide.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.data import HORIZON, ROOT, STEP, load_panel, make_submission
from rwb.features import DAY, cumsums
from rwb.forecaster import LEAN, Forecaster
from rwb.intervals import Conformal
from rwb.monitor import alerts, daily_metrics
from rwb.service import Service, ServiceConfig, forecast_table, route_level

log = logging.getLogger("pipeline")
BUNDLE = ROOT / "models/prod/service"
ROLL = ROOT / "models/rolling"
OUT = ROOT / "outputs"


def origin_index(panel, at: str | None) -> int:
    if at is None:
        return panel.n_times - 1
    ts = pd.Timestamp(at)
    i = panel.times.get_indexer([ts])[0]
    if i < 0:
        raise SystemExit(f"{ts} is not in the data ({panel.times[0]} .. {panel.times[-1]}, step 30 min)")
    return int(i)


def cmd_train(a: argparse.Namespace) -> None:
    panel, cums = load_panel(), None
    cums = cumsums(panel)
    cut = origin_index(panel, a.until)
    t0 = time.time()
    prod = ROOT / "models/prod/lean"
    if a.until is None and (prod / "forecaster.json").exists():
        fc = Forecaster.load(prod)  # fitted by scripts/rolling.py --final on the same data
        log.info("reusing %s", prod)
    else:
        fc = Forecaster(LEAN).fit(panel, cums, cut)
    cal = np.load(ROLL / f"{a.name}_fold0.npz")
    svc = Service.calibrate(fc, cums, panel, cal["origins"], cal["pred"].astype(np.float64), ServiceConfig(risk=a.risk))
    svc.save(BUNDLE)
    log.info("bundle %s ready in %.0fs (data until %s)", BUNDLE, time.time() - t0, panel.times[cut])


def cmd_forecast(a: argparse.Namespace) -> None:
    t0 = time.time()
    svc = Service.load(BUNDLE)
    panel = load_panel()
    cums = cumsums(panel)
    o = origin_index(panel, a.at)
    if o < svc.forecaster.label_cutoff and a.at is not None:
        log.warning("forecasting from %s with a model trained up to %s: in-sample, for demo only",
                    panel.times[o], panel.times[svc.forecaster.label_cutoff])
    t1 = time.time()
    fc = svc.forecast(panel, cums, o)
    t_fc = time.time()
    dock = svc.dock_state(panel, cums, o)  # a live system reads it from the warehouse instead
    plan = svc.plan(panel, o, fc, dock)
    t2 = time.time()
    OUT.mkdir(exist_ok=True)
    stamp = panel.times[o].strftime("%Y%m%d_%H%M")
    forecast_table(panel, o, fc).to_csv(OUT / f"forecast_{stamp}.csv", index=False)
    plan.to_csv(OUT / f"trucks_{stamp}.csv", index=False)
    if o == panel.n_times - 1:
        make_submission(panel, fc["p50"]).to_csv(OUT / "submission_service.csv", index=False)
    log.info("forecast from %s: load %.1fs, forecast %.1fs, dock replay + plan %.1fs",
             panel.times[o], t1 - t0, t_fc - t1, t2 - t_fc)
    log.info("routes: call a truck now %d, within 4h %d, stale data %d; 4h volume %.3g (p90 %.3g)",
             int(plan["call_now"].sum()), int(plan["call_at"].notna().sum()), int(plan["stale_data"].sum()),
             fc["inflow"][:, -1].sum(), fc["inflow_p90"][:, -1].sum())


def truth_at(panel, origins: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.stack([panel.y[:, o + 1 : o + 1 + HORIZON] for o in origins]))


def monitor_frame(panel, cums, name: str, daily: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast log of the latest week (bands calibrated on the week before) against actuals."""
    cal, ev = np.load(ROLL / f"{name}_fold1.npz"), np.load(ROLL / f"{name}_fold0.npz")
    bands = Conformal().fit(cal["pred"].astype(np.float64), truth_at(panel, cal["origins"]),
                            route_level(cums, int(cal["origins"][0])))
    o = ev["origins"]
    keep = (o - o[0]) % DAY == 0 if daily else np.ones(len(o), dtype=bool)
    o, pred = o[keep], ev["pred"][keep].astype(np.float64)
    lvl = route_level(cums, int(ev["origins"][0]))
    df = daily_metrics(pred, truth_at(panel, o), panel.times[o],
                       bands.quantile(pred, lvl, 0.1), bands.quantile(pred, lvl, 0.9))
    return df, alerts(df)


def cmd_monitor(a: argparse.Namespace) -> None:
    """Replays the forecast log (rolling forecasts of the latest week) against what was shipped."""
    panel = load_panel()
    df, al = monitor_frame(panel, cumsums(panel), a.name, a.daily)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "monitor_metrics.csv", index=False)
    al.to_csv(OUT / "monitor_alerts.csv", index=False)
    log.info("metrics:\n%s", df.round({c: 4 for c in df.columns if c != "origin"}).to_string(index=False))
    log.info("alerts:\n%s", al.to_string(index=False) if len(al) else "none")


def board_data(a: argparse.Namespace) -> dict:
    """Everything the dispatcher board shows for the forecast origin: plan, bands, monitor, backtest."""
    svc = Service.load(BUNDLE)
    panel = load_panel()
    cums = cumsums(panel)
    o = origin_index(panel, a.at)
    fc = svc.forecast(panel, cums, o)
    dock = svc.dock_state(panel, cums, o)
    plan = svc.plan(panel, o, fc, dock)
    df, al = monitor_frame(panel, cums, a.name, daily=True)
    bt = json.loads((ROOT / f"reports/service/{a.name}.json").read_text(encoding="utf-8"))

    # Network band: ratio actual / forecast of the all-route total per horizon over the latest week.
    ev = np.load(ROLL / f"{a.name}_fold0.npz")
    ratio = truth_at(panel, ev["origins"]).sum(axis=1) / ev["pred"].astype(np.float64).sum(axis=1)  # (n_o, H)
    net_lo, net_hi = np.quantile(ratio, 0.1, axis=0), np.quantile(ratio, 0.9, axis=0)

    now = panel.times[o]
    hist_n = 16
    hist = np.nan_to_num(panel.y[:, o - hist_n + 1 : o + 1])
    cap = svc.capacity
    wait = [-1 if pd.isna(c) else int((c - now) / STEP) for c in plan["call_at"]]
    routes = [{
        "id": int(panel.routes[r]), "cap": round(float(cap[r])), "bl": round(float(dock.backlog[r] / cap[r]), 3),
        "way": int(dock.on_the_way[r]), "dock": int(dock.at_dock[r]), "call": wait[r],
        "h": np.round(hist[r]).astype(int).tolist(),
        "p10": np.round(fc["p10"][r]).astype(int).tolist(), "p50": np.round(fc["p50"][r]).astype(int).tolist(),
        "p90": np.round(fc["p90"][r]).astype(int).tolist(),
        "c50": np.round(fc["inflow"][r]).astype(int).tolist(), "c90": np.round(fc["inflow_p90"][r]).astype(int).tolist(),
        "cr": np.round(fc["inflow_risk"][r]).astype(int).tolist(),  # inflow the call rule uses
    } for r in range(panel.n_routes)]
    p50_total = fc["p50"].sum(axis=0)
    data = {
        "now": now.isoformat(), "step_min": 30, "hist_n": hist_n,
        "model": {"trained_until": str(panel.times[svc.forecaster.label_cutoff]), "lib": svc.forecaster.cfg.lib,
                  "objective": svc.forecaster.cfg.objective, "lead_h": svc.cfg.lead / 2,
                  "fill_hours": svc.cfg.fill_hours, "max_wait_h": svc.cfg.max_wait / 2, "risk": svc.cfg.risk},
        "network": {"hist": np.round(hist.sum(axis=0)).tolist(), "p50": np.round(p50_total).tolist(),
                    "lo": np.round(p50_total * net_lo).tolist(), "hi": np.round(p50_total * net_hi).tolist()},
        "routes": routes,
        "monitor": [{"day": r.origin.strftime("%d.%m"), "wape": r.wape, "bias": r.bias, "cov": r.coverage}
                    for r in df.itertuples()],
        "alerts": [{"day": pd.Timestamp(r.origin).strftime("%d.%m"), "signal": r.signal, "detail": r.detail,
                    "action": r.action} for r in al.itertuples()],
        "backtest": {"sensitivity": bt["dispatch"]["sensitivity"], "per_day": bt["dispatch"]["per_day"],
                     "intervals": {k: v["coverage"] for k, v in bt["intervals"].items() if isinstance(v, dict)},
                     "point": bt["point"]["all_origins"], "week": [str(panel.times[ev["origins"][0]].date()),
                                                                   str(panel.times[ev["origins"][-1]].date())]},
    }
    return data


def cmd_report(a: argparse.Namespace) -> None:
    """Self-contained dispatcher board (one HTML file, works offline) for the forecast origin."""
    data = board_data(a)
    html = (ROOT / "src/rwb/board.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    OUT.mkdir(exist_ok=True)
    out = OUT / "dispatch_board.html"
    out.write_text(html, encoding="utf-8")
    log.info("board %s (%.0f KB)", out, out.stat().st_size / 1024)


def cmd_export(a: argparse.Namespace) -> None:
    """The board data as JSON for the React dispatcher console (frontend/)."""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(board_data(a), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log.info("board data %s (%.0f KB)", out, out.stat().st_size / 1024)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--until", default=None, help="last timestamp whose label may be used")
    t.add_argument("--name", default="lean", help="rolling forecasts used to calibrate the bands")
    t.add_argument("--risk", type=float, default=0.5, help="quantile of expected inflow for the call rule")
    f = sub.add_parser("forecast")
    f.add_argument("--at", default=None, help="forecast origin, default: the last timestamp in the data")
    m = sub.add_parser("monitor")
    m.add_argument("--name", default="lean")
    m.add_argument("--daily", action="store_true", help="only the test clock time, one origin per day")
    r = sub.add_parser("report")
    r.add_argument("--at", default=None)
    r.add_argument("--name", default="lean")
    e = sub.add_parser("export")
    e.add_argument("--at", default=None)
    e.add_argument("--name", default="lean")
    e.add_argument("--out", default=str(ROOT / "frontend/public/board.json"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    {"train": cmd_train, "forecast": cmd_forecast, "monitor": cmd_monitor, "report": cmd_report,
     "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
