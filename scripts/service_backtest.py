"""Backtest of the forecasting service on the rolling forecasts of scripts/rolling.py.

Everything is calibrated on the earlier validation week and scored on the later one:
point accuracy from every half-hour, interval coverage, and the truck-call simulation that
compares the reactive practice, simple rules and the model.

    python scripts/service_backtest.py --name lean
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.data import HORIZON, ROOT, load_panel
from rwb.dispatch import inflow_actual, simulate
from rwb.flows import legacy_cumulative_inflow


def inflow_forecast(pred, x0, k):
    """Keep this historical v1 experiment reproducible; use validate_service_v2.py for v2."""
    return legacy_cumulative_inflow(pred, x0)[..., k - 1]
from rwb.features import DAY, WEEK, cumsums
from rwb.intervals import Conformal, coverage, pinball

log = logging.getLogger("service")
ROLL = ROOT / "models/rolling"
OUT = ROOT / "reports/service"


def truth(panel, origins: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.stack([panel.y[:, o + 1 : o + 1 + HORIZON] for o in origins]))


def score(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean over origins of WAPE + |bias|, and mean |bias| alone."""
    s = y.sum(axis=(1, 2))
    wape = np.abs(pred - y).sum(axis=(1, 2)) / s
    bias = np.abs(pred.sum(axis=(1, 2)) / s - 1)
    return float((wape + bias).mean()), float(bias.mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="lean")
    p.add_argument("--lead", type=int, default=2, help="truck lead time in half-hours")
    p.add_argument("--fill_hours", type=float, default=2.0, help="truck capacity in hours of the route's mean flow")
    p.add_argument("--max_wait", type=int, default=4, help="half-hours a truck waits before leaving under-filled")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    cums = cumsums(panel)
    x = cums.x
    cal = np.load(ROLL / f"{a.name}_fold1.npz")  # earlier week: calibration
    ev = np.load(ROLL / f"{a.name}_fold0.npz")  # later week: evaluation
    o_cal, p_cal = cal["origins"], cal["pred"].astype(np.float64)
    o_ev, p_ev = ev["origins"], ev["pred"].astype(np.float64)
    y_cal, y_ev = truth(panel, o_cal), truth(panel, o_ev)
    # Route level known before the calibration week starts.
    level = x[:, o_cal[0] - 4 * WEEK + 1 : o_cal[0] + 1].mean(axis=1)
    res: dict = {"assumptions": {"lead_h": a.lead / 2, "capacity_hours_of_mean_flow": a.fill_hours,
                                 "max_wait_h": a.max_wait / 2}}

    # 1. Point accuracy from every half-hour of the evaluation week, and at the test clock time.
    ts = panel.times[o_ev]
    slot = (ts.hour * 2 + ts.minute // 30).to_numpy()
    last = panel.times[-1]
    at_test = slot == last.hour * 2 + last.minute // 30
    res["point"] = {"all_origins": score(p_ev, y_ev)[0], "at_10_30": score(p_ev[at_test], y_ev[at_test])[0],
                    "by_hour": {int(hh): score(p_ev[slot // 2 == hh], y_ev[slot // 2 == hh])[0] for hh in range(24)}}
    log.info("point: all origins %.4f, at 10:30 %.4f", res["point"]["all_origins"], res["point"]["at_10_30"])

    # 2. Intervals: quantiles from the earlier week applied to the later one.
    conf = Conformal().fit(p_cal, y_cal, level)
    lvl_ev = x[:, o_ev[0] - 4 * WEEK + 1 : o_ev[0] + 1].mean(axis=1)
    iv = {}
    for lo_p, hi_p in [(0.1, 0.9), (0.05, 0.95), (0.25, 0.75)]:
        lo, hi = conf.quantile(p_ev, lvl_ev, lo_p), conf.quantile(p_ev, lvl_ev, hi_p)
        iv[f"{round((hi_p - lo_p) * 100)}%"] = {
            "coverage": coverage(lo, hi, y_ev), "coverage_10_30": coverage(lo[at_test], hi[at_test], y_ev[at_test]),
            "rel_width": float((hi - lo).sum() / p_ev.sum()),
            "by_horizon": [coverage(lo[..., h], hi[..., h], y_ev[..., h]) for h in range(HORIZON)]}
    iv["upper_90_exceeded"] = float((y_ev > conf.quantile(p_ev, lvl_ev, 0.9)).mean())
    iv["pinball_p90"] = pinball(conf.quantile(p_ev, lvl_ev, 0.9), y_ev, 0.9)
    res["intervals"] = iv
    log.info("intervals: %s", {k: round(v["coverage"], 3) for k, v in iv.items() if isinstance(v, dict)})

    # 3. Truck calls on the evaluation week.
    L = a.lead
    cap = np.maximum(a.fill_hours * 2 * level, 1.0)  # route-specific: so many hours of its average flow
    f_ev = inflow_forecast(p_ev, x[:, o_ev].T, L)
    a_ev = inflow_actual(x, o_ev, L)
    res["inflow_wape"] = float(np.abs(f_ev - a_ev).sum() / a_ev.sum())
    rules = call_rules(x, o_cal, p_cal, o_ev, p_ev, level, lvl_ev, L)
    days = len(o_ev) / DAY
    sim = {}
    for name, exp in rules.items():
        sim[name] = simulate(x, o_ev, exp, cap, L, a.max_wait).per_day(days)
        log.info("%-20s %s", name, {k: round(v, 2) for k, v in sim[name].items()})
    # Trade-off curves: scaling the expectation moves a rule between early and late trucks.
    curves = {}
    for name in ("last_2h_pace", "same_time_last_week", "model_p50"):
        curves[name] = [{"k": k, **simulate(x, o_ev, rules[name] * k, cap, L, a.max_wait).per_day(days)}
                        for k in (0.5, 0.75, 1.0, 1.25, 1.5)]
    # The data gives no truck sizes or lead times: the ranking must hold across plausible values.
    sens = []
    for lead in (1, 2, 4):
        rl = call_rules(x, o_cal, p_cal, o_ev, p_ev, level, lvl_ev, lead)
        for fill in (1.0, 2.0, 4.0):
            c = np.maximum(fill * 2 * level, 1.0)
            row = {"lead_h": lead / 2, "fill_hours": fill}
            for name in ("reactive", "last_2h_pace", "same_time_last_week", "model_p50", "oracle"):
                r = simulate(x, o_ev, rl[name], c, lead, a.max_wait)
                row[name] = 60 * (r.idle_h + r.late_h) / max(r.trucks, 1)  # lost minutes per truck
            sens.append(row)
            log.info("sensitivity %s", {k: round(v, 1) for k, v in row.items()})
    res["dispatch"] = {"per_day": sim, "curves": curves, "sensitivity": sens}

    # 4. Intraday level check: forecasts made 4 hours ago vs what has been shipped since.
    res["intraday"] = intraday_correction(p_cal, y_cal, o_cal, p_ev, y_ev, o_ev, at_test)
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("saved %s", OUT / f"{a.name}.json")


def call_rules(x, o_cal, p_cal, o_ev, p_ev, level, lvl_ev, L: int) -> dict[str, np.ndarray]:
    """Expected inflow over the next L steps under each rule, at every evaluation origin."""
    f_cal = inflow_forecast(p_cal, x[:, o_cal].T, L)
    a_cal = inflow_actual(x, o_cal, L)
    f_ev = inflow_forecast(p_ev, x[:, o_ev].T, L)
    conf_in = Conformal().fit(f_cal[:, :, None], a_cal[:, :, None], level)

    def q_in(p: float) -> np.ndarray:
        return conf_in.quantile(f_ev[:, :, None], lvl_ev, p)[:, :, 0]

    rate = np.stack([x[:, o - 3 : o + 1].mean(axis=1) for o in o_ev]) * L  # last 2 hours' pace
    seas = np.stack([np.mean([x[:, o + 1 - k * DAY : o + 1 + L - k * DAY].sum(axis=1) for k in range(1, 8)], axis=0)
                     for o in o_ev])  # same clock time over the last week
    return {"reactive": np.zeros_like(f_ev), "last_2h_pace": rate, "same_time_last_week": seas,
            "model_p25": q_in(0.25), "model_p50": q_in(0.5), "model_p75": q_in(0.75),
            "oracle": inflow_actual(x, o_ev, L)}


def intraday_correction(p_cal, y_cal, o_cal, p_ev, y_ev, o_ev, at_test) -> dict:
    """Does the network-wide error of the last 4 hours predict the next 4 hours' level?

    At origin o the forecasts issued at o-8 for o-7..o are fully observed; their total ratio
    rho = actual / forecast scales the new forecast as rho ** alpha. Alpha is chosen on the
    calibration week only.
    """

    def ratios(pred, y, origins):
        idx = {o: i for i, o in enumerate(origins)}
        rho = np.full(len(origins), np.nan)
        for i, o in enumerate(origins):
            j = idx.get(o - HORIZON)
            if j is not None:
                rho[i] = y[j].sum() / pred[j].sum()
        return rho

    def apply(pred, rho, alpha):
        k = np.where(np.isfinite(rho), rho, 1.0) ** alpha
        return pred * k[:, None, None]

    r_cal, r_ev = ratios(p_cal, y_cal, o_cal), ratios(p_ev, y_ev, o_ev)
    ok_cal, ok_ev = np.isfinite(r_cal), np.isfinite(r_ev)
    alphas = np.round(np.arange(0, 1.01, 0.1), 1)
    cal_scores = {float(al): score(apply(p_cal[ok_cal], r_cal[ok_cal], al), y_cal[ok_cal])[0] for al in alphas}
    best = min(cal_scores, key=cal_scores.get)
    m = ok_ev
    t = ok_ev & at_test
    return {"alpha_cal": best, "cal_scores": cal_scores,
            "eval_all": {"off": score(p_ev[m], y_ev[m]), "on": score(apply(p_ev[m], r_ev[m], best), y_ev[m])},
            "eval_10_30": {"off": score(p_ev[t], y_ev[t]), "on": score(apply(p_ev[t], r_ev[t], best), y_ev[t])}}


if __name__ == "__main__":
    main()
