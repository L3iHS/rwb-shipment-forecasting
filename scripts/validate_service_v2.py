"""Forward evaluation of v1 and reconciled flow on archived rolling forecasts.

Both methods are recalibrated with purged boundary labels. No model training or file upload.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import logging
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwb.data import HORIZON, load_panel
from rwb.features import DAY, cumsums
from rwb.flows import FLOW_METHOD, legacy_cumulative_inflow, reconcile_hourly
from rwb.intervals import Conformal, coverage, size_groups
from rwb.dispatch import simulate
from rwb.model_audit import summarize, available_calibration_mask
from compare_service_models import sha256, write_json

log = logging.getLogger("service_validation")


def run(output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh result directory")
    output.mkdir(parents=True, exist_ok=True)
    files = [*sorted((ROOT / "src/rwb").glob("*.py")), Path(__file__),
             ROOT / "models/rolling/lean_fold0.npz", ROOT / "models/rolling/lean_fold1.npz",
             ROOT / "data/raw/train_solo_track.parquet", ROOT / "docs/next_stage_plan.md"]
    manifest = {"status": "planned", "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in files},
        "plan": {"fit": "fold1, retaining only origins whose final label is known at first evaluation origin",
            "evaluate": "fold0", "weights": "fixed equal least-squares weights", "quantile_projection": "sort horizons",
            "methods": ["legacy_v1", FLOW_METHOD], "dispatch": "lead 1/2/4 steps; capacity 1/2/4 hours of mean flow; max_wait 4 steps",
            "independent_holdout": False, "future_model_training": False,
            "acceptance": "physical invariants mandatory; quantify WAPE, coverage and simulation changes without claiming uniform improvement"}}
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    panel = load_panel(cache=False)
    cums = cumsums(panel)
    with np.load(ROOT / "models/rolling/lean_fold1.npz", allow_pickle=False) as z:
        oc, pc = z["origins"], z["pred"].astype(float)
        cutoff_c = int(z["label_cutoff"])
    with np.load(ROOT / "models/rolling/lean_fold0.npz", allow_pickle=False) as z:
        oe, pe = z["origins"], z["pred"].astype(float)
        cutoff_e = int(z["label_cutoff"])
    keep = available_calibration_mask(oc, int(oe.min()), HORIZON)
    purged = int((~keep).sum())
    oc, pc = oc[keep], pc[keep]
    if cutoff_c > oc.min() or cutoff_e > oe.min() or oc.max() + HORIZON > oe.min() or oe.max() + HORIZON >= panel.n_times:
        raise ValueError("Rolling archive labels are not available in chronological order")
    yc = np.stack([panel.y[:, o + 1:o + 9] for o in oc])
    ye = np.stack([panel.y[:, o + 1:o + 9] for o in oe])
    ac = np.stack([np.cumsum(cums.x[:, o + 1:o + 9], axis=1) for o in oc])
    ae = np.stack([np.cumsum(cums.x[:, o + 1:o + 9], axis=1) for o in oe])
    level = cums.x[:, oc[0] - 28 * DAY + 1:oc[0] + 1].mean(axis=1)
    groups, _ = size_groups(level)
    at_test = (panel.times[oe].hour == 10) & (panel.times[oe].minute == 30)
    point_conf = Conformal().fit(pc, yc, level)
    lo, hi = point_conf.quantile(pe, level, .1), point_conf.quantile(pe, level, .9)
    point_report = {"all_origins": summarize(pe, ye)["mean_score"], "at_10_30": summarize(pe[at_test], ye[at_test])["mean_score"]}
    hourly_intervals = {"80%": {"coverage": coverage(lo, hi, ye), "coverage_10_30": coverage(lo[at_test], hi[at_test], ye[at_test])}}
    net_level = np.array([level.sum()])
    net_hourly = Conformal().fit(pc.sum(axis=1, keepdims=True), yc.sum(axis=1, keepdims=True), net_level)
    net_pred = pe.sum(axis=1, keepdims=True)
    network_hourly_coverage = coverage(net_hourly.quantile(net_pred, net_level, .1),
                                      net_hourly.quantile(net_pred, net_level, .9), ye.sum(axis=1, keepdims=True))
    timing = {}
    log.info("Purged %d boundary origins; calibration %d, evaluation %d", purged, len(oc), len(oe))
    flow_start = time.perf_counter()
    rc, re = reconcile_hourly(pc, cums.x[:, oc].T), reconcile_hourly(pe, cums.x[:, oe].T)
    timing["reconciliation_calibration_and_evaluation_seconds"] = time.perf_counter() - flow_start
    log.info("Reconciled %d route/origin paths", (len(oc) + len(oe)) * panel.n_routes)
    methods = {"legacy_v1": (legacy_cumulative_inflow(pc, cums.x[:, oc].T), legacy_cumulative_inflow(pe, cums.x[:, oe].T)),
               FLOW_METHOD: (rc.cumulative, re.cumulative)}
    interval_rows, summaries, sensitivity, daily = [], [], [], []
    for method, (fc, fe) in methods.items():
        conf = Conformal().fit(fc, ac, level)
        quantile = conf.cumulative_quantile if method == FLOW_METHOD else conf.quantile
        low, med, high = [quantile(fe, level, p) for p in (.1, .5, .9)]
        record = {"method": method, "wape_all_horizons": float(np.abs(fe - ae).sum() / ae.sum()),
            "coverage_80": coverage(low, high, ae), "relative_band_width": float((high - low).sum() / ae.sum()),
            "negative_flows": int((np.diff(np.concatenate([np.zeros_like(fe[..., :1]), fe], axis=-1), axis=-1) < -1e-8).sum()),
            "decreasing_point": int((np.diff(fe, axis=-1) < -1e-8).sum()),
            "decreasing_median": int((np.diff(med, axis=-1) < -1e-8).sum()),
            "quantile_crossings": int(((low > med) | (med > high)).sum())}
        net_conf = Conformal().fit(fc.sum(axis=1, keepdims=True), ac.sum(axis=1, keepdims=True), net_level)
        net_quantile = net_conf.cumulative_quantile if method == FLOW_METHOD else net_conf.quantile
        net_flow = fe.sum(axis=1, keepdims=True)
        record["network_coverage_80"] = coverage(net_quantile(net_flow, net_level, .1),
            net_quantile(net_flow, net_level, .9), ae.sum(axis=1, keepdims=True))
        summaries.append(record)
        for h in range(HORIZON):
            for group in range(5):
                mask = groups == group
                interval_rows.append({"method": method, "horizon_minutes": 30 * (h + 1), "size_group": group,
                    "coverage_80": coverage(low[:, mask, h], high[:, mask, h], ae[:, mask, h]),
                    "wape": float(np.abs(fe[:, mask, h] - ae[:, mask, h]).sum() / ae[:, mask, h].sum())})
        for hour in range(24):
            mask = panel.times[oe].hour == hour
            interval_rows.append({"method": method, "hour": hour, "coverage_80": coverage(low[mask], high[mask], ae[mask])})
        model_daily = {}
        for lead in (1, 2, 4):
            rate = np.stack([cums.x[:, o - 3:o + 1].mean(axis=1) for o in oe]) * lead
            seasonal = np.stack([np.mean([cums.x[:, o + 1 - k * DAY:o + 1 + lead - k * DAY].sum(axis=1)
                                          for k in range(1, 8)], axis=0) for o in oe])
            rules = {"reactive": np.zeros_like(rate), "last_2h_pace": rate, "same_time_last_week": seasonal,
                     "model_p50": med[..., lead - 1], "oracle": ae[..., lead - 1]}
            for fill in (1., 2., 4.):
                cap = np.maximum(fill * 2 * level, 1.)
                row = {"method": method, "lead_h": lead / 2, "fill_hours": fill}
                for name, expected in rules.items():
                    result = simulate(cums.x, oe, expected, cap, lead, max_wait=4)
                    row[name] = 60 * (result.idle_h + result.late_h) / max(result.trucks, 1)
                    if lead == 2 and fill == 2:
                        per_day = result.per_day(len(oe) / DAY)
                        daily.append({"method": method, "rule": name, **per_day})
                        model_daily[name] = per_day
                sensitivity.append(row)
        record["dispatch_lost_minutes_per_truck"] = next(r["model_p50"] for r in sensitivity if r["method"] == method and r["lead_h"] == 1 and r["fill_hours"] == 2)
        log.info("%s cumulative WAPE %.5f, band coverage %.4f, lost minutes %.2f", method,
                 record["wape_all_horizons"], record["coverage_80"], record["dispatch_lost_minutes_per_truck"])
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(output / "intervals.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "sensitivity.csv", index=False)
    pd.DataFrame(daily).to_csv(output / "dispatch_per_day.csv", index=False)
    v2 = next(r for r in summaries if r["method"] == FLOW_METHOD)
    if v2["negative_flows"] or v2["quantile_crossings"] or v2["decreasing_median"]:
        raise ValueError("Physical output invariants failed")
    report = {"schema_version": 2, "flow_method": FLOW_METHOD, "point": point_report, "intervals": hourly_intervals,
        "network_hourly_coverage_80": network_hourly_coverage,
        "dispatch": {"sensitivity": [{k: v for k, v in r.items() if k != "method"} for r in sensitivity if r["method"] == FLOW_METHOD],
            "per_day": {r["rule"]: {k:v for k,v in r.items() if k not in ("method", "rule")} for r in daily if r["method"] == FLOW_METHOD}},
        "comparison": summaries, "purged_calibration_origins": purged,
        "calibration_end_label": str(panel.times[oc.max() + HORIZON]), "evaluation_start": str(panel.times[oe.min()]),
        "adjustment_relative_to_hourly_total": float(np.abs(re.hourly_adjustment).sum() / pe.sum()),
        "limitations": ["Previously reviewed periods, no independent holdout", "Simulated capacity, dock state and lead time",
            "Half-hour ground truth is reconstructed from hourly observations", "Marginal bands are not joint path guarantees"],
        "timing": {**timing, "total_seconds": time.perf_counter() - started}}
    write_json(output / "report.json", report)
    manifest["status"] = "complete"
    manifest["outputs"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/colhound/service_v2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.output.resolve())
