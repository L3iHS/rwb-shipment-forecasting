"""Reproduce archived ML comparisons with provenance and forward-only calibration.

No training, submission changes, or network calls. A fresh output directory is required.
Run: python scripts/compare_service_models.py --output reports/colhound/ml
Optional --benchmark measures the saved CPU bundle on original data, without the panel cache.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwb.backtest import default_folds, truth_for
from rwb.data import HORIZON, STEP, Panel
from rwb.forecaster import LEAN
from rwb.model_audit import check_chronology, check_truth, choose_scale, per_origin, summarize
from rwb.models import GBMConfig

log = logging.getLogger("model_audit")
NAMES = ("lgbm_huber", "lgbm_huber_blr", "lgbm_tweedie")
SCALE_GRID = np.round(np.arange(0.95, 1.1001, 0.0025), 4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()


def target_panel() -> Panel:
    frame = pd.read_parquet(ROOT / "data/raw/train_solo_track.parquet", columns=["route_id", "timestamp", "target_1h"])
    if frame.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Duplicate raw route/timestamp")
    table = frame.pivot(index="route_id", columns="timestamp", values="target_1h").sort_index().sort_index(axis=1)
    times = pd.DatetimeIndex(table.columns)
    if not times.equals(pd.date_range(times[0], times[-1], freq=STEP)) or table.isna().any().any():
        raise ValueError("Raw target grid must be complete and regular")
    # Statuses are deliberately absent from this scoring-only panel.
    return Panel(times, table.index.to_numpy(), table.to_numpy(dtype=np.float64), np.empty((0,)), pd.DataFrame())


def normalized_config(cfg: dict) -> dict:
    full = asdict(GBMConfig(**cfg))
    full["extra"].pop("num_threads", None)
    return full


def configuration_audit() -> dict:
    bundle = json.loads((ROOT / "models/prod/service/model/forecaster.json").read_text())
    records = [json.loads(line) for line in (ROOT / "models/results.jsonl").read_text().splitlines() if line.strip()]
    record = next(r for r in reversed(records) if r["name"] == "lgbm_huber_blr")
    expected = normalized_config(asdict(LEAN))
    deployed = normalized_config(bundle["cfg"])
    archived = normalized_config(record["cfg"])
    return {
        "lean_matches_saved_bundle": expected == deployed,
        "lean_matches_oof_log_excluding_thread_count": expected == archived,
        "runtime_threads_excluded_from_comparison": True,
        "bundle_config": bundle,
        "oof_log_record": record,
        "limitation": "OOF files have no embedded config or route IDs; log association is not a cryptographic training audit.",
    }


def benchmark(threads: int) -> dict:
    from rwb.data import load_panel
    from rwb.features import cumsums
    from rwb.forecaster import Forecaster
    from rwb.service import Service

    start = time.perf_counter()
    panel = load_panel(cache=False)
    raw_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cums = cumsums(panel)
    cums_seconds = time.perf_counter() - start
    start = time.perf_counter()
    svc = Service.load(ROOT / "models/prod/service_v2")
    for model in svc.forecaster.models:
        # Changing native parameters on a file-loaded Booster crashes some Windows builds.
        # Supply the inference setting to predict, leaving the trained model untouched.
        model.predict = partial(model.predict, num_threads=threads)
    load_seconds = time.perf_counter() - start
    origin = panel.n_times - 1
    if svc.forecaster.label_cutoff != origin:
        raise ValueError("Bundle cutoff does not match the raw-data endpoint")
    times, pred = [], None
    for _ in range(3):
        start = time.perf_counter()
        pred = svc.forecaster.predict(panel, cums, np.array([origin]))[0]
        times.append(time.perf_counter() - start)
    saved = np.load(ROOT / "models/rolling/lean_test.npy", allow_pickle=False)
    diff = float(np.abs(pred.astype(float) - saved.astype(float)).max())
    consistent = bool(np.allclose(pred, saved, rtol=1e-5, atol=0.05))
    start = time.perf_counter()
    reloaded = Forecaster.load(ROOT / "models/prod/service/model")
    for model in reloaded.models:
        model.predict = partial(model.predict, num_threads=threads)
    reload_pred = reloaded.predict(panel, cums, np.array([origin]))[0]
    reload_seconds = time.perf_counter() - start
    np.testing.assert_array_equal(pred, reload_pred)
    start = time.perf_counter()
    fc = svc.forecast(panel, cums, origin)
    forecast_seconds = time.perf_counter() - start
    start = time.perf_counter()
    dock = svc.dock_state(panel, cums, origin)
    dock_seconds = time.perf_counter() - start
    start = time.perf_counter()
    plan = svc.plan(panel, origin, fc, dock)
    plan_seconds = time.perf_counter() - start
    if len(plan) != panel.n_routes:
        raise ValueError("Incomplete route plan")
    usage = {"peak_rss_bytes": None, "measurement": "unavailable"}
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            usage = {"peak_rss_bytes": counter.PeakWorkingSetSize, "measurement": "Windows PeakWorkingSetSize, entire audit process"}
    return {
        "inference_threads": threads, "raw_load_no_cache_seconds": raw_seconds, "cumsums_seconds": cums_seconds,
        "bundle_load_seconds": load_seconds, "prediction_seconds_3_runs": times,
        "reload_and_predict_seconds": reload_seconds, "reload_predictions_identical": True,
        "matches_archived_test_forecast": consistent, "archived_test_max_abs_difference": diff,
        "service_forecast_seconds": forecast_seconds, "dock_replay_seconds": dock_seconds,
        "plan_seconds": plan_seconds, "cold_pipeline_components_seconds": raw_seconds + cums_seconds + load_seconds + forecast_seconds + dock_seconds + plan_seconds,
        "bundle_bytes": sum(p.stat().st_size for p in (ROOT / "models/prod/service_v2").rglob("*") if p.is_file()),
        "model_count": len(svc.forecaster.models), "memory": usage,
        "scope": "One local dataset/snapshot; includes simulated dock state, excludes HTTP and LLM. Full-audit peak is not isolated inference RAM.",
        "service_semantics": {
            "point_outside_p10_p90_count": int(((fc["point_forecast"] < fc["p10"]) | (fc["point_forecast"] > fc["p90"])).sum()),
            "decreasing_cumulative_inflow_count": int((np.diff(fc["inflow_point"], axis=1) < -1e-8).sum()),
        },
    }


def run(output: Path, do_benchmark: bool, threads: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; prior evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    ensemble = json.loads((ROOT / "submissions/ensemble_v2.json").read_text())
    sources = [ROOT / f"models/oof/{name}.npz" for name in (*NAMES, *ensemble["models"])]
    sources += [ROOT / "models/oof/lgbm_huber_blr3.npz", ROOT / "models/results.jsonl",
        ROOT / "submissions/ensemble_v2.json", ROOT / "data/raw/train_solo_track.parquet", ROOT / "data/raw/test_solo_track.parquet"]
    sources += list((ROOT / "models/prod/service").rglob("*"))
    sources += list((ROOT / "models/prod/service_v2").rglob("*"))
    sources += [ROOT / "models/rolling/lean_test.npy", ROOT / "docs/validation_windows.md"]
    code = [Path(__file__), *sorted((ROOT / "src/rwb").glob("*.py")), ROOT / "scripts/ensemble.py"]
    manifest = {
        "status": "planned", "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "working_tree": git("status", "--short"),
        "argv": sys.argv, "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "lightgbm", "pyarrow"]},
        "files": {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in sorted(set(sources + code)) if p.is_file()},
        "plan": {"calibration_fold": "fold1", "evaluation_fold": "fold0", "calibration_dates": ["2025-10-18", "2025-10-24"],
            "evaluation_dates": ["2025-10-25", "2025-10-31"], "origin_time": "10:30", "service_score_tolerance": 0.002,
            "scale_grid": SCALE_GRID.tolist(), "new_holdout": False, "forecast_training": "none; archived predictions only",
            "forward_ensemble": "refit weights on fold1 only with existing ensemble.fit_weights; no per-horizon scale",
            "benchmark": do_benchmark, "benchmark_threads": threads},
        "limitations": ["Both October weeks were used in prior model development.",
            "OOF dates/route ordering are inferred from default_folds and validated against raw truth.",
            "Forward calibration tests parameter chronology, not an untouched model selection holdout."]}
    write_json(output / "manifest.json", manifest)
    log.info("Manifest recorded before scoring")
    panel = target_panel()
    folds = {f.name: f for f in default_folds(panel, n_folds=3)}
    cfg_audit = configuration_audit()
    if not cfg_audit["lean_matches_saved_bundle"] or not cfg_audit["lean_matches_oof_log_excluding_thread_count"]:
        raise ValueError("Service configurations differ; inspect before comparison")
    write_json(output / "configuration_audit.json", cfg_audit)
    predictions, truth = {}, {}
    for name in (*NAMES, *ensemble["models"], "lgbm_huber_blr3"):
        if name in predictions:
            continue
        predictions[name] = {}
        with np.load(ROOT / f"models/oof/{name}.npz", allow_pickle=False) as archive:
            for fold_name, fold in folds.items():
                key = f"pred_{fold_name}"
                if key not in archive:
                    continue
                actual = truth_for(panel, fold.val_origins)
                check_truth(archive[f"truth_{fold_name}"], actual, f"{name}/{fold_name}")
                predictions[name][fold_name] = archive[key].astype(np.float64)
                truth[fold_name] = actual
    rows, horizons, summaries = [], [], []
    def record(name, fold_name, pred, mode):
        actual = truth[fold_name]
        metrics = per_origin(pred, actual)
        for i, origin in enumerate(folds[fold_name].val_origins):
            rows.append({"model": name, "mode": mode, "fold": fold_name, "origin": str(panel.times[origin]),
                **{key: float(v[i]) for key, v in metrics.items()}})
        summaries.append({"model": name, "mode": mode, "fold": fold_name, "origins": len(pred), **summarize(pred, actual)})
        for h in range(HORIZON):
            horizons.append({"model": name, "mode": mode, "fold": fold_name, "horizon_minutes": (h + 1) * 30,
                **summarize(pred[:, :, h:h+1], actual[:, :, h:h+1])})
    for name in NAMES:
        for fold_name, pred in predictions[name].items():
            record(name, fold_name, pred, "raw")
    record("lgbm_huber_blr", "fold2", predictions["lgbm_huber_blr3"]["fold2"], "raw_additional_reviewed_week")
    for fold_name in ("fold1", "fold0"):
        pred = sum(predictions[name][fold_name] * weight for name, weight in zip(ensemble["models"], ensemble["weights"]))
        pred *= np.asarray(ensemble["h_scale"])
        record("ensemble_v2", fold_name, pred, "final_weights_development")
    cal, ev = "fold1", "fold0"
    check_chronology(panel.times[folds[cal].val_origins], panel.times[folds[ev].val_origins])
    calibration = {}
    for name in NAMES:
        scale = choose_scale(predictions[name][cal], truth[cal], SCALE_GRID)
        calibration[name] = {"scale": scale, "fit_fold": cal, "evaluate_fold": ev}
        record(name, ev, predictions[name][ev] * scale, "forward_scale")
    log.info("Refitting ensemble weights on the earlier week only")
    from ensemble import fit_weights
    fit_start = time.perf_counter()
    P = np.stack([predictions[name][cal] for name in ensemble["models"]], axis=-1)
    weights, hs = fit_weights(P, truth[cal], per_h_scale=False)
    elapsed = time.perf_counter() - fit_start
    calibration["ensemble_v2"] = {"weights": weights.tolist(), "h_scale": hs.tolist(), "fit_fold": cal, "evaluate_fold": ev,
        "weight_fit_seconds": elapsed}
    forward_pred = sum(predictions[name][ev] * weight for name, weight in zip(ensemble["models"], weights)) * hs
    record("ensemble_v2", ev, forward_pred, "forward_weights")
    write_json(output / "calibration.json", calibration)
    df = pd.DataFrame(rows)
    df.to_csv(output / "per_origin.csv", index=False)
    pd.DataFrame(horizons).to_csv(output / "by_horizon.csv", index=False)
    # Summaries for all available dates retain each model's evaluation mode.
    for (name, mode), group in df.groupby(["model", "mode"]):
        summaries.append({"model": name, "mode": mode, "fold": "all_available", "origins": len(group),
            **{f"mean_{key}": float(group[key].mean()) for key in ["score", "wape", "signed_bias", "abs_bias"]},
            "worst_score": float(group.score.max()), "pooled_score": None})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    # Paired comparison uses raw service output and chronologically refitted ensemble.
    service_scores = per_origin(predictions["lgbm_huber_blr"][ev], truth[ev])["score"]
    ensemble_scores = per_origin(forward_pred, truth[ev])["score"]
    delta = service_scores - ensemble_scores
    pd.DataFrame({"origin": panel.times[folds[ev].val_origins], "service_score": service_scores,
        "forward_ensemble_score": ensemble_scores, "service_minus_ensemble": delta}).to_csv(output / "paired.csv", index=False)
    scoring_seconds = time.perf_counter() - start
    runtime = {"benchmark": benchmark(threads) if do_benchmark else None,
        "archive_audit_seconds": scoring_seconds, "total_audit_seconds": time.perf_counter() - start,
        "forecast_models_retrained": False}
    write_json(output / "runtime.json", runtime)
    decision = {
        "status": "provisional_service_candidate", "service_model": "lgbm_huber_blr", "competition_model": "ensemble_v2",
        "forward_service_minus_ensemble_mean": float(delta.mean()), "service_wins_of_7": int((delta < 0).sum()),
        "within_engineering_tolerance": bool(delta.mean() <= 0.002),
        "limitations": manifest["limitations"], "production_ready": False}
    write_json(output / "decision.json", decision)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", manifest)
    log.info("Completed: mean service minus forward ensemble %.6f; outputs %s", delta.mean(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/colhound/ml")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(args.output.resolve(), args.benchmark, args.threads)
