"""Backtest a forecaster on 10:30 origins and optionally produce a test forecast.

Examples:
    python scripts/experiment.py --naive
    python scripts/experiment.py --name lgbm_base --lib lgbm --days 56 --stride 2
    python scripts/experiment.py --name lgbm_base --lib lgbm --final
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.backtest import default_folds, run, score_origins, truth_for
from rwb.data import ROOT, load_panel, make_submission
from rwb.features import cumsums
from rwb.models import GBMConfig, gbm_fit_predict, naive

log = logging.getLogger("experiment")
OOF = ROOT / "models/oof"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--naive", action="store_true")
    p.add_argument("--name", default="exp")
    p.add_argument("--only_final", action="store_true", help="skip the backtest, just fit on all data and forecast")
    p.add_argument("--final", action="store_true", help="also fit on all data and forecast the test")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--extra", default="{}", help="JSON with extra library params")
    for f, v in asdict(GBMConfig()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{f}", type=lambda x: x.lower() in {"1", "true", "yes"}, default=v)
        elif f != "extra":
            p.add_argument(f"--{f}", type=type(v), default=v)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    panel = load_panel()
    cums = cumsums(panel)
    folds = default_folds(panel, n_folds=a.folds)

    if a.naive:
        for kind in ["last", "day", "week", "slot7", "base"]:
            scores = []
            for fold in folds:
                pred = naive(panel, fold.val_origins, kind, cums)
                scores += score_origins(pred, truth_for(panel, fold.val_origins))
            log.info("naive %-6s mean=%.4f  per-origin=%s", kind, np.mean(scores), np.round(scores, 3).tolist())
        return

    cfg = GBMConfig(**{k: getattr(a, k) for k in asdict(GBMConfig()) if k != "extra"}, extra=json.loads(a.extra))
    log.info("config %s", cfg)
    t0 = time.time()
    if a.only_final:
        a.final = True
    res = None if a.only_final else run(panel, folds, lambda fold: gbm_fit_predict(panel, cums, fold.label_cutoff, fold.val_origins, cfg))
    if res is not None:
        summ = res.summary()
        # Global multiplicative calibration: report how much a pooled scale would help.
        pred_all = np.concatenate(list(res.preds.values()))
        truth_all = np.concatenate(list(res.truth.values()))
        grid = np.round(np.arange(0.9, 1.15, 0.01), 2)
        cal = {float(k): float(np.mean(score_origins(pred_all * k, truth_all))) for k in grid}
        best_k = min(cal, key=cal.get)
        log.info("RESULT %s %s | best scale %.2f -> %.4f | %.0fs", a.name, {k: round(v, 4) for k, v in summ.items()},
                 best_k, cal[best_k], time.time() - t0)

        OOF.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OOF / f"{a.name}.npz", **{f"pred_{k}": v for k, v in res.preds.items()},
                            **{f"truth_{k}": v for k, v in res.truth.items()})
        with (ROOT / "models/results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"name": a.name, "cfg": asdict(cfg), "summary": summ, "best_scale": best_k,
                                 "scaled": cal[best_k], "sec": round(time.time() - t0)}) + "\n")

    if a.final:
        last = panel.n_times - 1
        pred = gbm_fit_predict(panel, cums, last, np.array([last]), cfg)[0]
        np.save(OOF / f"{a.name}_test.npy", pred)
        sub = make_submission(panel, pred)
        out = ROOT / f"submissions/{a.name}.csv"
        sub.to_csv(out, index=False)
        log.info("submission %s (unscaled, sum=%.0f)", out, sub["y_pred"].sum())


if __name__ == "__main__":
    main()
