"""Backtest the GPU MLP forecaster; same folds and outputs as scripts/experiment.py.

    python scripts/experiment_nn.py --name mlp_a --epochs 6
    python scripts/experiment_nn.py --name mlp_a --final
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

from rwb.backtest import default_folds, run, score_origins
from rwb.data import ROOT, load_panel, make_submission
from rwb.features import cumsums
from rwb.nn import NNConfig, nn_fit_predict

log = logging.getLogger("experiment_nn")
OOF = ROOT / "models/oof"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="mlp")
    p.add_argument("--only_final", action="store_true", help="skip the backtest, just fit on all data and forecast")
    p.add_argument("--final", action="store_true")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--seeds", type=int, default=1, help="average this many seeds")
    for f, v in asdict(NNConfig()).items():
        p.add_argument(f"--{f}", type=type(v), default=v)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    panel = load_panel()
    cums = cumsums(panel)
    folds = default_folds(panel, n_folds=a.folds)
    base_cfg = NNConfig(**{k: getattr(a, k) for k in asdict(NNConfig())})
    log.info("config %s seeds=%d", base_cfg, a.seeds)

    def fit(cutoff: int, origins: np.ndarray) -> np.ndarray:
        preds = [nn_fit_predict(panel, cums, cutoff, origins, NNConfig(**{**asdict(base_cfg), "seed": base_cfg.seed + s}))
                 for s in range(a.seeds)]
        return np.mean(preds, axis=0)

    t0 = time.time()
    if a.only_final:
        a.final = True
    res = None if a.only_final else run(panel, folds, lambda fold: fit(fold.label_cutoff, fold.val_origins))
    if res is not None:
        summ = res.summary()
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
            fh.write(json.dumps({"name": a.name, "cfg": asdict(base_cfg), "seeds": a.seeds, "summary": summ,
                                 "best_scale": best_k, "scaled": cal[best_k], "sec": round(time.time() - t0)}) + "\n")

    if a.final:
        last = panel.n_times - 1
        pred = fit(last, np.array([last]))[0]
        np.save(OOF / f"{a.name}_test.npy", pred)
        sub = make_submission(panel, pred)
        out = ROOT / f"submissions/{a.name}.csv"
        sub.to_csv(out, index=False)
        log.info("submission %s (unscaled, sum=%.0f)", out, sub["y_pred"].sum())


if __name__ == "__main__":
    main()
