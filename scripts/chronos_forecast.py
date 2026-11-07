"""Chronos-2 zero-shot forecasts saved in the backtest OOF format, as an ensemble member.

Runs in the separate AutoGluon venv (.venv-ag, Python 3.13): chronos-forecasting pins
pandas<2.4 and does not support Python 3.14, so it cannot import src/rwb. The panel,
origins and metric are therefore rebuilt here to mirror rwb.data / rwb.backtest.

    .venv-ag\\Scripts\\python scripts/chronos_forecast.py --context 2048
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "models/oof"
H = 8
STEP = pd.Timedelta(minutes=30)
log = logging.getLogger("chronos_forecast")
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_panel() -> tuple[np.ndarray, pd.DatetimeIndex]:
    df = pd.read_parquet(ROOT / "data/raw/train_solo_track.parquet", columns=["route_id", "timestamp", "target_1h"])
    routes = np.sort(df["route_id"].unique())
    times = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=STEP)
    y = np.full((len(routes), len(times)), np.nan)
    r = np.searchsorted(routes, df["route_id"].to_numpy())
    t = ((df["timestamp"] - times[0]) / STEP).to_numpy().astype(np.int64)
    y[r, t] = df["target_1h"].to_numpy(np.float64)
    return y, times


def score(truth: np.ndarray, pred: np.ndarray) -> float:
    s = truth.sum()
    return float(np.abs(truth - pred).sum() / s + abs((pred - truth).sum()) / s)


def forecast(pipe: Chronos2Pipeline, y: np.ndarray, origins: list[int], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Returns (median, quantile-average) of shape (n_origins, R, H)."""
    med, avg = [], []
    for o in origins:
        log.info("origin %d", o)
        ctx = torch.tensor(y[:, max(0, o + 1 - args.context) : o + 1], dtype=torch.float32)
        q, _ = pipe.predict_quantiles(
            list(ctx), prediction_length=H, quantile_levels=QUANTILES,
            batch_size=args.batch_size, context_length=args.context, cross_learning=args.cross_learning,
        )
        q = torch.stack(q).squeeze(1).double().numpy()  # (R, H, Q)
        med.append(q[..., QUANTILES.index(0.5)])
        # Chronos-2 has no sampled mean; averaging evenly spaced quantiles approximates it.
        avg.append(q.mean(-1))
    return np.clip(np.stack(med), 0, None), np.clip(np.stack(avg), 0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="autogluon/chronos-2")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--cross-learning", action="store_true")
    ap.add_argument("--name", default="chronos2")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.35)
    ap.add_argument("--out-dir", type=Path, default=OOF)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Other training jobs share this machine.
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)

    y, times = load_panel()
    assert not np.isnan(y).any(), "grid has gaps"
    last = y.shape[1] - 1
    folds = {
        "fold0": sorted(last - 48 * j for j in range(1, 8)),
        "fold1": sorted(last - 48 * j for j in range(8, 15)),
    }
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=device, torch_dtype=torch.float32)

    t0 = time.time()
    med_out, avg_out, report = {}, {}, {}
    for name, origins in folds.items():
        med, avg = forecast(pipe, y, origins, args)
        truth = np.stack([y[:, o + 1 : o + 1 + H] for o in origins])
        med_out[f"pred_{name}"], avg_out[f"pred_{name}"] = med, avg
        med_out[f"truth_{name}"] = avg_out[f"truth_{name}"] = truth
        report[name] = {
            "origins": [str(times[o]) for o in origins],
            "median": [score(truth[i], med[i]) for i in range(len(origins))],
            "mean": [score(truth[i], avg[i]) for i in range(len(origins))],
        }
    test_med, test_avg = forecast(pipe, y, [last], args)
    elapsed = time.time() - t0

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.name}.npz", **med_out)
    np.savez(out / f"{args.name}_mean.npz", **avg_out)
    np.save(out / f"{args.name}_test.npy", test_med[0])
    np.save(out / f"{args.name}_mean_test.npy", test_avg[0])

    for kind in ("median", "mean"):
        per_fold = {f: float(np.mean(report[f][kind])) for f in folds}
        overall = float(np.mean(report["fold0"][kind] + report["fold1"][kind]))
        log.info("%-6s overall=%.4f %s", kind, overall, " ".join(f"{f}={v:.4f}" for f, v in per_fold.items()))
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    log.info(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "seconds": round(elapsed, 1), "peak_gpu_gb": round(peak, 2)}))


if __name__ == "__main__":
    main()
