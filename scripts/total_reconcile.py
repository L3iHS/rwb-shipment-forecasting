"""Temporal hierarchy over the 4-hour window: a direct model of each route's 8-step total,
then the per-step blend is pulled toward it.

The total averages out half of the half-hour noise, so its level estimate can be steadier than
the sum of eight separately fitted horizons. Reconciliation is multiplicative per route:
    p'_h = p_h * (S_hat / sum_h p_h) ** alpha
with alpha and the global scale chosen cross-fitted between folds.

    python scripts/total_reconcile.py --blend ensemble_v2            # backtest
    python scripts/total_reconcile.py --blend ensemble_v2 --final    # + reconciled submission
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.backtest import default_folds, score_origins
from rwb.data import HORIZON, ROOT, Panel, load_panel, make_submission
from rwb.features import (
    DAY,
    MAX_LOOKBACK,
    Cumsums,
    base_forecast,
    build,
    cumsums,
    flat_scale,
)
from rwb.models import GBMConfig, _fit_one, _predict_one

log = logging.getLogger("total_reconcile")
OOF = ROOT / "models/oof"
MID = HORIZON // 2  # target-slot features are taken at the middle of the window


def total_features(panel: Panel, cums: Cumsums, origins: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Features at the window middle plus the summed seasonal base; returns X, names, base total (flat)."""
    X, names, _ = build(panel, cums, origins, MID)
    base_tot = sum(base_forecast(panel, cums, origins, h) for h in range(1, HORIZON + 1))
    flat_base = np.ascontiguousarray(base_tot.T).ravel()
    X = np.column_stack([X, (flat_base / (HORIZON * flat_scale(cums, origins))).astype(np.float32)])
    return X, [*names, "base_total"], flat_base


def fit_predict_total(panel: Panel, cums: Cumsums, label_cutoff: int, pred_origins: np.ndarray,
                      cfg: GBMConfig) -> np.ndarray:
    """(n_o, R) forecasts of sum_{h=1..8} y[o+h], trained only on labels <= label_cutoff."""
    hi = label_cutoff - HORIZON
    hi -= (hi - int(pred_origins[0])) % cfg.stride
    lo = max(MAX_LOOKBACK, hi - cfg.days * DAY)
    tr = np.arange(hi, lo - 1, -cfg.stride)[::-1]
    X, names, base = total_features(panel, cums, tr)
    y = np.concatenate([panel.y[:, o + 1 : o + 1 + HORIZON].sum(axis=1) for o in tr])
    scale = HORIZON * flat_scale(cums, tr)
    model = _fit_one(X, (y - base) / scale, scale, names, cfg)
    Xp, _, base_p = total_features(panel, cums, pred_origins)
    pred = base_p + HORIZON * flat_scale(cums, pred_origins) * _predict_one(model, Xp, names, cfg)
    return np.clip(pred, 0, None).reshape(len(pred_origins), panel.n_routes)


def reconcile(p: np.ndarray, s_hat: np.ndarray, alpha: float) -> np.ndarray:
    tot = p.sum(axis=2) + 1.0
    return p * ((s_hat + 1.0) / tot)[:, :, None] ** alpha


ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)
SCALES = np.arange(0.97, 1.05, 0.0025)


def fit_alpha_scale(p: np.ndarray, s_hat: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    best = (np.inf, 0.0, 1.0)
    for a in ALPHAS:
        r = reconcile(p, s_hat, a)
        for k in SCALES:
            s = float(np.mean(score_origins(r * k, y)))
            if s < best[0]:
                best = (s, float(a), float(k))
    return best[1], best[2], best[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="ensemble_v2", help="submissions/<blend>.json with members and weights")
    p.add_argument("--name", default="total_xgb")
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    panel = load_panel()
    cums = cumsums(panel)
    folds = default_folds(panel)
    cfg = GBMConfig(lib="xgb", days=28, stride=2, rounds=600, lr=0.03, leaves=31, min_child=500)
    spec = json.loads((ROOT / f"submissions/{a.blend}.json").read_text(encoding="utf-8"))
    w = np.array(spec["weights"]) / np.sum(spec["weights"])  # shares only; the scale is refitted below

    path = OOF / f"{a.name}.npz"
    if path.exists():
        z = np.load(path)
        s_hat = {f.name: z[f"pred_{f.name}"] for f in folds}
    else:
        s_hat = {f.name: fit_predict_total(panel, cums, f.label_cutoff, f.val_origins, cfg) for f in folds}
        np.savez_compressed(path, **{f"pred_{k}": v for k, v in s_hat.items()})

    P, Y = {}, {}
    for f in folds:
        P[f.name] = np.stack([np.load(OOF / f"{m}.npz")[f"pred_{f.name}"] for m in spec["models"]], -1) @ w
        Y[f.name] = np.load(OOF / f"{spec['models'][0]}.npz")[f"truth_{f.name}"]
        tot_true = Y[f.name].sum(axis=2)
        log.info("%s: route-total WAPE  blend-sum %.4f | total model %.4f", f.name,
                 np.abs(P[f.name].sum(axis=2) * 1.02 - tot_true).sum() / tot_true.sum(),
                 np.abs(s_hat[f.name] - tot_true).sum() / tot_true.sum())

    cross = []
    for fit_f, eval_f in itertools.permutations(sorted(P), 2):
        al, k, _ = fit_alpha_scale(P[fit_f], s_hat[fit_f], Y[fit_f])
        base_k = min(SCALES, key=lambda kk: np.mean(score_origins(P[fit_f] * kk, Y[fit_f])))
        s_rec = np.mean(score_origins(reconcile(P[eval_f], s_hat[eval_f], al) * k, Y[eval_f]))
        s_base = np.mean(score_origins(P[eval_f] * base_k, Y[eval_f]))
        cross.append((s_base, s_rec))
        log.info("fit %s -> eval %s: blend %.4f | reconciled %.4f (alpha %.1f, scale %.4f)", fit_f, eval_f, s_base, s_rec, al, k)
    c = np.array(cross).mean(axis=0)
    Pa = np.concatenate([P[f] for f in sorted(P)])
    Sa = np.concatenate([s_hat[f] for f in sorted(P)])
    Ya = np.concatenate([Y[f] for f in sorted(P)])
    al, k, s_in = fit_alpha_scale(Pa, Sa, Ya)
    log.info("RESULT cross-fitted: blend %.4f -> reconciled %.4f | all folds: alpha %.1f scale %.4f in-sample %.4f",
             c[0], c[1], al, k, s_in)

    if a.final:
        last = panel.n_times - 1
        s_test = fit_predict_total(panel, cums, last, np.array([last]), cfg)
        np.save(OOF / f"{a.name}_test.npy", s_test[0])
        test = np.stack([np.load(OOF / f"{m}_test.npy") for m in spec["models"]], -1) @ w
        pred = reconcile(test[None], s_test, al)[0] * k
        out = ROOT / f"submissions/{a.blend}_rec.csv"
        sub = make_submission(panel, pred)
        sub.to_csv(out, index=False)
        log.info("submission %s sum=%.0f (alpha %.1f, scale %.4f)", out, sub["y_pred"].sum(), al, k)


if __name__ == "__main__":
    main()
