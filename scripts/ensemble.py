"""Blend saved backtest predictions and choose a calibration scale without peeking.

Weights and scale are fitted on one fold and scored on the other (cross-fitting),
so the reported number estimates how the blend transfers to an unseen day.

    python scripts/ensemble.py xgb_a lgbm_b --final
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.backtest import score_origins
from rwb.data import ROOT, load_panel, make_submission

log = logging.getLogger("ensemble")
OOF = ROOT / "models/oof"


def load(names: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    preds: dict[str, list[np.ndarray]] = {}
    truth: dict[str, np.ndarray] = {}
    for n in names:
        z = np.load(OOF / f"{n}.npz")
        for k in z.files:
            fold = k.split("_", 1)[1]
            if k.startswith("pred_"):
                preds.setdefault(fold, []).append(z[k])
            else:
                truth[fold] = z[k]
    return {f: np.stack(v, axis=-1) for f, v in preds.items()}, truth


SCALES = np.arange(0.95, 1.10, 0.0025)


def best_scale(pred: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    scores = [np.mean(score_origins(pred * k, Y)) for k in SCALES]
    i = int(np.argmin(scores))
    return float(SCALES[i]), float(scores[i])


def fractions(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def fit_weights(P: np.ndarray, Y: np.ndarray, per_h_scale: bool) -> tuple[np.ndarray, np.ndarray]:
    """Blend shares on the simplex (Powell over softmax logits) times one global scale from a grid."""
    m = P.shape[-1]
    if m == 1:
        k, _ = best_scale(P[..., 0], Y)
        w = np.array([k])
    else:
        res = minimize(lambda z: best_scale(P @ fractions(z), Y)[1], np.zeros(m), method="Powell",
                       options={"xtol": 1e-2, "ftol": 1e-6, "maxiter": 3000})
        f = fractions(res.x)
        k, _ = best_scale(P @ f, Y)
        w = f * k
    hs = np.ones(P.shape[2])
    if per_h_scale:
        base = P @ w
        grid = np.arange(0.9, 1.12, 0.005)
        mask = np.arange(P.shape[2])
        # Coordinate descent: each horizon's factor is chosen given the current factors of the others.
        for _ in range(3):
            for h in range(P.shape[2]):
                sc = []
                for g in grid:
                    trial = hs.copy()
                    trial[h] = g
                    sc.append(np.mean(score_origins(base * trial[mask], Y)))
                hs[h] = grid[int(np.argmin(sc))]
    return w, hs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--per_h_scale", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--out", default="ensemble")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    preds, truth = load(a.names)
    folds = sorted(preds)
    for n, j in zip(a.names, range(len(a.names))):
        raw = [np.mean(score_origins(preds[f][..., j], truth[f])) for f in folds]
        log.info("single %-20s %s", n, np.round(raw, 4).tolist())

    cross = []
    for fit_f, eval_f in itertools.permutations(folds, 2):
        w, hs = fit_weights(preds[fit_f], truth[fit_f], a.per_h_scale)
        s = np.mean(score_origins((preds[eval_f] @ w) * hs, truth[eval_f]))
        cross.append(s)
        log.info("fit %s -> eval %s: %.4f  w=%s hs=%s", fit_f, eval_f, s, np.round(w, 3).tolist(), np.round(hs, 3).tolist())
    P = np.concatenate([preds[f] for f in folds])
    Y = np.concatenate([truth[f] for f in folds])
    w, hs = fit_weights(P, Y, a.per_h_scale)
    log.info("RESULT cross-fitted=%.4f | all-folds weights=%s hs=%s in-sample=%.4f", np.mean(cross),
             np.round(w, 3).tolist(), np.round(hs, 3).tolist(), np.mean(score_origins((P @ w) * hs, Y)))

    if a.final:
        panel = load_panel()
        test = np.stack([np.load(OOF / f"{n}_test.npy") for n in a.names], axis=-1)
        pred = (test @ w) * hs
        sub = make_submission(panel, pred)
        out = ROOT / f"submissions/{a.out}.csv"
        sub.to_csv(out, index=False)
        (ROOT / f"submissions/{a.out}.json").write_text(json.dumps(
            {"models": a.names, "weights": w.tolist(), "h_scale": hs.tolist(), "cross_fitted": float(np.mean(cross))},
            indent=2), encoding="utf-8")
        log.info("submission %s sum=%.0f", out, sub["y_pred"].sum())


if __name__ == "__main__":
    main()
