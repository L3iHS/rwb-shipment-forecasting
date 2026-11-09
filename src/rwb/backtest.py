"""Rolling-origin backtest that mirrors the test setup (one origin, 8 steps, all routes)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rwb.data import HORIZON, Panel, wape_rbias
from rwb.features import DAY

log = logging.getLogger(__name__)


@dataclass
class Fold:
    name: str
    val_origins: np.ndarray
    # Last time index whose label may be used for training.
    label_cutoff: int


@dataclass
class Result:
    per_origin: dict[str, list[float]] = field(default_factory=dict)
    preds: dict[str, np.ndarray] = field(default_factory=dict)
    truth: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        out = {f: float(np.mean(v)) for f, v in self.per_origin.items()}
        out["mean"] = float(np.mean([x for v in self.per_origin.values() for x in v]))
        return out


def default_folds(panel: Panel, days_per_fold: int = 7, n_folds: int = 2) -> list[Fold]:
    """Validation origins at the same clock time as the test origin on previous days."""
    last = panel.n_times - 1
    folds = []
    for f in range(n_folds):
        js = np.arange(f * days_per_fold + 1, (f + 1) * days_per_fold + 1)
        val = np.sort(last - js * DAY)
        folds.append(Fold(f"fold{f}", val, int(val.min())))
    return folds


def train_origins(cutoff: int, h: int, days: int, stride: int, min_origin: int) -> np.ndarray:
    """Origins whose label at o+h is not later than ``cutoff``."""
    hi = cutoff - h
    lo = max(min_origin, hi - days * DAY)
    return np.arange(hi, lo - 1, -stride)[::-1]


def score_origins(pred: np.ndarray, truth: np.ndarray) -> list[float]:
    return [wape_rbias(truth[i], pred[i]) for i in range(len(pred))]


def truth_for(panel: Panel, origins: np.ndarray) -> np.ndarray:
    return np.stack([panel.y[:, o + 1 : o + 1 + HORIZON] for o in origins])


def run(panel: Panel, folds: list[Fold], fit_predict: Callable[[Fold], np.ndarray]) -> Result:
    res = Result()
    for fold in folds:
        pred = fit_predict(fold)
        truth = truth_for(panel, fold.val_origins)
        res.preds[fold.name] = pred
        res.truth[fold.name] = truth
        res.per_origin[fold.name] = score_origins(pred, truth)
        log.info("%s: mean=%.4f per-origin=%s", fold.name, np.mean(res.per_origin[fold.name]),
                 np.round(res.per_origin[fold.name], 4).tolist())
    return res
