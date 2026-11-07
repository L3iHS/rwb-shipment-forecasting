"""Organizer's Ridge baseline re-run on our backtest folds, for a like-for-like reference.

Mirrors data/reference/baseline.ipynb: features = route one-hot + current statuses,
8 targets, last 14 days of origins, Ridge(alpha=4), clip at 0. Only labels up to the fold
cutoff are used, which the notebook's own split did not guarantee.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.backtest import default_folds, score_origins, truth_for
from rwb.data import HORIZON, load_panel
from rwb.features import DAY

log = logging.getLogger("organizer_baseline")


def design(s: np.ndarray, origins: np.ndarray, R: int, scaler: StandardScaler, fit: bool) -> sparse.csr_matrix:
    num = np.concatenate([s[:, :, o].T for o in origins])  # (n_o*R, 6)
    num = scaler.fit_transform(num) if fit else scaler.transform(num)
    onehot = sparse.csr_matrix((np.ones(len(origins) * R), (np.arange(len(origins) * R), np.tile(np.arange(R), len(origins)))),
                               shape=(len(origins) * R, R))
    return sparse.hstack([sparse.csr_matrix(num), onehot]).tocsr()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    panel = load_panel()
    R = panel.n_routes
    scores = []
    for fold in default_folds(panel):
        hi = fold.label_cutoff - HORIZON
        tr = np.arange(hi - 14 * DAY, hi + 1)
        Y = np.concatenate([panel.y[:, o + 1 : o + 1 + HORIZON] for o in tr])
        sc = StandardScaler()
        model = Ridge(alpha=4.0).fit(design(panel.s, tr, R, sc, True), Y)
        pred = model.predict(design(panel.s, fold.val_origins, R, sc, False)).reshape(len(fold.val_origins), R, HORIZON)
        s = score_origins(np.clip(pred, 0, None), truth_for(panel, fold.val_origins))
        log.info("%s %.4f", fold.name, np.mean(s))
        scores += s
    log.info("RESULT organizer ridge mean=%.4f", np.mean(scores))


if __name__ == "__main__":
    main()
