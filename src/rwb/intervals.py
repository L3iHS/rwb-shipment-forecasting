"""Prediction intervals by split-conformal calibration of relative errors.

A route's error grows with its volume, so residuals are taken relative to the forecast
(e = y / (pred + eps) - 1) and their empirical quantiles are kept per horizon and per route-size
group. Calibrated on recent rolling forecasts, they turn the point forecast into a band such as
"with 90% probability no more than X", which the truck decision needs.
"""

from dataclasses import dataclass, field

import numpy as np

# Groups by route volume: small routes are relatively noisier.
N_GROUPS = 5


def size_groups(level: np.ndarray, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Route group ids (R,) from a per-route volume level; edges are reused at prediction time."""
    if edges is None:
        edges = np.quantile(level, np.linspace(0, 1, N_GROUPS + 1)[1:-1])
    return np.searchsorted(edges, level, side="right"), edges


@dataclass
class Conformal:
    """Quantiles of the relative error per (horizon, size group)."""

    probs: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    eps: float = 1.0
    table: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (H, G, n_probs)
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, pred: np.ndarray, truth: np.ndarray, level: np.ndarray) -> "Conformal":
        """pred, truth: (n_origins, R, H); level: (R,) route volume measured before the calibration data."""
        if pred.ndim != 3 or pred.shape != truth.shape or level.shape != (pred.shape[1],):
            raise ValueError("Invalid calibration shapes")
        if not np.isfinite(pred).all() or not np.isfinite(truth).all() or not np.isfinite(level).all():
            raise ValueError("Calibration requires finite data; missing facts are not zeros")
        if (pred < 0).any() or (truth < 0).any() or (level < 0).any() or not len(pred):
            raise ValueError("Calibration requires nonnegative data and at least one origin")
        groups, self.edges = size_groups(level)
        rel = truth / (pred + self.eps) - 1.0
        H = pred.shape[2]
        self.table = np.empty((H, N_GROUPS, len(self.probs)))
        for g in range(N_GROUPS):
            r = rel[:, groups == g, :]
            if not r.shape[1]:
                # Network calibration (one series) and small panels have empty size groups.
                r = rel
            for h in range(H):
                v = r[:, :, h].ravel()
                self.table[h, g] = np.quantile(v[np.isfinite(v)], self.probs)
        return self

    def quantile(self, pred: np.ndarray, level: np.ndarray, p: float) -> np.ndarray:
        """Forecast quantile p for pred (..., R, H)."""
        if p not in self.probs:
            raise ValueError(f"probability {p} was not calibrated; available: {self.probs}")
        groups, _ = size_groups(level, self.edges)
        k = self.probs.index(p)
        q = self.table[:, groups, k].T  # (R, H)
        return np.clip((pred + self.eps) * (1.0 + q), 0, None)

    def cumulative_quantile(self, pred: np.ndarray, level: np.ndarray, p: float) -> np.ndarray:
        """Monotone rearrangement over horizons; coverage must be evaluated after this step.

        Sorting each horizon sequence preserves quantile order between probabilities.
        These remain marginal horizon bands, not simultaneous path guarantees.
        """
        return np.sort(self.quantile(pred, level, p), axis=-1)


def coverage(lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> float:
    return float(((truth >= lo) & (truth <= hi)).mean())


def pinball(q_pred: np.ndarray, truth: np.ndarray, p: float) -> float:
    """Quantile loss normalised by the total volume, comparable across quantiles and to WAPE."""
    d = truth - q_pred
    return float(np.maximum(p * d, (p - 1) * d).sum() / truth.sum())
