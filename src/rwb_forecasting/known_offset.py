"""Refine the already observed half of V3's first forecast hour."""
import numpy as np

from .halfhour import halfhour_bounds

CANDIDATE = "phase8_h1_full_causal_known_offset"


def replace_h1_offset(raw_history, incumbent):
    """Use only hourly observations up to the origin. Arrays have route columns/rows respectively."""
    y = np.asarray(raw_history, dtype=np.float64)
    base = np.asarray(incumbent, dtype=np.float64)
    if y.ndim != 2 or not y.size or base.shape != (y.shape[1], 8):
        raise ValueError("Expected time-by-route observations and route-by-eight forecasts")
    if not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Incumbent must contain finite nonnegative forecasts")
    # Production V3 uses the unclipped history for this feature. It starts the
    # first 16 rows at y/2 and falls back to y/2 on a route with conflicting bounds.
    old = y[-1] / 2
    if len(y) > 16:
        low = np.zeros(y.shape[1])
        high = y[-1].copy()
        alternating = np.zeros(y.shape[1])
        for lag in range(16):
            alternating += ((-1) ** lag) * y[-1 - lag]
            if lag % 2:
                low = np.maximum(low, alternating)
            else:
                high = np.minimum(high, alternating)
        old = np.where(low <= high, (low + high) / 2, old)
    lower, upper = halfhour_bounds(y, rounding_tolerance=0.5)
    full = (lower[-1] + upper[-1]) / 2
    corrected = base.copy()
    corrected[:, 0] = np.maximum(base[:, 0] - old + full, 0)
    return corrected
