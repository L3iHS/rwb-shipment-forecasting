"""Forward-only feasible half-hour bounds; local diagnostic reference."""
import numpy as np


def halfhour_bounds(hour_target, rounding_tolerance=0.5):
    """Return bounds for x[t] given only y[:t+1], y[t] ~= x[t-1]+x[t].

    This does not use future zero anchors or backward smoothing. Each latent
    half-hour volume is constrained to be nonnegative. Rounding tolerance is
    an observation error bound, not a fitted smoothing hyperparameter.
    """
    y = np.asarray(hour_target, dtype=np.float64)
    eps = float(rounding_tolerance)
    if (y.ndim != 2 or 0 in y.shape or not np.isfinite(y).all() or (y < 0).any()
            or not np.isfinite(eps) or eps < 0):
        raise ValueError('Expected a nonempty nonnegative finite time-by-route panel and tolerance')
    lower, upper = np.empty_like(y), np.empty_like(y)
    lower[0], upper[0] = 0., y[0] + eps
    for t in range(1, len(y)):
        lower[t] = np.maximum(0., y[t] - eps - upper[t-1])
        upper[t] = y[t] + eps - lower[t-1]
        if (upper[t] < lower[t] - 1e-8).any():
            raise ValueError(f'No feasible nonnegative rolling-sum sequence at source index {t}')
        # Only remove sub-epsilon floating-point roundoff, not substantive conflicts.
        upper[t] = np.maximum(upper[t], lower[t])
    return lower, upper
