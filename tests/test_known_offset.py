import numpy as np
import pandas as pd

from rwb_forecasting.data import ForecastData
from rwb_forecasting.halfhour import halfhour_bounds
from rwb_forecasting.known_offset import replace_h1_offset
from rwb_forecasting.v3_features import V3Context


def test_replaces_only_observed_offset_and_uses_no_later_rows():
    rng = np.random.default_rng(22)
    latent = rng.gamma(2, 4, size=(1471, 4))
    latent[::23, 0] = 0
    observed = np.rint(latent[:-1] + latent[1:])
    times = pd.date_range("2000-01-01", periods=len(observed), freq="30min")
    panel = ForecastData(times, np.arange(4), np.ones((len(times), 4, 6)), observed)
    base = np.arange(32, dtype=float).reshape(4, 8) + 1 / 3
    for prefix in (panel.history(times[-4]), panel.history(times[-1])):
        old = V3Context.from_history(prefix).half_midpoint[-1]
        lo, hi = halfhour_bounds(prefix.target, rounding_tolerance=0.5)
        new = (lo[-1] + hi[-1]) / 2
        assert np.any(old != new)
        result = replace_h1_offset(prefix.target, base)
        np.testing.assert_array_equal(result[:, 0], np.maximum(base[:, 0] - old + new, 0))
        np.testing.assert_array_equal(result[:, 1:].view("u8"), base[:, 1:].view("u8"))
