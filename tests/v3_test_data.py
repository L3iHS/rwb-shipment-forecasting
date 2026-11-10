"""Generated panels only; independent of private/local reference adapters."""
import numpy as np
import pandas as pd
from rwb_forecasting.data import ForecastData


def constant_panel(steps=1600, routes=3):
    return ForecastData(pd.date_range("2000-01-01", periods=steps, freq="30min"),
        np.arange(routes), np.broadcast_to(np.arange(1., 7.) * 10, (steps, routes, 6)).copy(),
        np.full((steps, routes), 100.))


def random_panel(steps=1600, routes=3):
    rng = np.random.default_rng(714)
    target = rng.uniform(50, 300, (steps, routes)); target[::53] *= 50
    statuses = rng.uniform(10, 500, (steps, routes, 6))
    return ForecastData(pd.date_range("2000-01-01", periods=steps, freq="30min"),
                       np.arange(routes), statuses, target)
