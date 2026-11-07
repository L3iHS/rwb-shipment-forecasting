import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.data import HORIZON, Panel, wape_rbias
from rwb.features import MAX_LOOKBACK, build, cumsums


def _panel(seed: int = 0, R: int = 5, T: int = MAX_LOOKBACK + 200) -> Panel:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-07-28", periods=T, freq="30min")
    y = rng.gamma(2.0, 100.0, size=(R, T)).astype(np.float32)
    s = rng.gamma(2.0, 50.0, size=(6, R, T)).astype(np.float32)
    return Panel(times, np.arange(R), y, s, pd.DataFrame())


@pytest.mark.parametrize("glob", [False, True])
@pytest.mark.parametrize("h", range(1, HORIZON + 1))
def test_features_ignore_future(h: int, glob: bool) -> None:
    panel = _panel()
    origins = np.array([MAX_LOOKBACK + 50, MAX_LOOKBACK + 120])
    X, names, base = build(panel, cumsums(panel), origins, h, glob, glob)

    # Corrupt everything strictly after the earliest origin; its rows must not change.
    o = origins[0]
    corrupted = Panel(panel.times, panel.routes, panel.y.copy(), panel.s.copy(), panel.test)
    corrupted.y[:, o + 1 :] = 1e6
    corrupted.s[:, :, o + 1 :] = 1e6
    X2, _, base2 = build(corrupted, cumsums(corrupted), origins, h, glob, glob)
    R = panel.n_routes
    np.testing.assert_allclose(X[:R], X2[:R], rtol=1e-5, err_msg=str(names))
    np.testing.assert_allclose(base[:R], base2[:R], rtol=1e-6)


def test_metric_matches_definition() -> None:
    y = np.array([10.0, 0.0, 5.0])
    p = np.array([8.0, 1.0, 5.0])
    assert wape_rbias(y, p) == pytest.approx(3 / 15 + abs(14 / 15 - 1))


def test_build_at_last_origin() -> None:
    """The test forecast starts at the final column; targets lie beyond the panel."""
    panel = _panel()
    last = panel.n_times - 1
    for h in range(1, HORIZON + 1):
        X, names, _ = build(panel, cumsums(panel), np.array([last]), h)
        slot_t = X[:, names.index("slot_t")]
        expected = (panel.times[-1] + h * pd.Timedelta(minutes=30))
        assert np.all(slot_t == expected.hour * 2 + expected.minute // 30)
        assert np.isfinite(X).all()
