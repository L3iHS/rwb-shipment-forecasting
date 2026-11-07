"""Regression for loading native LightGBM before training the MLP on macOS."""
from importlib.util import find_spec
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(find_spec("torch") is None, reason="Optional PyTorch backend is required")
def test_tree_and_neural_training_use_distinct_processes(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               VECLIB_MAXIMUM_THREADS="2")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                   env=env, check=True, timeout=90)


if __name__ == "__main__":
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from rwb_forecasting.data import ForecastData
    from rwb_forecasting.mlp_isolation import predict_latent_mlp_isolated

    rng = np.random.default_rng(21)
    features = rng.normal(size=(128, 3))
    lgb.train({"objective": "regression", "num_threads": 2, "verbosity": -1},
              lgb.Dataset(features, label=features[:, 0]), num_boost_round=2)
    times = pd.date_range("2000-01-01", periods=352, freq="30min")
    half = rng.poisson(8, (len(times) + 1, 4))
    panel = ForecastData(times, np.arange(4), rng.poisson(4, (len(times), 4, 6)),
                         half[:-1] + half[1:])
    result, audit = predict_latent_mlp_isolated(panel, Path(sys.argv[1]) / "model.pt")
    receipt = audit["tensor_training"]["process_isolation"]
    assert result.shape == (4, 8) and np.isfinite(result).all()
    assert receipt["pid"] != os.getpid() and receipt["checkpoint_reload_bitwise"]
    assert not receipt["lightgbm_imported"] and "torch" not in sys.modules
