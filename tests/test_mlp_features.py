"""Generated causal feature/scaler checks; no Torch dependency."""
import unittest

import numpy as np
import pandas as pd

from rwb_forecasting.data import ForecastData
from rwb_forecasting.latent_volume import infer_half_hour
from rwb_forecasting.linear_candidates import source_indices
from rwb_forecasting.mlp_features import prepare_training, raw_numeric, source_scales
from rwb_forecasting.mlp_model import predict_latent_mlp


def generated_history():
    times = pd.date_range("2000-01-01", periods=1500, freq="30min")
    step = np.arange(len(times))[:, None]
    half = 1400 + 70 * np.arange(3) + 200 * np.sin(step / 31) + step / 4
    hourly = half + np.vstack([half[:1], half[:-1]])
    statuses = np.stack([half * (k + 1) / 6 for k in range(6)], axis=2)
    return ForecastData(times, np.array([7, 11, 42]), statuses, hourly)


class MLPFeatureTests(unittest.TestCase):
    def test_training_shapes_cutoff_scaler_and_route_encoder(self):
        history = generated_history()
        payload, state, audit = prepare_training(history)
        self.assertEqual(payload["x_train"].shape, (673 * 3, 78))
        self.assertEqual(payload["y_train"].shape, (673 * 3, 8))
        self.assertEqual(audit["latest_label_timestamp"], str(history.timestamps[-1]))
        self.assertEqual(audit["last_source_timestamp"], str(history.timestamps[-9]))
        self.assertEqual(audit["scaler_fit_rows"], 673 * 3)
        np.testing.assert_array_equal(payload["route_infer"], np.arange(3))
        np.testing.assert_array_equal(state["route_categories"], history.routes)
        np.testing.assert_allclose(payload["weight_train"].mean(dtype=np.float64), 1, atol=1e-7)
        latent, _ = infer_half_hour(history.target)
        latent_history = ForecastData(history.timestamps, history.routes, history.statuses, latent)
        sources = source_indices(latent_history, 14, None)
        numeric, _, _ = raw_numeric(history, latent_history, sources)
        np.testing.assert_allclose(state["scaler_mean"], numeric.mean(0), atol=1e-12)
        np.testing.assert_allclose(state["scaler_var"], numeric.var(0), atol=1e-12)

    def test_raw_hourly_scaling_and_feature_source_prefix_invariance(self):
        history = generated_history()
        source = 1000
        expected = np.maximum(history.target[source - 47:source + 1].mean(0), 1000.)
        np.testing.assert_allclose(source_scales(history, np.array([source]))[0], expected, atol=1e-10)
        latent, _ = infer_half_hour(history.target)
        point, level, _ = raw_numeric(history,
            ForecastData(history.timestamps, history.routes, history.statuses, latent), np.array([source]))
        poisoned_target, poisoned_statuses = history.target.copy(), history.statuses.copy()
        # Change only future half flows and rebuild valid overlapping hourly
        # sums; arbitrary hourly poisoning could violate the input domain.
        poisoned_target[source + 1:] = 200000
        poisoned_target[source + 1] = latent[source] + 100000
        poisoned_statuses[source + 1:] += 100000
        poisoned = ForecastData(history.timestamps, history.routes, poisoned_statuses, poisoned_target)
        changed_latent, _ = infer_half_hour(poisoned_target)
        changed, changed_level, _ = raw_numeric(poisoned,
            ForecastData(history.timestamps, history.routes, poisoned_statuses, changed_latent), np.array([source]))
        np.testing.assert_array_equal(point.view(np.uint64), changed.view(np.uint64))
        np.testing.assert_array_equal(level.view(np.uint64), changed_level.view(np.uint64))

    def test_fresh_history_predictor_calls_one_generated_trainer(self):
        history = generated_history()
        calls = []
        def trainer(payload, audit):
            calls.append(payload["x_train"].shape)
            return np.ones((3, 8), dtype=np.float32) / 2, {
                "origin":audit["origin"], "latest_label_timestamp":audit["latest_label_timestamp"],
                "latest_feature_timestamp":audit["last_source_timestamp"], "epochs_completed":20}
        raw, audit = predict_latent_mlp(history, tensor_trainer=trainer)
        self.assertEqual(calls, [(673 * 3, 78)])
        self.assertEqual(raw.shape, (3, 8))
        self.assertTrue(np.isfinite(raw).all() and (raw >= 0).all())
        self.assertEqual(audit["latest_label_timestamp"], str(history.timestamps[-1]))


if __name__ == "__main__":
    unittest.main()
