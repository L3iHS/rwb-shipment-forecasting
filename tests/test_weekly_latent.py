"""Generated regressions for the selected weekly latent training/decoding contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import numpy as np
import pandas as pd

from rwb_forecasting.data import ForecastData
from rwb_forecasting.weekly_latent import (
    CANDIDATE,
    fit_weekly,
    full_context,
    hourly_from_native,
    redistribute_known_h1,
    validate_week,
    load_checkpoint,
    file_sha256,
)


def history():
    times = pd.date_range("2025-01-01", "2025-02-01 10:30", freq="30min")
    t = np.arange(len(times))[:, None]
    r = np.arange(2)[None, :]
    half = 3000 + 300 * np.sin(t / 9 + r) + 500 * r
    raw = half + np.vstack([half[:1], half[:-1]])
    status = np.stack([half * (1 + j / 10) + 50 * np.cos(t / 5 + j) for j in range(6)], axis=2)
    return ForecastData(times, np.arange(2), status, raw)


class WeeklyTests(unittest.TestCase):
    def test_future_half_labels_source_order_scales_and_cutoff(self):
        h = history()
        context = full_context(h)
        captured = {}

        def dataset(X, **kwargs):
            captured.update(
                X=np.array(X),
                **{k: np.array(v) if k in ("label", "weight") else v for k, v in kwargs.items()},
            )
            return captured

        def trainer(params, data, *, num_boost_round):
            self.assertEqual(num_boost_round, 600)
            self.assertEqual(params["objective"], "tweedie")
            self.assertEqual(params["tweedie_variance_power"], 1.3)
            return SimpleNamespace(
                current_iteration=lambda: 600, feature_name=lambda: captured["feature_name"]
            )

        with TemporaryDirectory() as temp:
            _, audit = fit_weekly(h, Path(temp), dataset_factory=dataset, trainer=trainer)
        self.assertLessEqual(pd.Timestamp(audit["latest_label_timestamp"]), h.timestamps[-1])
        self.assertEqual(audit["latest_label_timestamp"], "2025-01-31 17:00:00")
        sources = context.training_sources()
        expected = []
        weights = []
        matrices = []
        for source in sources:
            features, scale = context.features(int(source))
            expected.extend((context.half_midpoint[source + 1 : source + 9] / scale).reshape(-1))
            weights.extend(np.tile(scale, 8))
            matrices.append(features.drop(columns="route_id").to_numpy())
        np.testing.assert_array_equal(captured["label"], expected)
        np.testing.assert_array_equal(captured["weight"], weights)
        np.testing.assert_array_equal(captured["X"], np.vstack(matrices))
        self.assertEqual(captured["X"].dtype, np.float64)
        self.assertEqual(len(captured["feature_name"]), 123)
        self.assertEqual(captured["categorical_feature"], [])

    def test_full_prefix_features_ignore_later_values(self):
        h = history()
        source = 1420
        original = full_context(h)
        raw = h.target.copy()
        statuses = h.statuses.copy()
        known = original.half_midpoint[source]
        raw[source + 1 :] = 200000
        raw[source + 1] = known + 100000
        statuses[source + 1 :] += 1000000
        changed = full_context(ForecastData(h.timestamps, h.routes, statuses, raw))
        expected, scale = original.features(source)
        actual, scale2 = changed.features(source)
        np.testing.assert_array_equal(expected.to_numpy(), actual.to_numpy())
        np.testing.assert_array_equal(scale, scale2)

    def test_disjoint_half_decoding_is_horizon_major_and_keeps_known_h1(self):
        vector = np.arange(16, dtype=float) - 1
        scale = np.array([2.0, 3.0])
        known = np.array([10.0, 20.0])
        future = np.maximum(vector.reshape(8, 2).T, 0) * scale[:, None]
        result = hourly_from_native(vector, scale, known)
        np.testing.assert_array_equal(result[:, 0], known + future[:, 0])
        np.testing.assert_array_equal(result[:, 1:], future[:, 1:] + future[:, :-1])

    def test_redistribution_reserves_known_h1_and_every_horizon_total(self):
        known = np.array([100.0, 200.0])
        anchor = np.tile([10.0, 30.0], (8, 1)).T
        anchor[:, 0] += known
        alternative = np.tile([30.0, 10.0], (8, 1)).T
        alternative[:, 0] += known
        result = redistribute_known_h1(anchor, alternative, known)
        np.testing.assert_allclose(result.sum(0), anchor.sum(0), rtol=0, atol=1e-10)
        self.assertTrue((result[:, 0] >= known).all())
        self.assertFalse(np.array_equal(result[:, 0], anchor[:, 0]))
        residual = result[:, 0] - known
        np.testing.assert_allclose(
            residual, 0.5 * np.array([10.0, 30.0]) + 0.5 * np.array([30.0, 10.0])
        )
        zero = np.zeros_like(anchor)
        zero[:, 0] = known
        np.testing.assert_array_equal(redistribute_known_h1(anchor, zero, known), anchor)
        with self.assertRaises(ValueError):
            redistribute_known_h1(anchor, alternative, known + 1000)

    def test_wrong_week_and_future_labels_rejected(self):
        audit = dict(
            fit_origin="2025-02-01 10:30:00",
            last_source="2025-01-31 13:00:00",
            latest_label_timestamp="2025-01-31 17:00:00",
        )
        validate_week(audit, "2025-02-07 10:30")
        with self.assertRaises(ValueError):
            validate_week(audit, "2025-02-08 10:30")
        with self.assertRaises(ValueError):
            validate_week(
                {**audit, "latest_label_timestamp": "2025-02-01 11:00:00"}, "2025-02-01 10:30"
            )
        with self.assertRaises(ValueError):
            validate_week({**audit, "last_source": "2025-01-31 12:30:00"}, "2025-02-01 10:30")

    def test_native_checkpoint_hash_checked_before_model_construction(self):
        with TemporaryDirectory() as temp:
            model = Path(temp) / "native.txt"
            model.write_text("generated native fixture")
            audit = Path(temp) / "metadata.json"
            fit = dict(
                fit_origin="2025-02-01 10:30:00",
                last_source="2025-01-31 13:00:00",
                latest_label_timestamp="2025-01-31 17:00:00",
            )
            audit.write_text(json.dumps(dict(audit=fit, boosters={CANDIDATE: file_sha256(model)})))
            factory = lambda **kw: SimpleNamespace(current_iteration=lambda: 600)
            _, restored = load_checkpoint(model, audit, booster_factory=factory)
            self.assertEqual(restored, fit)
            model.write_text("changed")
            with self.assertRaises(ValueError):
                load_checkpoint(
                    model, audit, booster_factory=lambda **kw: self.fail("must not construct")
                )


if __name__ == "__main__":
    unittest.main()
