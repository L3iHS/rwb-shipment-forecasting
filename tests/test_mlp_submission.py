"""Generated ID mapping and fixed nested-share composition checks."""
import unittest

import numpy as np
import pandas as pd

from rwb_forecasting.data import ForecastData
from rwb_forecasting.mass_preserving import mass_preserving_blend
from rwb_forecasting.mlp_submission import csv_forecast, compose_v3_mlp, make_mlp_share_submission


class MLPSubmissionTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2000-01-01 10:00", periods=2, freq="30min")
        self.data = ForecastData(times, np.arange(3), np.ones((2, 3, 6)), np.ones((2, 3)))
        future = pd.date_range(times[-1] + pd.Timedelta(minutes=30), periods=8, freq="30min")
        self.test = pd.MultiIndex.from_product([self.data.routes, future], names=["route_id", "timestamp"]).to_frame(index=False)
        self.test["id"] = np.arange(24)[::-1] * 5 + 17
        self.mae = np.arange(24, dtype=float).reshape(3, 8) + 10
        self.v3 = self.mae[::-1] * 2
        self.raw = self.mae ** 2 / 10
        self.mae_csv = pd.DataFrame(dict(id=self.test.id, y_pred=self.mae.reshape(-1))).iloc[::-1]
        self.v3_csv = pd.DataFrame(dict(id=self.test.id, y_pred=self.v3.reshape(-1))).sample(frac=1, random_state=3)

    def test_shuffled_inputs_map_by_ids_and_expected_future_keys(self):
        np.testing.assert_array_equal(csv_forecast(self.data, self.test.sample(frac=1, random_state=9), self.v3_csv), self.v3)
        bad = self.test.copy(); bad.loc[0, "timestamp"] += pd.Timedelta(minutes=30)
        with self.assertRaises(ValueError):
            csv_forecast(self.data, bad, self.v3_csv)
        missing = self.v3_csv.iloc[:-1]
        with self.assertRaises(ValueError):
            csv_forecast(self.data, self.test, missing)

    def test_exact_two_layers_h1_totals_and_common_output_mapping(self):
        inner, _ = mass_preserving_blend(self.mae, self.raw, .5)
        expected, _ = mass_preserving_blend(self.v3, inner, .5)
        actual, _ = compose_v3_mlp(self.mae, self.v3, self.raw)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, 0].view(np.uint64), self.v3[:, 0].view(np.uint64))
        np.testing.assert_allclose(actual.sum(0), self.v3.sum(0), atol=1e-6, rtol=5e-14)
        calls = []
        def raw_predictor(history):
            calls.append(history)
            return self.raw, dict(origin=str(history.timestamps[-1]),
                latest_label_timestamp=str(history.timestamps[-1]), latest_feature_timestamp=str(history.timestamps[-1]))
        output, audit = make_mlp_share_submission(self.data, self.test.sample(frac=1, random_state=5),
            self.mae_csv, self.v3_csv, raw_predictor=raw_predictor)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(output), 24)
        mapped = output.set_index("id").reindex(self.test.id).y_pred.to_numpy().reshape(3, 8)
        np.testing.assert_array_equal(mapped, expected)
        self.assertFalse(audit["platform_submission"])


if __name__ == "__main__":
    unittest.main()
