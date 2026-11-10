"""Generated input-mapping and no-overwrite regressions; no learner execution."""

from contextlib import redirect_stderr
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import json
import unittest

import numpy as np
import pandas as pd

from rwb_forecasting.data import ForecastData
from rwb_forecasting.mlp_submission import csv_forecast
from rwb_forecasting.submission import make_submission

spec = spec_from_file_location(
    "weekly_latent_cli", Path(__file__).resolve().parents[1] / "scripts/run_weekly_latent.py"
)
cli = module_from_spec(spec)
spec.loader.exec_module(cli)


class CliTests(unittest.TestCase):
    def test_test_origin_and_id_mapping_are_independent_of_csv_order(self):
        times = pd.date_range("2025-02-01 10:00", periods=2, freq="30min")
        data = ForecastData(times, np.arange(2), np.zeros((2, 2, 6)), np.zeros((2, 2)))
        future = pd.date_range(times[-1] + pd.Timedelta(minutes=30), periods=8, freq="30min")
        test = pd.DataFrame(
            dict(
                id=np.arange(16)[::-1] + 100,
                route_id=np.repeat([0, 1], 8),
                timestamp=np.tile(future, 2),
            )
        )
        target = np.arange(16, dtype=np.float64).reshape(2, 8)
        anchor = pd.DataFrame(dict(id=test.id.to_numpy(), y_pred=target.reshape(-1))).sample(
            frac=1, random_state=4
        )
        test = test.sample(frac=1, random_state=5)
        self.assertEqual(cli.test_origin(test), times[-1])
        aligned = csv_forecast(data, test, anchor)
        np.testing.assert_array_equal(aligned, target)
        result, _ = make_submission(
            data, test, predictor=lambda h: (aligned, dict(latest_feature_timestamp=str(times[-1])))
        )
        expected = anchor.sort_values("id").reset_index(drop=True)
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())

    def test_existing_output_rejected_before_fit_or_input_loading(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "exists.csv"
            output.write_text("existing")
            argv = [
                "--config",
                "unused",
                "--train",
                "unused",
                "--test",
                "unused",
                "--anchor",
                "unused",
                "--output",
                str(output),
            ]
            with patch.object(
                cli, "fit_weekly", side_effect=AssertionError("must not fit")
            ), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.main(argv)

    def test_racing_output_is_preserved_and_partial_links_are_rolled_back(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            staged = {
                name: root / ("staged_" + name) for name in ("model", "fit", "metadata", "csv")
            }
            final = {name: root / name for name in staged}
            for path in staged.values():
                path.write_text("new")
            final["metadata"].write_text("other run")
            with self.assertRaises(FileExistsError):
                cli.publish_outputs(staged, final)
            self.assertEqual(final["metadata"].read_text(), "other run")
            self.assertTrue(all(not final[name].exists() for name in ("model", "fit", "csv")))

    def test_config_refuses_changed_statistical_parameters(self):
        path = Path(__file__).resolve().parents[1] / "configs/weekly_v3_latent_half_shares.json"
        config, _ = cli.load_config(path)
        with TemporaryDirectory() as temp:
            changed = Path(temp) / "changed.json"
            config["native_parameters"]["learning_rate"] = 0.01
            changed.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                cli.load_config(changed)


if __name__ == "__main__":
    unittest.main()
