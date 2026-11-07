import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import HORIZON, STEP
from v3_test_data import constant_panel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rwb_v3_cli_test", ROOT / "scripts/run_v3.py")
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


def test_cli_requires_explicit_origins_and_fixed_config():
    with pytest.raises(SystemExit):
        cli.argument_parser().parse_args(["backtest", "--config", "config.json", "--train", "train.parquet", "--output-dir", "run"])
    config, runtime = cli.load_config(ROOT / "configs/v3_tweedie_h1_600.json")
    assert runtime.native_threads == 2 and config["native_parameters"]["lambda_l2"] == 1.


def test_config_does_not_accept_extra_statistical_knobs(tmp_path):
    config = json.loads((ROOT / "configs/v3_tweedie_h1_600.json").read_text())
    config["multiplier"] = 1.02
    path = tmp_path / "changed.json"; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fields"):
        cli.load_config(path)


def test_backtest_passes_only_origin_prefix_and_scores_per_origin():
    data = constant_panel(1700); origins = [data.timestamps[1509], data.timestamps[1557]]
    observed = []
    def fake(history):
        observed.append(history.timestamps[-1])
        assert len(history.timestamps) == data.timestamps.get_loc(history.timestamps[-1]) + 1
        return np.full((3, 8), 100.), {"latest_label_timestamp": str(history.timestamps[-1])}
    result = cli.backtest_explicit(data, origins, fake)
    assert observed == origins and len(result["predictions"]) == 48
    assert result["summary"].iloc[0].mean_score == 0.
    assert set(result["folds"].origin) == set(origins)


def test_submission_maps_shuffled_IDs_using_existing_project_contract():
    data = constant_panel(); future = pd.date_range(data.timestamps[-1] + STEP, periods=HORIZON, freq=STEP)
    test = pd.DataFrame({"id": np.arange(24), "route_id": np.repeat(data.routes, 8), "timestamp": np.tile(future, 3)})
    values = np.arange(24, dtype=float).reshape(3, 8)
    result, _ = cli.local_submission(data, test.sample(frac=1, random_state=8),
        lambda history: (values, {"latest_label_timestamp": str(history.timestamps[-1])}))
    np.testing.assert_array_equal(result.id, np.arange(24))
    np.testing.assert_array_equal(result.y_pred, values.reshape(-1))


def test_existing_CSV_and_backtest_directory_are_never_overwritten(tmp_path):
    path = tmp_path / "existing.csv"; path.write_text("keep\n")
    with pytest.raises(FileExistsError):
        cli.write_submission(pd.DataFrame({"id": [1], "y_pred": [1.]}), path, {})
    assert path.read_text() == "keep\n"
    directory = tmp_path / "existing"; directory.mkdir()
    with pytest.raises(FileExistsError):
        cli.ensure_new_output(directory, directory=True)


@pytest.mark.parametrize("kind", ["backtest", "csv", "metadata"])
def test_work_directory_cannot_create_output_before_training(tmp_path, kind):
    destination = tmp_path / ("run" if kind == "backtest" else "forecast.csv")
    work_parent = destination.with_suffix(".metadata.json") if kind == "metadata" else destination
    args = ["backtest" if kind == "backtest" else "submit",
            "--config", str(tmp_path / "missing-config.json"),
            "--train", str(tmp_path / "missing-train.parquet"),
            "--work-dir", str(work_parent / "scratch")]
    if kind == "backtest":
        args += ["--origins", "2025-10-18T10:30:00", "--output-dir", str(destination)]
    else:
        args += ["--test", str(tmp_path / "missing-test.parquet"), "--output", str(destination)]
    with pytest.raises(ValueError, match="inside an output destination"):
        cli.main(args)
    assert not destination.exists()
    assert not work_parent.exists()
