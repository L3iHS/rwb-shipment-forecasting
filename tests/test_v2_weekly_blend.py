from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = spec_from_file_location(
    "v2_weekly_cli", Path(__file__).resolve().parents[1] / "scripts/make_v2_weekly_blend.py"
)
cli = module_from_spec(spec)
spec.loader.exec_module(cli)


def test_forecasts_are_joined_by_id_when_orders_differ():
    test = pd.DataFrame(
        {
            "id": [8, 2, 5],
            "route_id": [0, 0, 1],
            "timestamp": pd.to_datetime(
                ["2025-01-01 11:00", "2025-01-01 11:30", "2025-01-01 11:00"]
            ),
        }
    )
    peer = pd.DataFrame({"id": [5, 8, 2], "y_pred": [20.0, 40.0, 60.0]})
    weekly = pd.DataFrame({"id": [2, 5, 8], "y_pred": [20.0, 40.0, 60.0]})
    result = cli.blend(test, peer, weekly)
    np.testing.assert_array_equal(result.id, [2, 5, 8])
    np.testing.assert_array_equal(result.y_pred, [40.0, 30.0, 50.0])


@pytest.mark.parametrize(
    "ids,values",
    [([1, 1], [2.0, 3.0]), ([1, 3], [2.0, 3.0]), ([1, 2], [2.0, np.nan]), ([1, 2], [2.0, -1.0])],
)
def test_incomplete_or_invalid_forecasts_are_rejected(ids, values):
    test = pd.DataFrame(
        {
            "id": [1, 2],
            "route_id": [0, 0],
            "timestamp": pd.to_datetime(["2025-01-01 11:00", "2025-01-01 11:30"]),
        }
    )
    good = pd.DataFrame({"id": [1, 2], "y_pred": [3.0, 5.0]})
    with pytest.raises(ValueError):
        cli.blend(test, good, pd.DataFrame({"id": ids, "y_pred": values}))
