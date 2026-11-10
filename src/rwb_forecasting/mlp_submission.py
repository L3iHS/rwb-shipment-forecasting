"""Join certified CSVs by test IDs and compose fixed V3 totals with MLP shares."""
import numpy as np
import pandas as pd

from .data import HORIZON, STEP
from .mass_preserving import mass_preserving_blend
from .submission import make_submission


CANDIDATE = "v3_totals_mlp_shares50"


def csv_forecast(data, test, forecast):
    """Convert id,y_pred to route-major forecasts through the exact test grid."""
    required = ["id", "route_id", "timestamp"]
    if (not set(required).issubset(test) or test[required].isna().any().any()
            or not test.id.is_unique or list(forecast) != ["id", "y_pred"]
            or forecast.isna().any().any() or not forecast.id.is_unique
            or set(forecast.id) != set(test.id)):
        raise ValueError("Unique matching forecast IDs and complete test keys are required")
    times = pd.date_range(data.timestamps[-1] + STEP, periods=HORIZON, freq=STEP)
    expected = pd.MultiIndex.from_product([data.routes, times], names=["route_id", "timestamp"])
    keyed = test[required].copy(); keyed["timestamp"] = pd.to_datetime(keyed.timestamp)
    actual = pd.MultiIndex.from_frame(keyed[["route_id", "timestamp"]])
    if len(actual) != len(expected) or not actual.is_unique or not actual.difference(expected).empty:
        raise ValueError("Test must be the next eight points for every known route")
    ids = keyed.set_index(["route_id", "timestamp"]).reindex(expected).id
    values = forecast.set_index("id").y_pred.reindex(ids).to_numpy(dtype=np.float64).reshape(len(data.routes), HORIZON)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Finite nonnegative CSV forecasts are required")
    return values


def compose_v3_mlp(mae25, v3, raw_mlp):
    """Recreate the original final MLP first, then blend its shares into V3."""
    final_mlp, inner = mass_preserving_blend(mae25, raw_mlp, weight=.5)
    result, outer = mass_preserving_blend(v3, final_mlp, weight=.5)
    np.testing.assert_array_equal(result[:, 0].view(np.uint64), np.asarray(v3, dtype=np.float64)[:, 0].view(np.uint64))
    np.testing.assert_allclose(result.sum(0), np.asarray(v3).sum(0), atol=1e-6, rtol=5e-14)
    return result, dict(original_final_mlp=inner, v3_totals_mlp_shares=outer)


def make_mlp_share_submission(data, test, mae25_csv, v3_csv, *, raw_predictor):
    """Predict only the MLP; the two certified expensive forecasts are inputs."""
    mae25 = csv_forecast(data, test, mae25_csv)
    v3 = csv_forecast(data, test, v3_csv)
    raw, audit = raw_predictor(data)
    result, blend = compose_v3_mlp(mae25, v3, raw)
    audit = {**audit, "model": CANDIDATE, "composition": blend, "platform_submission": False}
    # Reuse the project's route/timestamp→ID validation and output mapping.
    output, audit = make_submission(data, test, predictor=lambda history: (result, audit))
    return output, audit
