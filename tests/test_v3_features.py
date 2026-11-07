import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.v3_features import V3Context
from v3_test_data import constant_panel, random_panel


def test_constant_throughput_features_have_known_values_and_H1_offsets():
    context = V3Context.from_history(constant_panel())
    frame, scales = context.features(1550)
    assert frame.shape == (24, 124) and frame.columns[-1] == "route_id"
    assert len(frame.drop(columns="route_id").columns) == 123
    np.testing.assert_array_equal(scales, np.full(3, 101.))
    for name in ["lag0", "m3h", "m1d", "m7d", "m28d", "slotmean7", "yday", "xwidth"]:
        np.testing.assert_array_equal(frame[name], np.full(24, 100. / 101.))
    np.testing.assert_array_equal(frame["s6_l7"], np.full(24, 60. / 101.))
    np.testing.assert_array_equal(frame["x0"], np.full(24, 50. / 101.))
    np.testing.assert_array_equal(frame["known"], np.r_[np.full(3, 50. / 101.), np.zeros(21)])
    assert frame.sd1d.eq(0).all() and frame.g_ytd6.eq(0).all() and frame.g_last.eq(1).all()
    # The ordered source columns are part of the published predictor contract.
    columns_hash = hashlib.sha256(json.dumps(list(frame), separators=(",", ":")).encode()).hexdigest()
    assert columns_hash == "302540f27285de638dabec716d9795e49969e9c76de9738f47e844c97d7811f9"


def test_source_features_ignore_later_observations_and_match_own_prefix():
    data = random_panel(1800)
    context = V3Context.from_history(data); source = 1550
    expected, scale = context.features(source)
    own, own_scale = V3Context.from_history(data.history(data.timestamps[source])).features(source)
    pd.testing.assert_frame_equal(expected, own, check_exact=True)
    np.testing.assert_array_equal(scale, own_scale)
    target, statuses = data.target.copy(), data.statuses.copy()
    target[source + 1:] = 1e10; statuses[source + 1:] = 1e11
    poisoned = type(data)(data.timestamps, data.routes, statuses, target)
    changed, changed_scale = V3Context.from_history(poisoned).features(source)
    pd.testing.assert_frame_equal(expected, changed, check_exact=True)
    np.testing.assert_array_equal(scale, changed_scale)


@pytest.mark.parametrize("days,count", [(30, 11), (55, 286), (80, 462)])
def test_source_counts_follow_available_mature_history(days, count):
    context = V3Context.from_history(constant_panel(days * 48 + 22))
    sources = context.training_sources()
    assert len(sources) == count
    assert (sources + 8 < len(context.times)).all()
    assert (context.slots[sources] >= 16).all() and (context.slots[sources] <= 26).all()
    assert context.times[sources[-1]] == context.origin.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
    assert context.times[sources[-1] + 8] == context.origin.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=17)


def test_daily_feature_clipping_preserves_raw_labels_and_input():
    data = constant_panel(); data.target[29 * 48 + 17] = 10000.
    context = V3Context.from_history(data)
    np.testing.assert_array_equal(context.target[29 * 48 + 17], np.full(3, 401.))
    np.testing.assert_array_equal(context.raw_target[29 * 48 + 17], np.full(3, 10000.))
    np.testing.assert_array_equal(data.target[29 * 48 + 17], np.full(3, 10000.))
