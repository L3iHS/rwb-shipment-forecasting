import hashlib
import os
import numpy as np
import pytest

from rwb_forecasting.v3_boosting import RuntimeConfig, native_parameters, predict, training_design
from rwb_forecasting.v3_features import V3Context
from v3_test_data import constant_panel, random_panel


def test_training_design_keeps_raw_H1_labels_scale_weights_and_cleans_files(tmp_path):
    data = constant_panel(); data.target[29 * 48 + 17] = 10000.
    context = V3Context.from_history(data)
    with training_design(context, tmp_path) as design:
        folder = design.directory
        assert design.features.dtype == np.float64 and design.features.shape[1] == 124
        assert design.model_matrix.shape[1] == 123 and not design.model_matrix.flags.c_contiguous
        np.testing.assert_array_equal(design.raw_labels[:3], np.full(3, 10000.))
        np.testing.assert_array_equal(design.scales[:3], np.full(3, 101.))
        np.testing.assert_array_equal(design.normalized_labels[:3], np.full(3, 10000. / 101. - 50. / 101.))
    assert not folder.exists() and list(tmp_path.iterdir()) == []


def test_fake_backend_preserves_all_eight_H1_structure_and_shape(tmp_path):
    def fake(design, current, runtime):
        assert runtime.native_threads == 2
        assert native_parameters(runtime)["lambda_l2"] == 1.
        return np.full(len(current), .2)
    prediction, audit = predict(constant_panel(), tmp_path, fitter=fake)
    expected = np.full((3, 8), .2 * 101.); expected[:, 0] = (.2 + 50. / 101.) * 101.
    np.testing.assert_array_equal(prediction, expected)
    assert audit["latest_label_timestamp"] <= audit["origin"]
    assert audit["source_feature_count"] == 124 and audit["model_feature_count"] == 123
    assert list(tmp_path.iterdir()) == []


def test_failure_cleans_memmaps_and_invalid_runtime_fails(tmp_path):
    def failed(*args):
        raise RuntimeError("generated backend failure")
    with pytest.raises(RuntimeError, match="generated"):
        predict(constant_panel(), tmp_path, fitter=failed)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        RuntimeConfig(native_threads=0)


@pytest.mark.skipif(os.environ.get("RWB_V3_NATIVE_TESTS") != "1", reason="optional generated600-tree native smoke")
def test_optional_native600_generated_panel(tmp_path):
    pytest.importorskip("lightgbm")
    prediction, audit = predict(random_panel(), tmp_path, RuntimeConfig(native_threads=2))
    assert prediction.shape == (3, 8) and np.isfinite(prediction).all() and (prediction >= 0).all()
    assert audit["n_estimators"] == 600 and audit["training_sources"] == 44
    (tmp_path / "native_prediction.sha256").write_text(hashlib.sha256(prediction.tobytes()).hexdigest())
