"""Tiny generated-only tensor/checkpoint tests, runnable without pytest."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Optional MLP tests require torch==2.15.0")

from rwb_forecasting.mlp_torch import (LatentMLP, TrainingConfig, configure_runtime, train_predict, weighted_mse)


def generated(rows=137, routes=3):
    rng = np.random.default_rng(123)
    x = rng.normal(size=(rows, 78)).astype(np.float32)
    labels = np.logaddexp(0, x[:, :8] * .2).astype(np.float32)
    weight = rng.uniform(.2, 2, size=rows)
    weight = (weight / weight.mean()).astype(np.float32)
    return {"x_train": x, "y_train": labels, "weight_train": weight,
            "route_train": (np.arange(rows) % routes).astype(np.int64),
            "x_infer": rng.normal(size=(routes, 78)).astype(np.float32),
            "route_infer": np.arange(routes, dtype=np.int64)}


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny functional checks use one CPU thread while the reference fit owns the heavy slot.
        configure_runtime(threads=1, interop_threads=1)

    def test_architecture_shapes_positive_float32_and_exact_parameter_count(self):
        model = LatentMLP(1000).to(device="cpu", dtype=torch.float32)
        result = model(torch.zeros((7, 78)), torch.arange(7, dtype=torch.int64))
        self.assertEqual(tuple(result.shape), (7, 8))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 36680)
        self.assertTrue(torch.isfinite(result).all() and (result >= 0).all())

    def test_weighted_loss_uses_batch_size_not_batch_weight_sum(self):
        prediction = torch.zeros((2, 8))
        target = torch.tensor([[1.] * 8, [2.] * 8])
        weight = torch.tensor([2., 4.])
        self.assertEqual(weighted_mse(prediction, target, weight).item(), 9.)

    def test_mid_epoch_resume_reloads_optimizer_rng_permutation_and_cursor_exactly(self):
        payload = generated()
        config = TrainingConfig(epochs=20, batch_size=32)
        with tempfile.TemporaryDirectory(prefix="mlp_generated_") as folder:
            full_path = Path(folder) / "full.pt"
            partial_path = Path(folder) / "partial.pt"
            full = train_predict(payload, 3, config=config, checkpoint_path=full_path)
            interrupted = train_predict(payload, 3, config=config, checkpoint_path=partial_path, stop_after_steps=7)
            self.assertEqual(interrupted["loop"]["epoch"], 1)
            self.assertEqual(interrupted["loop"]["cursor"], 64)
            self.assertEqual(len(interrupted["loop"]["permutation"]), 137)
            resumed = train_predict(payload, 3, config=config, checkpoint_path=partial_path, resume=True)
            np.testing.assert_array_equal(full["normalized_half"], resumed["normalized_half"])
            assert_tree_equal(self, full["model"].state_dict(), resumed["model"].state_dict())
            assert_tree_equal(self, full["optimizer"].state_dict(), resumed["optimizer"].state_dict())
            assert_tree_equal(self, full["loop"], resumed["loop"])
            saved_full = torch.load(full_path, map_location="cpu", weights_only=True)
            saved_resumed = torch.load(partial_path, map_location="cpu", weights_only=True)
            assert_tree_equal(self, saved_full, saved_resumed)
            self.assertEqual(resumed["loop"]["steps"], 100)  # Includes the final9rows each epoch.
            self.assertEqual(resumed["audit"]["epochs_completed"], 20)
            self.assertEqual(resumed["audit"]["shuffle_cursor"], 0)
            self.assertFalse(resumed["model"].training)
            self.assertTrue(np.isfinite(resumed["audit"]["training_loss_by_epoch"]).all())
            with self.assertRaises(ValueError):
                train_predict(payload, 3, config=replace(config, batch_size=16), checkpoint_path=partial_path, resume=True)

    def test_changed_supervised_data_cannot_reuse_checkpoint(self):
        payload = generated()
        config = TrainingConfig(epochs=1, batch_size=32)
        with tempfile.TemporaryDirectory(prefix="mlp_generated_") as folder:
            path = Path(folder) / "model.pt"
            train_predict(payload, 3, config=config, checkpoint_path=path)
            changed = {key: value.copy() for key, value in payload.items()}
            changed["y_train"][0, 0] += 1
            with self.assertRaises(ValueError):
                train_predict(changed, 3, config=config, checkpoint_path=path, resume=True)


if __name__ == "__main__":
    unittest.main()
