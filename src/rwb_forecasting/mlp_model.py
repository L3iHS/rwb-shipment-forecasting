"""Fresh-history original latent MLP; Torch is an optional, lazily loaded backend."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json

import numpy as np

from .experiments import validate_training_boundary
from .mlp_features import prepare_training, reconstruct_hourly
from .mlp_spec import specification_sha


def _preprocessing(path, state, audit, resume):
    path = Path(path)
    metadata = dict(origin=audit["origin"], specification_sha256=specification_sha(),
                    training_rows=audit["training_rows"], latest_label_timestamp=audit["latest_label_timestamp"])
    if path.exists():
        if not resume:
            raise FileExistsError("Preprocessing state exists; use explicit resume with the same history")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(state) | {"metadata"} or json.loads(str(saved["metadata"].item())) != metadata:
                raise ValueError("Preprocessing belongs to a different origin or recipe")
            for name, array in state.items():
                np.testing.assert_array_equal(array, saved[name])
    else:
        if resume:
            raise FileNotFoundError("Resume requires the saved preprocessing state")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez(stream, **state, metadata=np.array(json.dumps(metadata, sort_keys=True)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_tensor_trainer(payload, training_audit, *, checkpoint_path=None, resume=False):
    """Exactly20epochs, AdamW, CPU2, no evaluation or early stopping."""
    try:
        import torch
        from .mlp_torch import TrainingConfig, configure_runtime, train_predict
    except ImportError as error:
        raise ImportError("The original latent MLP requires the optional torch==2.15.0 package") from error
    if version("torch") != "2.15.0":
        raise ValueError("The fixed latent MLP runtime requires torch2.15.0")
    configure_runtime(threads=2, interop_threads=2)
    result = train_predict(payload, len(payload["route_categories"]), config=TrainingConfig(),
                           checkpoint_path=checkpoint_path, resume=resume)
    audit = result["audit"]
    if (audit["epochs_completed"] != 20 or audit["shuffle_cursor"] != 0
            or audit["steps"] != 20 * ((len(payload["x_train"]) + 4095) // 4096)
            or audit["synthetic_stop_hook_used"] or audit["device"] != "cpu"):
        raise ValueError("The complete fixed20epoch CPU recipe is required")
    audit.update(origin=training_audit["origin"],
        first_source_timestamp=training_audit["first_source_timestamp"],
        last_source_timestamp=training_audit["last_source_timestamp"],
        latest_label_timestamp=training_audit["latest_label_timestamp"],
        future_scoring_truth_received=False, training_labels_known_by_origin=True)
    return result["normalized_half"], audit


def predict_latent_mlp(history, *, checkpoint_path=None, resume=False, tensor_trainer=None):
    """Return raw hourly forecasts from one fit on the supplied history prefix.

    The optional callback supports generated tests and alternative process
    transports; it receives only causally prepared supervised tensors and audit.
    The public CLI always uses the fixed trainer above.
    """
    if tensor_trainer is None:
        try:
            backend_version = version("torch")
        except PackageNotFoundError as error:
            raise ImportError("The original latent MLP requires the optional torch==2.15.0 package") from error
        if backend_version != "2.15.0":
            raise ValueError("The fixed latent MLP runtime requires torch2.15.0")
    if resume and (checkpoint_path is None or not Path(checkpoint_path).is_file()):
        raise ValueError("Resume requires an existing checkpoint")
    if checkpoint_path is not None and Path(checkpoint_path).exists() and not resume:
        raise FileExistsError("Checkpoint exists; select a new path or explicitly resume")
    payload, state, training_audit = prepare_training(history)
    preprocessing = None
    if checkpoint_path is not None:
        preprocessing = _preprocessing(Path(checkpoint_path).with_suffix(".preprocessing.npz"),
                                       state, training_audit, resume)
    if tensor_trainer is None:
        normalized, tensor_audit = fixed_tensor_trainer(payload, training_audit,
            checkpoint_path=checkpoint_path, resume=resume)
    else:
        normalized, tensor_audit = tensor_trainer(payload, training_audit)
    validate_training_boundary(tensor_audit, history.timestamps[-1])
    raw = reconstruct_hourly(normalized, state)
    return raw, {**training_audit, "model": "latent_mlp128_raw_hourly", "mlp_training": training_audit,
        "tensor_training": tensor_audit, "preprocessing_sha256": preprocessing,
        "raw_mlp_is_diagnostic_only": True, "platform_submission": False}
