"""Run the unchanged latent MLP preparation with a fresh Torch-only trainer process."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .mlp_spec import specification_sha

_ARRAYS = {"x_train", "y_train", "weight_train", "route_train", "x_infer", "route_infer",
           "route_ids", "route_categories", "source_timestamps_ns"}
_SOURCES = ("mlp_isolation.py", "mlp_tensor_child.py", "mlp_torch.py", "mlp_spec.py")
_TIMEOUT_SECONDS = 7200


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_hashes():
    return {name: _sha(Path(__file__).with_name(name)) for name in _SOURCES}


def _validate_payload(payload, audit):
    if set(payload) != _ARRAYS:
        raise ValueError("Unexpected latent MLP payload fields")
    x, y, w, routes, now, now_routes = (payload[name] for name in
        ("x_train", "y_train", "weight_train", "route_train", "x_infer", "route_infer"))
    categories, route_ids, sources = (payload[name] for name in
        ("route_categories", "route_ids", "source_timestamps_ns"))
    n, r = len(x), len(categories)
    if (not n or not r or x.shape != (n, 78) or y.shape != (n, 8) or w.shape != (n,)
            or routes.shape != (n,) or now.shape != (r, 78) or now_routes.shape != (r,)
            or route_ids.shape != (r,) or categories.shape != (r,)
            or sources.ndim != 1 or len(sources) * r != n
            or any(a.dtype != np.float32 for a in (x, y, w, now))
            or any(a.dtype != np.int64 for a in (routes, now_routes, route_ids, categories, sources))
            or any(not np.isfinite(a).all() for a in (x, y, w, now))
            or (y < 0).any() or (w <= 0).any() or abs(float(w.mean(dtype=np.float64)) - 1) > 1e-5
            or not np.array_equal(categories, np.unique(categories))
            or not np.array_equal(route_ids, categories)
            or not np.array_equal(now_routes, np.arange(r))
            or not np.array_equal(routes, np.tile(np.arange(r), len(sources)))
            or (np.diff(sources) != 30 * 60 * 10**9).any()):
        raise ValueError("Invalid finite, source-major latent MLP tensors")
    origin = datetime.fromisoformat(audit["origin"])
    if origin.tzinfo is not None:
        raise ValueError("Expected timezone-naive original MLP timestamps")
    stamp = lambda key: int(np.datetime64(audit[key], "ns").astype(np.int64))
    if (audit["training_rows"] != n or audit["training_sources"] != len(sources)
            or audit["route_categories"] != r or audit["numeric_features"] != 78
            or audit["specification_sha256"] != specification_sha()
            or stamp("first_source_timestamp") != int(sources[0])
            or stamp("last_source_timestamp") != int(sources[-1])
            or stamp("latest_label_timestamp") != int(sources[-1]) + 8 * 30 * 60 * 10**9
            or stamp("latest_label_timestamp") > stamp("origin")
            or stamp("latest_feature_timestamp") > stamp("origin")):
        raise ValueError("Training audit does not match the mature supervised payload")


def _validate_completion(audit, rows):
    identity = audit["identity"]
    config = identity["training_config"]
    if (audit["epochs_completed"] != 20 or audit["shuffle_cursor"] != 0
            or audit["steps"] != 20 * ((rows + 4095) // 4096)
            or audit["training_rows"] != rows or audit["batch_size"] != 4096
            or audit["synthetic_stop_hook_used"] or audit["device"] != "cpu"
            or audit["native_threads"] != 2 or audit["interop_threads"] != 2
            or audit["dtype"] != "float32" or len(audit["training_loss_by_epoch"]) != 20
            or not np.isfinite(audit["training_loss_by_epoch"]).all()
            or config != {"epochs": 20, "batch_size": 4096, "learning_rate": .001,
                          "weight_decay": .0001, "seed": 42, "device": "cpu"}
            or identity["specification_sha256"] != specification_sha()
            or identity["torch_version"] != "2.15.0"
            or identity["native_threads"] != 2 or identity["interop_threads"] != 2
            or identity["deterministic_algorithms"] is not True
            or audit["eval_set"] is not None or audit["early_stopping"] is not False
            or audit["gradient_clipping"] is not None):
        raise ValueError("Child did not complete the unchanged fixed20epoch CPU2 recipe")


def _isolated_trainer(payload, training_audit, checkpoint, directory):
    _validate_payload(payload, training_audit)
    payload_path = directory / "payload.npz"
    with payload_path.open("xb") as stream:
        np.savez(stream, **payload)
    request = dict(schema=1, payload_sha256=_sha(payload_path), training_audit=training_audit,
                   checkpoint=str(checkpoint), source_sha256=_source_hashes(),
                   specification_sha256=specification_sha(), parent_pid=os.getpid())
    request_path = directory / "request.json"
    with request_path.open("x") as stream:
        stream.write(json.dumps(request, sort_keys=True, indent=2) + "\n")
    request_sha = _sha(request_path)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        env[name] = "2"
    # Use this package's exact checkout even when the caller was imported without PYTHONPATH.
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with (directory / "child.log").open("x") as log:
        try:
            child = subprocess.run([sys.executable, "-m", "rwb_forecasting.mlp_tensor_child",
                                    str(request_path), request_sha], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=_TIMEOUT_SECONDS,
                                   check=False, close_fds=True)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"Torch child exceeded {_TIMEOUT_SECONDS}s; no retry; see {directory / 'child.log'}") from error
    if child.returncode:
        raise RuntimeError(f"Torch child failed with exit{child.returncode}; no retry; see {directory / 'child.log'}")
    if (_sha(request_path) != request_sha or _sha(payload_path) != request["payload_sha256"]
            or _source_hashes() != request["source_sha256"]):
        raise ValueError("Changed trainer sources or request during child execution")
    response_path = directory / "response.npz"
    receipt = json.loads((directory / "receipt.json").read_text())
    if (receipt["request_sha256"] != request_sha or receipt["response_sha256"] != _sha(response_path)
            or receipt["checkpoint_sha256"] != _sha(checkpoint)
            or receipt["source_sha256"] != request["source_sha256"]
            or receipt["pid"] == os.getpid() or receipt["lightgbm_imported"]
            or receipt["feature_preparation_imported"] or receipt["permutation_length"] != 0
            or receipt["checkpoint_reload_bitwise"] is not True):
        raise ValueError("Invalid isolated trainer receipt")
    with np.load(response_path, allow_pickle=False) as response:
        if set(response.files) != {"normalized_half", "audit"}:
            raise ValueError("Unexpected child response arrays")
        normalized = response["normalized_half"].copy()
        audit = json.loads(str(response["audit"].item()))
    if (normalized.shape != (len(payload["route_ids"]), 8) or normalized.dtype != np.float32
            or not np.isfinite(normalized).all() or (normalized < 0).any()):
        raise ValueError("Invalid isolated normalized half forecasts")
    _validate_completion(audit, len(payload["x_train"]))
    audit["process_isolation"] = {**receipt, "request": str(request_path)}
    return normalized, audit


def predict_latent_mlp_isolated(history, checkpoint_path=None):
    """Original fresh fit with preparation in the caller and Torch in a new process.

    Persistent checkpoints retain `.preprocessing.npz`, `.pt` and a sibling
    `.tensor` directory containing numeric input, request, response and receipt.
    Existing artifacts fail closed; this helper never resumes or retries a fit.
    With no checkpoint path these artifacts live in a temporary directory.
    """
    from .mlp_model import predict_latent_mlp

    if checkpoint_path is None:
        with tempfile.TemporaryDirectory(prefix="latent-mlp-") as temporary:
            return predict_latent_mlp_isolated(history, Path(temporary) / "mlp.pt")
    checkpoint = Path(checkpoint_path).resolve()
    if checkpoint.exists() or checkpoint.with_suffix(".preprocessing.npz").exists():
        raise FileExistsError("Fresh isolated MLP requires new model and preprocessing paths")
    directory = checkpoint.with_suffix(".tensor")
    directory.mkdir(parents=True, exist_ok=False)
    called = False

    def trainer(payload, audit):
        nonlocal called
        if called:
            raise RuntimeError("The isolated MLP permits exactly one trainer call")
        called = True
        return _isolated_trainer(payload, audit, checkpoint, directory)

    return predict_latent_mlp(history, checkpoint_path=checkpoint, resume=False, tensor_trainer=trainer)
