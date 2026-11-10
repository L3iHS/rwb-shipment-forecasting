"""Private fresh-process entry point: no panel reads, feature build or LightGBM."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np

from .mlp_isolation import _sha, _source_hashes, _validate_completion, _validate_payload
from .mlp_spec import specification_sha


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("Expected request JSON and its SHA-256")
    request_path = Path(args[0]).resolve()
    if _sha(request_path) != args[1]:
        raise ValueError("Changed child request")
    request = json.loads(request_path.read_text())
    if (set(request) != {"schema", "payload_sha256", "training_audit", "checkpoint", "source_sha256",
                        "specification_sha256", "parent_pid"}
            or request["schema"] != 1 or request["source_sha256"] != _source_hashes()
            or request["specification_sha256"] != specification_sha() or version("torch") != "2.15.0"):
        raise ValueError("Changed child schema, frozen source, specification or Torch runtime")
    directory = request_path.parent
    checkpoint = Path(request["checkpoint"])
    if checkpoint.exists() or not checkpoint.is_absolute() or directory != checkpoint.with_suffix(".tensor"):
        raise ValueError("Fresh tensor checkpoint does not match its transport directory")
    if any((directory / name).exists() for name in ("fit.claim", "response.npz", "receipt.json")):
        raise FileExistsError("This input already claimed its single fit; no automatic retry")
    payload_path = directory / "payload.npz"
    if _sha(payload_path) != request["payload_sha256"]:
        raise ValueError("Changed supervised arrays")
    with np.load(payload_path, allow_pickle=False) as saved:
        payload = {name: saved[name] for name in saved.files}
    _validate_payload(payload, request["training_audit"])
    with (directory / "fit.claim").open("x") as stream:
        stream.write(args[1] + "\n")

    import torch
    from .mlp_torch import (LatentMLP, TrainingConfig, configure_runtime,
                            predict_normalized, train_predict)
    configure_runtime(threads=2, interop_threads=2)
    if "lightgbm" in sys.modules or "rwb_forecasting.mlp_features" in sys.modules:
        raise RuntimeError("Tensor process must not import the native forecasting stack")
    result = train_predict(payload, len(payload["route_categories"]), config=TrainingConfig(),
                           checkpoint_path=checkpoint, resume=False)
    audit = result["audit"]
    _validate_completion(audit, len(payload["x_train"]))
    if len(result["loop"]["permutation"]):
        raise ValueError("Unfinished training permutation")
    native = request["training_audit"]
    audit.update(origin=native["origin"], first_source_timestamp=native["first_source_timestamp"],
                 last_source_timestamp=native["last_source_timestamp"],
                 latest_label_timestamp=native["latest_label_timestamp"],
                 future_scoring_truth_received=False, training_labels_known_by_origin=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["identity"] != result["identity"] or saved["loop"]["epoch"] != 20 or saved["loop"]["cursor"] != 0:
        raise ValueError("Saved model does not describe the completed fit")
    reloaded = LatentMLP(len(payload["route_categories"]))
    reloaded.load_state_dict(saved["model_state_dict"])
    repeated = predict_normalized(reloaded, torch.from_numpy(payload["x_infer"]),
                                  torch.from_numpy(payload["route_infer"]))
    np.testing.assert_array_equal(repeated.view("u4"), result["normalized_half"].view("u4"))
    if "lightgbm" in sys.modules or "rwb_forecasting.mlp_features" in sys.modules:
        raise RuntimeError("Native forecasting stack was loaded into the tensor process")
    response_path = directory / "response.npz"
    with response_path.open("xb") as stream:
        np.savez(stream, normalized_half=result["normalized_half"],
                 audit=np.array(json.dumps(audit, sort_keys=True)))
    receipt = dict(request_sha256=args[1], response_sha256=_sha(response_path),
                   checkpoint_sha256=_sha(checkpoint), source_sha256=_source_hashes(),
                   pid=os.getpid(), parent_pid=request["parent_pid"],
                   lightgbm_imported=False, feature_preparation_imported=False,
                   permutation_length=0, checkpoint_reload_bitwise=True)
    with (directory / "receipt.json").open("x") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
