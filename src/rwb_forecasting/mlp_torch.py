"""Compact explicit-device latent MLP and deterministic resumable training."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .mlp_spec import specification_sha


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"


class LatentMLP(nn.Module):
    """numeric:(B,78), route:(B,) -> nonnegative normalized half-flow:(B,8)."""
    def __init__(self, routes: int):
        super().__init__()
        if routes < 1:
            raise ValueError("At least one learned training route is required")
        # Seeded PyTorch defaults are the declared initialization policy.
        self.embedding = nn.Embedding(routes, 8)
        self.layers = nn.Sequential(nn.Linear(78 + 8, 128), nn.ReLU(inplace=False),
                                    nn.Linear(128, 128), nn.ReLU(inplace=False),
                                    nn.Linear(128, 8), nn.Softplus(beta=1, threshold=20))

    def forward(self, numeric, route):
        if numeric.ndim != 2 or numeric.shape[1] != 78 or route.shape != (len(numeric),):
            raise ValueError("MLP expects numeric(B,78) and route(B) inputs")
        return self.layers(torch.cat([numeric, self.embedding(route)], dim=1))


def configure_runtime(threads=2, interop_threads=2):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != interop_threads:
        torch.set_num_interop_threads(interop_threads)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def array_identity(payload):
    digest = hashlib.sha256()
    for name in ("x_train", "y_train", "weight_train", "route_train", "x_infer", "route_infer"):
        value = np.ascontiguousarray(payload[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _rng_state(generator):
    numpy = np.random.get_state()
    return {"torch_rng": torch.get_rng_state(), "shuffle_rng": generator.get_state(),
            "python_rng": random.getstate(),
            "numpy_rng": {"name": numpy[0], "keys": torch.tensor(numpy[1].astype(np.int64)),
                          "position": int(numpy[2]), "has_gauss": int(numpy[3]), "cached_gauss": float(numpy[4])}}


def _restore_rng(state, generator):
    torch.set_rng_state(state["torch_rng"].cpu())
    generator.set_state(state["shuffle_rng"].cpu())
    random.setstate(state["python_rng"])
    numpy = state["numpy_rng"]
    np.random.set_state((numpy["name"], numpy["keys"].cpu().numpy().astype(np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gauss"]))


def _config_identity(config, routes, input_identity):
    return {"training_config": asdict(config), "routes": routes, "input_identity": input_identity,
            "specification_sha256": specification_sha(), "dtype": "float32",
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "native_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def save_checkpoint(path, model, optimizer, loop, generator, identity):
    """Only tensors and primitive containers; load uses weights_only=True."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"identity": identity, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "loop": loop,
               "rng": _rng_state(generator)}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, generator, identity, rows):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["identity"] != identity:
        raise ValueError("Checkpoint belongs to a different frozen recipe or supervised input")
    loop = saved["loop"]
    permutation = loop["permutation"].cpu()
    if (not 0 <= loop["cursor"] <= rows or len(permutation) not in (0, rows)
            or (loop["cursor"] and len(permutation) != rows)
            or len(loop["losses"]) != loop["epoch"]):
        raise ValueError("Invalid checkpoint epoch/permutation/cursor")
    if len(permutation):
        if permutation.dtype != torch.int64 or not torch.equal(permutation.sort().values, torch.arange(rows)):
            raise ValueError("Checkpoint shuffle state is not a complete row permutation")
    model.load_state_dict(saved["model_state_dict"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    _restore_rng(saved["rng"], generator)
    loop["permutation"] = permutation
    return loop


def _validated_tensors(payload, routes, device):
    x, y, w, r = (np.asarray(payload[name]) for name in ("x_train", "y_train", "weight_train", "route_train"))
    now, now_routes = (np.asarray(payload[name]) for name in ("x_infer", "route_infer"))
    if (x.ndim != 2 or not len(x) or x.shape[1] != 78 or y.shape != (len(x), 8)
            or w.shape != (len(x),) or r.shape != (len(x),) or now.ndim != 2 or now.shape[1] != 78
            or now_routes.shape != (len(now),) or r.dtype != np.int64 or now_routes.dtype != np.int64
            or any(a.dtype != np.float32 for a in (x, y, w, now))
            or any(not np.isfinite(a).all() for a in (x, y, w, now))
            or (y < 0).any() or (w <= 0).any() or abs(float(w.mean(dtype=np.float64)) - 1) > 1e-5
            or (r < 0).any() or (r >= routes).any() or (now_routes < 0).any() or (now_routes >= routes).any()):
        raise ValueError("Invalid finite float32 supervised tensors or categorical route indices")
    return tuple(torch.from_numpy(a).to(device) for a in (x, y, w, r, now, now_routes))


@torch.inference_mode()
def predict_normalized(model, numeric, routes):
    model.eval()
    result = model(numeric, routes)
    if result.dtype != torch.float32 or not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError("MLP point forecasts must be finite nonnegative float32 values")
    return result.cpu().numpy().copy()


def weighted_mse(prediction, target, weight):
    # Global training-row weight normalization is done by the native exporter.
    # Do not divide by the particular batch's weight sum.
    return ((prediction - target).square().mean(dim=1) * weight).mean()


def train_predict(payload, routes, *, config=TrainingConfig(), checkpoint_path=None,
                  resume=False, stop_after_steps=None, input_identity=None, on_epoch=None):
    """The stop hook is for generated-data resume tests, never a real worker option.

    Production uses20epochs, batch4096 and no stop hook. All rows participate
    each epoch; sample weights are normalized once over the training set.
    """
    if (config.epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0 or config.weight_decay < 0
            or (stop_after_steps is not None and stop_after_steps < 1)):
        raise ValueError("Invalid training configuration")
    device = torch.device(config.device)  # Explicit caller choice; never auto-select hardware.
    seed_all(config.seed, device)
    tensors = _validated_tensors(payload, routes, device)
    x, y, weight, route, now, now_routes = tensors
    model = LatentMLP(routes).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                  foreach=False, fused=False)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    identity = _config_identity(config, routes, input_identity or array_identity(payload))
    loop = {"epoch": 0, "cursor": 0, "permutation": torch.empty(0, dtype=torch.int64),
            "epoch_loss_sum": 0.0, "losses": [], "steps": 0}
    if resume:
        if checkpoint_path is None:
            raise ValueError("Resume requires a checkpoint")
        loop = load_checkpoint(checkpoint_path, model, optimizer, generator, identity, len(x))
    if loop["epoch"] > config.epochs:
        raise ValueError("Checkpoint already exceeds frozen epoch budget")
    start = time.perf_counter()
    model.train()
    performed_steps = 0
    stopped = False
    while loop["epoch"] < config.epochs:
        if not len(loop["permutation"]):
            loop["permutation"] = torch.randperm(len(x), generator=generator)
        end = min(loop["cursor"] + config.batch_size, len(x))
        indices = loop["permutation"][loop["cursor"]:end].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x.index_select(0, indices), route.index_select(0, indices))
        loss = weighted_mse(prediction, y.index_select(0, indices), weight.index_select(0, indices))
        if not torch.isfinite(loss):
            raise FloatingPointError("Frozen training recipe produced nonfinite loss")
        loss.backward()
        optimizer.step()  # No clipping, scheduler, evalset or adaptive stopping.
        batch_rows = end - loop["cursor"]
        loop["epoch_loss_sum"] += float(loss.detach().cpu()) * batch_rows
        loop["cursor"] = end
        loop["steps"] += 1
        performed_steps += 1
        if loop["cursor"] == len(x):
            loop["losses"].append(loop["epoch_loss_sum"] / len(x))
            loop["epoch"] += 1
            loop["cursor"] = 0
            loop["epoch_loss_sum"] = 0.0
            loop["permutation"] = torch.empty(0, dtype=torch.int64)
            if on_epoch is not None:
                on_epoch(loop["epoch"], loop["losses"][-1])  # Logging only; return value is ignored.
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, optimizer, loop, generator, identity)
        if stop_after_steps is not None and performed_steps >= stop_after_steps:
            stopped = True
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, optimizer, loop, generator, identity)
            break
    normalized = predict_normalized(model, now, now_routes)
    audit = {"epochs_completed": loop["epoch"], "shuffle_cursor": loop["cursor"], "steps": loop["steps"],
        "training_rows": len(x), "batch_size": config.batch_size, "seed": config.seed,
        "training_seconds": time.perf_counter() - start, "device": str(device), "dtype": "float32",
        "native_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "loss": "batchsize mean of globally mean-normalized weighted row MSE over8outputs",
        "training_loss_by_epoch": list(loop["losses"]), "synthetic_stop_hook_used": stopped,
        "eval_set": None, "early_stopping": False, "gradient_clipping": None,
        "parameters": sum(p.numel() for p in model.parameters()), "identity": identity}
    return {"model": model, "optimizer": optimizer, "loop": loop, "generator": generator,
            "normalized_half": normalized, "audit": audit, "identity": identity}
