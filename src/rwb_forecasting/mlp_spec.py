"""Fixed latent MLP feature order and statistical recipe; no runtime side effects."""
import hashlib
import json

CANDIDATE = "latent_mlp128_share50"

def feature_names():
    statuses = [f"status_{i}" for i in range(1, 7)]
    return [*statuses, *[f"target_lag_{i}" for i in (0, 1, 2, 3, 6, 12)],
        *[f"target_mean_{i}" for i in (6, 12, 48)],
        *[f"target_aligned_{period}_{h}" for period in (48, 336) for h in range(1, 9)],
        *[f"{name}_lag_{lag}" for name in statuses for lag in (1, 2, 3, 6, 12)],
        *[f"{name}_mean_{window}" for name in statuses for window in (6, 12)],
        "slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "log_source_scale"]

def statistical_spec():
    return {"train_days": 14, "source_stride": 1, "horizon": 8, "status_dynamics": True,
        "normalization": "source-causal raw hourly target mean48 floor1000",
        "numeric_features": 78, "calendar": "sin/cos slot_of_day and integer day_of_week",
        "scaler": "StandardScaler on all78training numeric columns only",
        "route_encoder": "sorted unique training route IDs only", "route_embedding": 8,
        "hidden": [128, 128], "activation": "ReLU", "output": "8Softplus(beta1,threshold20)",
        "initialization": "seeded PyTorch nn.Linear/nn.Embedding defaults",
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
        "optimizer_parameters": "all model parameters; foreach=False; fused=False",
        "batch_size": 4096, "epochs": 20, "drop_last": False,
        "shuffle": "one torch randperm per epoch using a dedicated seeded CPU generator",
        "loss": "mean(sample_weight*mean8((prediction-normalized_latent_label)^2))",
        "sample_weight": "source raw hourly scale / mean training-row scale",
        "gradient_clipping": None, "eval_set": None, "early_stopping": False,
        "dropout": 0.0, "mixed_precision": False, "torch_compile": False,
        "seed": 42, "device": "cpu", "dtype": "float32", "native_threads": 2, "interop_threads": 2,
        "deterministic_algorithms": True, "rounding_tolerance": 0.5,
        "composition": "mass_preserving_blend(MAE25,raw_mlp_hourly,.5)",
        "share_weight": 0.5, "preserve_h1": True, "preserve_predicted_horizon_totals": True}

def specification_sha():
    return hashlib.sha256(json.dumps(statistical_spec(), sort_keys=True).encode()).hexdigest()
