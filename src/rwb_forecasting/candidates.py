"""Construct predictors from explicit, versioned experiment configurations."""

from functools import partial

from .models import predict_ridge


def configured_predictor(spec: dict):
    kind = spec["kind"]
    params = spec["params"]
    if kind == "ridge":
        return partial(predict_ridge, **params)
    if kind == "linear":
        from .linear_candidates import predict_linear_candidate
        return partial(predict_linear_candidate, **params)
    if kind == "boosting":
        from .boosting import BoostConfig, predict_boosting
        return partial(predict_boosting, config=BoostConfig(**params))
    raise ValueError(f"Unknown predictor kind: {kind}")
