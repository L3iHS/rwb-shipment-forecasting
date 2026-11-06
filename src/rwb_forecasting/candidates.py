"""Construct predictors from explicit, versioned experiment configurations."""

from functools import partial
from copy import deepcopy
import json

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
    if kind == "latent":
        from .latent_models import predict_latent_ridge
        return partial(predict_latent_ridge, **params)
    if kind in {"latent_hybrid", "latent_projection"}:
        from .latent_models import hybrid_first_horizon, project_incumbent
        predictor = hybrid_first_horizon if kind == "latent_hybrid" else project_incumbent
        return partial(predictor, incumbent_predictor=configured_predictor(spec["incumbent"]), **params)
    raise ValueError(f"Unknown predictor kind: {kind}")


def configured_comparison(specs: dict):
    """Reuse identical components within one immutable history prefix.

    A comparison passes the same history object to all its predictors, then a
    new object at the next origin. No cache survives a change of that object.
    Copies prevent a composition or caller from modifying a cached forecast.
    """
    built, cached = {}, {}
    current_history = None

    def build(spec):
        key = json.dumps(spec, sort_keys=True)
        if key in built:
            return built[key]
        if spec["kind"] in {"latent_hybrid", "latent_projection"}:
            from .latent_models import hybrid_first_horizon, project_incumbent
            function = hybrid_first_horizon if spec["kind"] == "latent_hybrid" else project_incumbent
            predictor = partial(function, incumbent_predictor=build(spec["incumbent"]), **spec["params"])
        else:
            predictor = configured_predictor(spec)

        def predict(history):
            nonlocal current_history
            if current_history is not history:
                cached.clear()
                current_history = history
            if key not in cached:
                cached[key] = predictor(history)
            prediction, audit = cached[key]
            return prediction.copy(), deepcopy(audit)

        built[key] = predict
        return predict

    return {name: build(spec) for name, spec in specs.items()}
