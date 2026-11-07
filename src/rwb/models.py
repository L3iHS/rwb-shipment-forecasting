"""Forecasters. Each returns predictions shaped (n_origins, R, HORIZON)."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rwb.data import HORIZON, Panel
from rwb.features import (
    DAY,
    MAX_LOOKBACK,
    WEEK,
    Cumsums,
    base_forecast,
    build,
    flat_scale,
    flat_target,
)

log = logging.getLogger(__name__)


def _unflat(v: np.ndarray, n_o: int, R: int) -> np.ndarray:
    return v.reshape(n_o, R)


def naive(panel: Panel, origins: np.ndarray, kind: str, cums: Cumsums | None = None) -> np.ndarray:
    """Simple references: last value, same slot yesterday / last week, 7-day slot mean, seasonal base."""
    out = np.empty((len(origins), panel.n_routes, HORIZON))
    for h in range(1, HORIZON + 1):
        t = origins + h
        if kind == "last":
            v = panel.y[:, origins]
        elif kind == "day":
            v = panel.y[:, t - DAY]
        elif kind == "week":
            v = panel.y[:, t - WEEK]
        elif kind == "slot7":
            v = np.mean([panel.y[:, t - k * DAY] for k in range(1, 8)], axis=0)
        elif kind == "base":
            assert cums is not None
            v = base_forecast(panel, cums, origins, h)
        else:
            raise ValueError(kind)
        out[:, :, h - 1] = v.T
    return out


@dataclass
class GBMConfig:
    lib: str = "lgbm"  # lgbm | xgb | cat
    days: int = 56
    stride: int = 2
    rounds: int = 800
    lr: float = 0.05
    leaves: int = 127
    min_child: int = 200
    feature_fraction: float = 0.7
    bagging: float = 0.7
    l2: float = 1.0
    objective: str = "l1"
    # Up-weight training origins near the test clock time (slot 21 = 10:30).
    tod_weight: float = 0.0
    tod_center: int = 21
    tod_width: float = 4.0
    route_cat: bool = False
    # Label denoising: cap at a per-route quantile; blend with neighbouring steps.
    clip_q: float = 0.0
    label_smooth: float = 0.0
    pooled: bool = False
    global_feats: bool = False
    flow_feats: bool = False
    decay: float = 1.0
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _origin_weights(panel: Panel, origins: np.ndarray, cfg: GBMConfig) -> np.ndarray:
    w = np.ones(len(origins))
    if cfg.decay < 1.0:
        # Recent behaviour matters more; age in days relative to the newest training origin.
        w *= cfg.decay ** ((origins.max() - origins) / DAY)
    if cfg.tod_weight > 0:
        ts = panel.times[origins]
        slot = (ts.hour * 2 + ts.minute // 30).to_numpy()
        dist = np.minimum(np.abs(slot - cfg.tod_center), 48 - np.abs(slot - cfg.tod_center))
        w *= 1.0 + cfg.tod_weight * np.exp(-0.5 * (dist / cfg.tod_width) ** 2)
    return w


def _fit_one(X: np.ndarray, y: np.ndarray, w: np.ndarray, names: list[str], cfg: GBMConfig) -> Any:
    if cfg.lib == "lgbm":
        import lightgbm as lgb

        params = {
            "objective": cfg.objective, "learning_rate": cfg.lr, "num_leaves": cfg.leaves,
            "min_child_samples": cfg.min_child, "feature_fraction": cfg.feature_fraction,
            "bagging_fraction": cfg.bagging, "bagging_freq": 1, "lambda_l2": cfg.l2,
            "seed": cfg.seed, "verbose": -1, "num_threads": 0, "max_bin": 255,
        }
        params.update(cfg.extra)
        ds = lgb.Dataset(X, y, weight=w, feature_name=names, categorical_feature=["route"] if cfg.route_cat else [], free_raw_data=True)
        return lgb.train(params, ds, num_boost_round=cfg.rounds)
    if cfg.lib == "xgb":
        import xgboost as xgb

        obj = {"l1": "reg:absoluteerror", "l2": "reg:squarederror"}[cfg.objective]
        params = {
            "objective": obj, "learning_rate": cfg.lr, "max_depth": 0, "max_leaves": cfg.leaves, "grow_policy": "lossguide",
            "min_child_weight": cfg.min_child, "subsample": cfg.bagging, "colsample_bytree": cfg.feature_fraction,
            "reg_lambda": cfg.l2, "tree_method": "hist", "device": "cuda", "seed": cfg.seed, "max_bin": 256,
        }
        params.update(cfg.extra)
        d = xgb.QuantileDMatrix(X, y, weight=w, feature_names=names)
        return xgb.train(params, d, num_boost_round=cfg.rounds)
    if cfg.lib == "cat":
        from catboost import CatBoostRegressor, Pool

        loss = {"l1": "MAE", "l2": "RMSE"}[cfg.objective]
        params = {
            "loss_function": loss, "learning_rate": cfg.lr, "depth": 8, "iterations": cfg.rounds, "task_type": "GPU",
            "random_seed": cfg.seed, "verbose": 0, "l2_leaf_reg": cfg.l2, "border_count": 128,
        }
        params.update(cfg.extra)
        m = CatBoostRegressor(**params)
        m.fit(Pool(X, y, weight=w, feature_names=names))
        return m
    raise ValueError(cfg.lib)


def _predict_one(model: Any, X: np.ndarray, names: list[str], cfg: GBMConfig) -> np.ndarray:
    if cfg.lib == "xgb":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X, feature_names=names))
    return model.predict(X)


def _train_set(panel: Panel, cums: Cumsums, label_cutoff: int, anchor: int, h: int,
               cfg: GBMConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    R = panel.n_routes
    # A smoothed label peeks one step past o+h, so keep that step inside the cutoff too.
    hi = label_cutoff - h - (1 if cfg.label_smooth > 0 else 0)
    # Align the subsampled grid with the forecast clock time so that slot is always present.
    hi -= (hi - anchor) % cfg.stride
    lo = max(MAX_LOOKBACK, hi - cfg.days * DAY)
    tr = np.arange(hi, lo - 1, -cfg.stride)[::-1]
    X, names, base = build(panel, cums, tr, h, cfg.global_feats, cfg.flow_feats)
    scale = flat_scale(cums, tr)
    y = flat_target(panel, tr, h)
    if cfg.label_smooth > 0:
        a = cfg.label_smooth
        y = (1 - 2 * a) * y + a * (flat_target(panel, tr, h - 1) + flat_target(panel, tr, h + 1))
    if cfg.clip_q > 0:
        # Per-route cap estimated on the training window only.
        cap = np.quantile(panel.y[:, lo : label_cutoff + 1], cfg.clip_q, axis=1)
        y = np.minimum(y, np.tile(cap, len(tr)))
    target = (y - base) / scale
    w = scale * np.repeat(_origin_weights(panel, tr, cfg), R)
    return X, target, w, names


def _predict_h(model: Any, panel: Panel, cums: Cumsums, pred_origins: np.ndarray, h: int,
               cfg: GBMConfig, pooled: bool) -> np.ndarray:
    Xp, names, base_p = build(panel, cums, pred_origins, h, cfg.global_feats, cfg.flow_feats)
    if pooled:
        Xp = np.column_stack([Xp, np.full(len(Xp), h, dtype=np.float32)])
        names = names + ["h"]
    pred = base_p + flat_scale(cums, pred_origins) * _predict_one(model, Xp, names, cfg)
    return _unflat(np.clip(pred, 0, None), len(pred_origins), panel.n_routes)


def gbm_fit_predict(panel: Panel, cums: Cumsums, label_cutoff: int, pred_origins: np.ndarray,
                    cfg: GBMConfig) -> np.ndarray:
    """Direct strategy learning (y - base) / scale with L1 weighted by scale.

    Weighted L1 on the scaled residual equals the sum of absolute errors in original units,
    i.e. the WAPE numerator, so big routes matter exactly as much as in the metric.
    With ``cfg.pooled`` one model serves all horizons (horizon is a feature), otherwise one per horizon.
    """
    out = np.empty((len(pred_origins), panel.n_routes, HORIZON))
    anchor = int(pred_origins[0])
    if cfg.pooled:
        parts = [_train_set(panel, cums, label_cutoff, anchor, h, cfg) for h in range(1, HORIZON + 1)]
        X = np.concatenate([np.column_stack([p[0], np.full(len(p[0]), h, dtype=np.float32)])
                            for h, p in enumerate(parts, start=1)])
        target = np.concatenate([p[1] for p in parts])
        w = np.concatenate([p[2] for p in parts])
        names = parts[0][3] + ["h"]
        del parts
        model = _fit_one(X, target, w, names, cfg)
        del X
        log.info("pooled model trained on %d rows", len(target))
        for h in range(1, HORIZON + 1):
            out[:, :, h - 1] = _predict_h(model, panel, cums, pred_origins, h, cfg, True)
        return out
    for h in range(1, HORIZON + 1):
        X, target, w, names = _train_set(panel, cums, label_cutoff, anchor, h, cfg)
        model = _fit_one(X, target, w, names, cfg)
        del X
        out[:, :, h - 1] = _predict_h(model, panel, cums, pred_origins, h, cfg, False)
        log.info("h=%d trained on %d rows", h, len(target))
    return out
