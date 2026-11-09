"""Forecasters. Each returns predictions shaped (n_origins, R, HORIZON)."""

import logging
import re
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
    flat_day_mean,
    flat_scale,
    flat_target,
    flat_x,
    lowrank_profiles,
)

BASE_RANK = 12

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
    # Comma-separated optional feature sets, see rwb.features.EXTRA_SETS.
    feat_sets: str = ""
    # Regex of feature names to leave out (e.g. statuses with no measured importance).
    drop: str = ""
    # >1: separate models per volume tier of routes; averaging with the all-routes model adds diversity.
    route_tiers: int = 1
    # "default": 0.5 * day mean + 0.5 * same slot over 7 days; "lowrank": the 7-day slot mean is
    # replaced by the 28-day cross-route smoothed profile, so the model learns a smaller correction.
    base: str = "default"
    base_days: int = 28  # profile length for the low-rank base
    decay: float = 1.0
    # "y": hourly target directly. "x": half-hour shipments, hourly forecast = x[o+h] + x[o+h-1]
    # with x[o] known exactly, so adjacent horizons share their half-hour estimates.
    # "x1": only the first horizon learns the unknown half-hour; later ones learn y as usual.
    # "ratio": non-negative y/scale without the seasonal base, for Tweedie-style objectives.
    target: str = "y"
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
    if cfg.lib == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Linear reference on the same features; a differently biased member for the blend.
        m = make_pipeline(StandardScaler(), Ridge(alpha=cfg.l2))
        m.fit(X, y, ridge__sample_weight=w)
        return m
    raise ValueError(cfg.lib)


def _extra(cfg: GBMConfig) -> tuple[str, ...]:
    return tuple(s.strip() for s in cfg.feat_sets.split(",") if s.strip())


def _features(panel: Panel, cums: Cumsums, origins: np.ndarray, h: int,
              cfg: GBMConfig) -> tuple[np.ndarray, list[str], np.ndarray]:
    X, names, base = build(panel, cums, origins, h, cfg.global_feats, cfg.flow_feats, _extra(cfg))
    if cfg.base in ("lowrank", "lowrank_mult"):
        smooth = lowrank_profiles(panel.y, origins, BASE_RANK, cfg.base_days)[1]  # (n_o, R, DAY), origin-major
        day = flat_day_mean(cums, origins)
        if cfg.base == "lowrank":
            base = 0.5 * day + 0.5 * smooth[:, :, h - 1].ravel()
        else:
            # Shape from the profile, height from the last 24 hours (multiplicative seasonal model).
            base = smooth[:, :, h - 1].ravel() * (day + 1.0) / (smooth.mean(axis=2).ravel() + 1.0)
    elif cfg.base != "default":
        raise ValueError(cfg.base)
    if cfg.drop:
        keep = [j for j, n in enumerate(names) if not re.search(cfg.drop, n)]
        X, names = X[:, keep], [names[j] for j in keep]
    return X, names, base


def _half_hour_label(cfg: GBMConfig, h: int) -> bool:
    return cfg.target == "x" or (cfg.target == "x1" and h == 1)


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
    X, names, base = _features(panel, cums, tr, h, cfg)
    scale = flat_scale(cums, tr)
    y = flat_target(panel, tr, h)
    if cfg.label_smooth > 0:
        a = cfg.label_smooth
        y = (1 - 2 * a) * y + a * (flat_target(panel, tr, h - 1) + flat_target(panel, tr, h + 1))
    if cfg.clip_q > 0:
        # Per-route cap estimated on the training window only.
        cap = np.quantile(panel.y[:, lo : label_cutoff + 1], cfg.clip_q, axis=1)
        y = np.minimum(y, np.tile(cap, len(tr)))
    if _half_hour_label(cfg, h):
        y = flat_x(cums, tr, h)
        base = 0.5 * base  # the hourly base split evenly over its two half-hours
    elif cfg.target == "ratio":
        base = np.zeros_like(base)  # non-negative label y/scale, for Tweedie/Poisson-style objectives
    target = (y - base) / scale
    w = scale * np.repeat(_origin_weights(panel, tr, cfg), R)
    return X, target, w, names


def _predict_h(model: Any, panel: Panel, cums: Cumsums, pred_origins: np.ndarray, h: int,
               cfg: GBMConfig, pooled: bool) -> np.ndarray:
    Xp, names, base_p = _features(panel, cums, pred_origins, h, cfg)
    if pooled:
        Xp = np.column_stack([Xp, np.full(len(Xp), h, dtype=np.float32)])
        names = names + ["h"]
    if _half_hour_label(cfg, h):
        base_p = 0.5 * base_p
    elif cfg.target == "ratio":
        base_p = np.zeros_like(base_p)
    pred = base_p + flat_scale(cums, pred_origins) * _predict_one(model, Xp, names, cfg)
    return _unflat(np.clip(pred, 0, None), len(pred_origins), panel.n_routes)


def _route_groups(panel: Panel, label_cutoff: int, tiers: int) -> list[np.ndarray]:
    """Split routes into volume tiers by their mean over the 4 weeks up to the cutoff (no future data)."""
    if tiers <= 1:
        return [np.ones(panel.n_routes, dtype=bool)]
    level = np.nan_to_num(panel.y[:, label_cutoff - 4 * WEEK + 1 : label_cutoff + 1]).mean(axis=1)
    edges = np.quantile(level, np.linspace(0, 1, tiers + 1)[1:-1])
    tier = np.searchsorted(edges, level, side="right")
    return [tier == k for k in range(tiers)]


def gbm_fit_predict(panel: Panel, cums: Cumsums, label_cutoff: int, pred_origins: np.ndarray,
                    cfg: GBMConfig) -> np.ndarray:
    """Direct strategy learning (y - base) / scale with L1 weighted by scale.

    Weighted L1 on the scaled residual equals the sum of absolute errors in original units,
    i.e. the WAPE numerator, so big routes matter exactly as much as in the metric.
    With ``cfg.pooled`` one model serves all horizons (horizon is a feature), otherwise one per horizon.
    """
    out = np.empty((len(pred_origins), panel.n_routes, HORIZON))
    anchor = int(pred_origins[0])
    if cfg.pooled and (cfg.target == "x" or cfg.route_tiers > 1):
        raise ValueError("target='x' and route_tiers need one model per horizon")
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
    groups = _route_groups(panel, label_cutoff, cfg.route_tiers)
    for h in range(1, HORIZON + 1):
        X, target, w, names = _train_set(panel, cums, label_cutoff, anchor, h, cfg)
        n_tr = len(target) // panel.n_routes
        for g in groups:
            rows = np.tile(g, n_tr)
            model = _fit_one(X[rows], target[rows], w[rows], names, cfg)
            out[:, g, h - 1] = _predict_h(model, panel, cums, pred_origins, h, cfg, False)[:, g]
        del X
        log.info("h=%d trained on %d rows in %d route group(s)", h, len(target), len(groups))
    x0 = cums.x[:, pred_origins].T  # (n_o, R), known at the origin
    if cfg.target == "x":
        # out holds half-hour estimates; the known x[o] completes the first hour.
        xs = np.concatenate([x0[:, :, None], out], axis=2)
        out = xs[:, :, 1:] + xs[:, :, :-1]
    elif cfg.target == "x1":
        out[:, :, 0] += x0
    return out
