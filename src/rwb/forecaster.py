"""Production forecaster: fit once, then forecast from any origin without refitting.

Wraps the same per-horizon boosting as ``gbm_fit_predict`` (identical features, targets and
weights) but keeps the fitted models, so a daily retrain can serve the half-hourly forecasts
and the bundle can be saved and reloaded.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rwb.data import HORIZON, Panel
from rwb.features import Cumsums
from rwb.models import GBMConfig, _fit_one, _predict_h, _train_set

log = logging.getLogger(__name__)

# Single CPU LightGBM: its Huber loss lands near the mean, so the forecast total needs no fitted
# multiplier (the piece that failed on the leaderboard for v3), and it runs without a GPU.
LEAN = GBMConfig(lib="lgbm", days=28, stride=4, rounds=400, leaves=31, min_child=1000, objective="huber",
                 feat_sets="lowrank12", base="lowrank", extra={"alpha": 1.0})


@dataclass
class Forecaster:
    cfg: GBMConfig
    label_cutoff: int = -1
    models: list[Any] = field(default_factory=list)

    def fit(self, panel: Panel, cums: Cumsums, label_cutoff: int) -> "Forecaster":
        """Train one model per horizon on labels up to ``label_cutoff`` (time index)."""
        if self.cfg.pooled or self.cfg.route_tiers > 1 or self.cfg.target in ("x", "x1"):
            raise ValueError("Forecaster supports the plain per-horizon setup only")
        self.models = []
        for h in range(1, HORIZON + 1):
            X, target, w, names = _train_set(panel, cums, label_cutoff, label_cutoff, h, self.cfg)
            self.models.append(_fit_one(X, target, w, names, self.cfg))
            del X
            log.info("h=%d trained on %d rows", h, len(target))
        self.label_cutoff = label_cutoff
        return self

    def predict(self, panel: Panel, cums: Cumsums, origins: np.ndarray, batch: int = 96) -> np.ndarray:
        """(n_origins, R, HORIZON) forecasts; origins are time indices, data after them is never read."""
        if not self.models:
            raise RuntimeError("fit or load the forecaster first")
        origins = np.asarray(origins)
        out = np.empty((len(origins), panel.n_routes, HORIZON), dtype=np.float32)
        # Feature matrices grow with the number of origins; batches keep memory flat.
        for i in range(0, len(origins), batch):
            o = origins[i : i + batch]
            for h, m in enumerate(self.models, start=1):
                out[i : i + len(o), :, h - 1] = _predict_h(m, panel, cums, o, h, self.cfg, False)
        return out

    def save(self, path: Path, meta: dict[str, Any] | None = None) -> None:
        if self.cfg.lib not in ("lgbm", "xgb"):
            raise ValueError("saving is implemented for lgbm and xgb")
        path.mkdir(parents=True, exist_ok=True)
        ext = "txt" if self.cfg.lib == "lgbm" else "json"
        for h, m in enumerate(self.models, start=1):
            m.save_model(str(path / f"h{h}.{ext}"))
        info = {"cfg": asdict(self.cfg), "label_cutoff": self.label_cutoff, **(meta or {})}
        (path / "forecaster.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Forecaster":
        info = json.loads((path / "forecaster.json").read_text(encoding="utf-8"))
        cfg = GBMConfig(**info["cfg"])
        models: list[Any] = []
        for h in range(1, HORIZON + 1):
            if cfg.lib == "lgbm":
                import lightgbm as lgb

                models.append(lgb.Booster(model_file=str(path / f"h{h}.txt")))
            else:
                import xgboost as xgb

                b = xgb.Booster()
                b.load_model(str(path / f"h{h}.json"))
                models.append(b)
        return cls(cfg, int(info["label_cutoff"]), models)
