"""Rebuild V3, the route-share MLP and both fixed references on each history prefix."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwb_forecasting.candidates import configured_comparison
from rwb_forecasting.data import ForecastData
from rwb_forecasting.experiments import compare_models
from rwb_forecasting.known_offset import CANDIDATE as H1, replace_h1_offset
from rwb_forecasting.mlp_model import predict_latent_mlp
from rwb_forecasting.mlp_submission import CANDIDATE, compose_v3_mlp
from rwb_forecasting.v3_boosting import CANDIDATE as RAW, predict as predict_v3

MAE = "mae_share_mass25"
MASS = "latent_mass_blend50"


def predictors(directory, include_known_h1=False):
    recipes = {name: json.loads((ROOT / "configs" / (name + ".json")).read_text())["models"][name]
               for name in (MAE, MASS)}
    refs = configured_comparison(recipes)
    last_history, forecasts = None, {}

    def get(history, name):
        nonlocal last_history, forecasts
        if last_history is not history:
            v3, v3_audit = predict_v3(history, directory)
            mae, mae_audit = refs[MAE](history)
            mass, mass_audit = refs[MASS](history)
            raw_mlp, mlp_audit = predict_latent_mlp(history)
            result, blend = compose_v3_mlp(np.ascontiguousarray(mae), np.ascontiguousarray(v3), raw_mlp)
            forecasts = {RAW: (v3, v3_audit), MAE: (mae, mae_audit), MASS: (mass, mass_audit),
                         CANDIDATE: (result, {"v3": v3_audit, "mae25": mae_audit,
                                              "mlp": mlp_audit, "composition": blend})}
            if include_known_h1:
                forecasts[H1] = (replace_h1_offset(history.target, result),
                                 {"incumbent": forecasts[CANDIDATE][1], "new_fitted_parameters": False,
                                  "latest_observation_timestamp": str(history.timestamps[-1])})
            last_history = history
        return forecasts[name]

    return {n: (lambda history, key=n: get(history, key)) for n in ((CANDIDATE, RAW, MAE, MASS, H1)
                                                   if include_known_h1 else (CANDIDATE, RAW, MAE, MASS))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--origins", nargs="+", required=True, help="History cutoffs in ISO format")
    parser.add_argument("--output-dir", type=Path, required=True, help="New results directory")
    parser.add_argument("--include-known-h1", action="store_true", help="Also score the full-history H1 offset")
    args = parser.parse_args()
    origins = list(pd.to_datetime(args.origins))
    if any(o.tzinfo is not None or o.time() != pd.Timestamp("10:30").time() for o in origins):
        parser.error("This benchmark uses timezone-naive 10:30 origins")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    panel = ForecastData.from_frame(pd.read_parquet(args.train, use_threads=False))
    with tempfile.TemporaryDirectory(dir=args.output_dir) as work, threadpool_limits(limits=2):
        results = compare_models(panel, origins, predictors(Path(work), args.include_known_h1), progress=print)
    for key in ("folds", "summary", "by_horizon", "predictions"):
        results[key].to_csv(args.output_dir / (key + ".csv"), index=False)
    (args.output_dir / "audit.json").write_text(json.dumps(results["audits"], indent=2) + "\n")
    print(results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
