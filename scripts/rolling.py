"""Rolling forecasts from every half-hour of the validation weeks, plus the production fit.

The model is fitted once at the start of each validation week (labels up to that moment only)
and then forecasts from all 336 half-hourly origins of the week, as a daily-retrained service
would, only more conservatively. The output feeds interval calibration, the truck-dispatch
simulation and intraday bias monitoring.

    python scripts/rolling.py --folds 2 --final
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.backtest import default_folds
from rwb.data import HORIZON, ROOT, load_panel
from rwb.features import WEEK, cumsums
from rwb.forecaster import LEAN, Forecaster

log = logging.getLogger("rolling")
OUT = ROOT / "models/rolling"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="lean")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--final", action="store_true", help="also fit on all data and save the production bundle")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    panel = load_panel()
    cums = cumsums(panel)
    last = panel.n_times - 1
    OUT.mkdir(parents=True, exist_ok=True)
    for fold in default_folds(panel, n_folds=a.folds):
        t0 = time.time()
        cut = fold.label_cutoff
        # Every origin of the week whose 8 labels are already known.
        origins = np.arange(cut, min(cut + WEEK, last - HORIZON + 1))
        fc = Forecaster(LEAN).fit(panel, cums, cut)
        pred = fc.predict(panel, cums, origins)
        np.savez_compressed(OUT / f"{a.name}_{fold.name}.npz", origins=origins, pred=pred, label_cutoff=cut)
        log.info("%s: cutoff %s, %d origins in %.0fs", fold.name, panel.times[cut], len(origins), time.time() - t0)

    if a.final:
        t0 = time.time()
        fc = Forecaster(LEAN).fit(panel, cums, last)
        bundle = ROOT / f"models/prod/{a.name}"
        fc.save(bundle, {"data_until": str(panel.times[last]), "routes": int(panel.n_routes)})
        np.save(OUT / f"{a.name}_test.npy", fc.predict(panel, cums, np.array([last]))[0])
        log.info("production bundle %s in %.0fs", bundle, time.time() - t0)


if __name__ == "__main__":
    main()
