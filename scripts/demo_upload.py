"""Two demo files for the "Свой файл" tab, cut from the last 2 hours of the real history (09:00–10:30).

    python scripts/demo_upload.py        # -> frontend/public/demo/*.csv

Both change only status_1..3 (goods passing the processing stages of this warehouse) on 200 routes from 09:30; shipments
(target_1h) stay as stored, so whatever moves in the forecast is the model reading the statuses ahead of the shipments.

zator_0900-1030.csv     a jam at sorting: processing drops to 5 %, the model expects fewer shipments.
vsplesk_0900-1030.csv   a surge of orders: processing doubles, the model expects more shipments.
obrazec.csv             the format: the same columns, 3 routes x 4 half-hours as stored, to download and look at.

A file with the history as it is gives the same forecast as the console, so it is not offered as a demo.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.data import ROOT, STATUS_COLS, load_panel  # noqa: E402

OUT = ROOT / "frontend/public/demo"
STEPS = 4  # two hours
FROM = 1  # the change starts at the second step of the file: 09:30
ROUTES = 200
SCENARIOS = [("zator_0900-1030.csv", 7, 0.05), ("vsplesk_0900-1030.csv", 11, 2.0)]  # file, seed of the routes, statuses x


def main() -> None:
    panel = load_panel()
    o = panel.n_times - 1
    t = np.arange(o - STEPS + 1, o + 1)
    R = panel.n_routes
    df = pd.DataFrame({
        "route_id": np.repeat(panel.routes, STEPS),
        "timestamp": np.tile(panel.times[t], R),
        **{c: np.round(panel.s[k][:, t]).astype(np.int64).ravel() for k, c in enumerate(STATUS_COLS)},
        "target_1h": np.round(panel.y[:, t]).astype(np.int64).ravel(),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    df[df["route_id"].isin(panel.routes[:3])].to_csv(OUT / "obrazec.csv", index=False)
    for name, seed, factor in SCENARIOS:
        out = df.copy()
        routes = np.sort(np.random.default_rng(seed).choice(panel.routes, ROUTES, replace=False))
        hit = out["route_id"].isin(routes) & (out["timestamp"] >= panel.times[t[FROM]])
        for c in STATUS_COLS[:3]:
            out.loc[hit, c] = np.round(out.loc[hit, c] * factor).astype(np.int64)
        out.to_csv(OUT / name, index=False)
        print(f"{OUT / name}: {len(out)} rows, status_1..3 x{factor} on {ROUTES} routes from {panel.times[t[FROM]]:%H:%M}")


if __name__ == "__main__":
    main()
