"""Forecast on uploaded data: the service model (models/prod/service) over the stored history (data/raw).

Loaded once, in the background at start-up; every upload copies the history, writes the file over it and forecasts
from the file's latest timestamp. The base forecast of a moment is computed once and kept.
"""

import logging
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path

ROOT = Path(os.environ.get("RWB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

import tools  # noqa: E402

log = logging.getLogger("uvicorn.error")
BUNDLE = ROOT / "models/prod/service"

# The network band of the console: actual / forecast of the all-route total per step, from board.json.
_net = tools.B["network"]
BAND = ([lo / p if p else 1.0 for lo, p in zip(_net["lo"], _net["p50"])],
        [hi / p if p else 1.0 for hi, p in zip(_net["hi"], _net["p50"])])


class Engine:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self.error: str | None = None
        self._lock = threading.Lock()
        self._base: OrderedDict = OrderedDict()

    def start(self) -> None:
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            from rwb.data import load_panel
            from rwb.features import cumsums
            from rwb.service import Service

            self.svc = Service.load(BUNDLE)
            self.panel = load_panel()
            self.cums = cumsums(self.panel)
            self.base_at(self.panel.n_times - 1)  # the console's own moment, the one the demo files use
            log.info("forecast engine ready: history until %s", self.panel.times[-1])
        except Exception as e:  # missing model or data: the endpoint says so instead of crashing the app
            self.error = f"{type(e).__name__}: {e}"
            log.exception("forecast engine failed to load")
        finally:
            self.ready.set()

    def base_at(self, origin: int):
        from rwb.upload import forecast

        with self._lock:
            if origin in self._base:
                self._base.move_to_end(origin)
                return self._base[origin]
        run = forecast(self.svc, self.panel, self.cums, origin)
        with self._lock:
            self._base[origin] = run
            while len(self._base) > 6:
                self._base.popitem(last=False)
        return run

    def run(self, raw: bytes) -> dict:
        from rwb.upload import run

        return run(self.svc, self.panel, self.cums, raw, self.base_at, BAND)


ENGINE = Engine()
