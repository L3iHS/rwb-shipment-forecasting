"""Forecast on uploaded data: the service model (models/prod/service) over the stored history (data/raw).

Loaded once, in the background at start-up; every upload copies the history, writes the file over it and forecasts
from the file's latest timestamp. The base forecast of a moment is computed once and kept.
"""

import logging
import os
import sys
import threading
import tempfile
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
            scratch = os.environ.get("RWB_SCRATCH")
            if scratch:
                import shutil
                for old in Path(scratch).glob("forecast-*"):
                    if old.is_dir() and not old.is_symlink():
                        shutil.rmtree(old)
            from rwb.data import load_panel
            from rwb.features import cumsums
            from rwb.service import Service

            self.svc = Service.load(BUNDLE)
            if os.environ.get("RWB_PREPARED"):
                from runtime import load
                self.panel, self.cums = load()
            else:
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
        # upload.py's response contract predates Service v2's explicit point/quantile names.
        run.fc["inflow"] = run.fc["inflow_point"]
        with self._lock:
            self._base[origin] = run
            while len(self._base) > 6:
                self._base.popitem(last=False)
        return run

    def run(self, raw: bytes) -> dict:
        from rwb.upload import UploadError, apply, compare, forecast, read_table
        from rwb.monitor import DataUnavailableError
        from runtime import mapped_cumsums

        table = read_table(raw)
        applied = apply(self.panel, table)
        try:
            base = self.base_at(applied.origin) if applied.origin < self.panel.n_times else None
            with tempfile.TemporaryDirectory(prefix="forecast-", dir=os.environ.get("RWB_SCRATCH")) as scratch:
                sums = mapped_cumsums(applied.panel, Path(scratch))
                same_dock = base is not None and not applied.changed_y
                new = forecast(self.svc, applied.panel, sums, applied.origin, base.dock if same_dock else None)
                new.fc["inflow"] = new.fc["inflow_point"]
                out = compare(self.svc, self.panel, applied, new, base, BAND)
                out["file"]["rows"] = len(table)
                return out
        except DataUnavailableError as error:
            raise UploadError("Время файла раньше даты обучения модели или в истории недостаточно данных") from error


ENGINE = Engine()
