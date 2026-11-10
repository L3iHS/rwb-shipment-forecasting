"""Forecast from data a user uploads: the rows go over the stored history and the service forecasts again.

The file has the train columns. Rows inside the stored history replace it, rows right after it extend it; the
forecast starts at the latest timestamp of the file. The comparison is the forecast from the same moment on the
stored history, so a file that only repeats the history gives the console's own forecast back.
"""

import copy
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rwb.data import HORIZON, STATUS_COLS, STEP, Panel
from rwb.dispatch import DockState, call_plan
from rwb.features import DAY, MAX_LOOKBACK, Cumsums, cumsums
from rwb.service import Service

COLUMNS = ["route_id", "timestamp", *STATUS_COLS, "target_1h"]
MAX_ROWS = 200_000
MAX_VALUE = 1e10  # far above any real half-hour volume; larger values only break the features
MAX_AHEAD = 96  # steps a file may run past the stored history: two days
HIST_N = 16  # steps of history the answer shows, as on the console


class UploadError(ValueError):
    """What is wrong with the file, in words the user can act on."""


def read_table(raw: bytes) -> pd.DataFrame:
    """Parquet or CSV (comma, semicolon or tab) with the train columns, checked and typed."""
    if not raw.strip():
        raise UploadError("Файл пустой")
    too_many = f"Строк больше {MAX_ROWS:,}".replace(",", " ") + ": загрузите последние двое суток"
    unreadable = "Не получилось прочитать файл: нужен CSV или Parquet"
    if raw[:4] == b"PAR1":
        # A few MB of compressed Parquet can unpack into gigabytes: check the size before reading, and read only our columns.
        import pyarrow.parquet as pq

        try:
            f = pq.ParquetFile(io.BytesIO(raw))
        except Exception:
            raise UploadError(unreadable)
        names = [str(c).strip() for c in f.schema_arrow.names]
        missing = [c for c in COLUMNS if c not in names]
        if missing:
            raise UploadError(f"В файле нет колонок: {', '.join(missing)}")
        if f.metadata.num_rows > MAX_ROWS:
            raise UploadError(too_many)
        try:
            df = f.read(columns=[n for n in f.schema_arrow.names if str(n).strip() in COLUMNS]).to_pandas()
        except Exception:
            raise UploadError(unreadable)
        df.columns = [str(c).strip() for c in df.columns]
    else:
        try:
            head = raw[:2000].decode("utf-8-sig", errors="replace").splitlines()[0]
            sep = max([",", ";", "\t"], key=head.count)
            df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding="utf-8-sig", nrows=MAX_ROWS + 1,
                             usecols=lambda c: str(c).strip() in COLUMNS)
        except Exception:
            raise UploadError(unreadable)
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise UploadError(f"В файле нет колонок: {', '.join(missing)}")
    if not len(df):
        raise UploadError("В файле нет строк")
    if len(df) > MAX_ROWS:
        raise UploadError(too_many)
    df = df[COLUMNS].copy()
    df["timestamp"] = parse_times(df["timestamp"])
    for c in COLUMNS:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = df[c].isna().to_numpy()
        if bad.any():
            raise UploadError(f"Колонка {c}: пустое или непонятное значение в строке {int(np.argmax(bad)) + 2}")
    vals = df[COLUMNS[2:]].to_numpy(np.float64)
    if (vals < 0).any():
        raise UploadError("Объёмы не могут быть отрицательными")
    big = ~np.isfinite(vals) | (vals > MAX_VALUE)
    if big.any():
        row, col = np.argwhere(big)[0]
        raise UploadError(f"Колонка {COLUMNS[2 + col]}: слишком большое значение в строке {row + 2}")
    return df


def parse_times(ts: pd.Series) -> pd.Series:
    """ISO times as they are; dotted dates day first (10.09.2025 is 10 September); times with an offset in Moscow time."""
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.dt.tz_convert("Europe/Moscow").dt.tz_localize(None) if ts.dt.tz is not None else ts
    text = ts.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    zoned = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if zoned.any():
        z = pd.to_datetime(text[zoned], format="ISO8601", errors="coerce", utc=True)
        out[zoned] = z.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    plain = ~zoned
    if plain.any():
        iso = pd.to_datetime(text[plain], format="ISO8601", errors="coerce")
        out[plain] = iso
        rest = plain & out.isna()
        if rest.any():
            out[rest] = pd.to_datetime(text[rest], format="mixed", dayfirst=True, errors="coerce")
    return out


@dataclass
class Applied:
    panel: Panel
    origin: int
    first: int  # time index of the earliest row in the file
    changed: np.ndarray  # (R,) bool: the file differs from the stored history for this route
    changed_y: bool  # shipments themselves differ, not only the statuses
    extended: int  # steps the file adds after the stored history
    routes: int  # routes in the file


def apply(panel: Panel, df: pd.DataFrame) -> Applied:
    """The stored history with the file written over it (or appended after it)."""
    r = np.searchsorted(panel.routes, df["route_id"].to_numpy())
    r = np.minimum(r, panel.n_routes - 1)
    unknown = np.unique(df["route_id"].to_numpy()[panel.routes[r] != df["route_id"].to_numpy()])
    if len(unknown):
        listed = ", ".join(str(int(v)) for v in unknown[:5])
        raise UploadError(f"Маршрутов нет в истории: {listed}{' и другие' if len(unknown) > 5 else ''}")
    delta = df["timestamp"] - panel.times[0]
    if not (delta % STEP == pd.Timedelta(0)).all():
        raise UploadError("Время должно идти с шагом 30 минут: 10:00, 10:30, 11:00")
    t = (delta // STEP).to_numpy().astype(np.int64)
    if len(np.unique(r * (panel.n_times + MAX_AHEAD + 1) + t)) < len(t):
        raise UploadError("Одна и та же пара route_id и timestamp встречается дважды")
    last = panel.n_times - 1
    # The model reads 4 weeks before each origin, and the dock replay goes one more day back from the file's end.
    if t.min() < MAX_LOOKBACK or t.max() < MAX_LOOKBACK + DAY - 1:
        earliest = (panel.times[0] + STEP * MAX_LOOKBACK).strftime("%d.%m.%Y %H:%M")
        end = (panel.times[0] + STEP * (MAX_LOOKBACK + DAY - 1)).strftime("%d.%m.%Y %H:%M")
        raise UploadError(f"Модели нужно 4 недели истории до этих данных: начало не раньше {earliest}, конец не раньше {end}")
    if t.max() > last + MAX_AHEAD:
        raise UploadError(f"Данные заходят дальше двух суток после истории ({panel.times[last]:%d.%m.%Y %H:%M})")

    ext = int(max(0, t.max() - last))
    if ext:
        # New time points must be complete: the model reads every route at every step.
        counts = np.bincount(t[t > last] - last - 1, minlength=ext)
        if (counts != panel.n_routes).any():
            raise UploadError("Для времени после истории нужны строки всех маршрутов без пропусков")
    y = np.concatenate([panel.y, np.full((panel.n_routes, ext), np.nan, np.float32)], axis=1)
    s = np.concatenate([panel.s, np.full((6, panel.n_routes, ext), np.nan, np.float32)], axis=2)
    new_y = df["target_1h"].to_numpy(np.float32)
    new_s = df[STATUS_COLS].to_numpy(np.float32).T
    inside = t <= last
    diff_y = inside & (y[r, t] != new_y)
    diff_s = inside & (s[:, r, t] != new_s).any(axis=0)
    changed = np.zeros(panel.n_routes, dtype=bool)
    changed[r[diff_y | diff_s | ~inside]] = True
    y[r, t] = new_y
    s[:, r, t] = new_s

    times = panel.times
    if ext:
        times = times.append(pd.date_range(times[-1] + STEP, periods=ext, freq=STEP))
    out = copy.copy(panel)
    out.times, out.y, out.s = times, y, s
    return Applied(out, int(t.max()), int(t.min()), changed, bool(diff_y.any() or ext), ext, len(np.unique(r)))


@dataclass
class Run:
    fc: dict[str, np.ndarray]
    dock: DockState
    call: np.ndarray  # (R,) steps until the next truck call, -1 = not within the horizon


def forecast(svc: Service, panel: Panel, cums: Cumsums, origin: int, dock: DockState | None = None) -> Run:
    """Forecast, dock state and truck calls from ``origin``; the dock is replayed unless given."""
    fc = svc.forecast(panel, cums, origin)
    dock = dock if dock is not None else svc.dock_state(panel, cums, origin)
    call = call_plan(dock.backlog, fc["inflow_risk"], svc.capacity, svc.cfg.lead, dock.inbound)
    return Run(fc, dock, call)


def _rint(a: np.ndarray) -> list[int]:
    return np.round(np.nan_to_num(a)).astype(int).tolist()


def _hist(panel: Panel, a: np.ndarray, origin: int) -> np.ndarray:
    """The last HIST_N steps up to ``origin`` along the time axis of ``a``."""
    return np.nan_to_num(a[..., origin - HIST_N + 1 : origin + 1].astype(np.float64))


def compare(svc: Service, stored: Panel, applied: Applied, new: Run, base: Run | None,
            band: tuple[list[float], list[float]]) -> dict:
    """What the screen shows: the forecast on the file against the forecast on the stored history."""
    p, o = applied.panel, applied.origin
    in_hist = base is not None
    known = min(HORIZON, stored.n_times - 1 - o)  # steps after "now" already in the stored history
    has_after = known > 0
    lo_r, hi_r = np.asarray(band[0]), np.asarray(band[1])
    cap = svc.capacity

    y_new, s_new = _hist(p, p.y, o), _hist(p, p.s, o)
    y_old = _hist(stored, stored.y, o) if in_hist else None
    s_old = _hist(stored, stored.s, o) if in_hist else None
    after = np.nan_to_num(stored.y[:, o + 1 : o + 1 + known]) if has_after else None

    def wave(call: np.ndarray) -> list[int]:
        """Routes to call a truck for at each step 0..HORIZON: the bars under the console's timeline."""
        return np.bincount(call[(call >= 0) & (call <= HORIZON)], minlength=HORIZON + 1).tolist()

    def group(mask: np.ndarray) -> dict:
        p50 = new.fc["p50"][mask].sum(axis=0)
        g = {"routes": int(mask.sum()), "hist": _rint(y_new[mask].sum(axis=0)), "p50": _rint(p50),
             "lo": _rint(p50 * lo_r), "hi": _rint(p50 * hi_r),
             "s13": _rint(s_new[0:3, mask].sum(axis=(0, 1))), "s46": _rint(s_new[3:6, mask].sum(axis=(0, 1))),
             "vol2h": round(float(new.fc["inflow"][mask, 3].sum())),
             "call_now": int((new.call[mask] == 0).sum()), "call_hour": int(((new.call[mask] >= 1) & (new.call[mask] <= 2)).sum()),
             "calls": wave(new.call[mask])}
        if in_hist:
            b50 = base.fc["p50"][mask].sum(axis=0)
            g |= {"b_hist": _rint(y_old[mask].sum(axis=0)), "b50": _rint(b50), "b_lo": _rint(b50 * lo_r),
                  "b_hi": _rint(b50 * hi_r), "b_s13": _rint(s_old[0:3, mask].sum(axis=(0, 1))),
                  "b_s46": _rint(s_old[3:6, mask].sum(axis=(0, 1))), "b_vol2h": round(float(base.fc["inflow"][mask, 3].sum())),
                  "b_call_now": int((base.call[mask] == 0).sum()),
                  "b_call_hour": int(((base.call[mask] >= 1) & (base.call[mask] <= 2)).sum()),
                  "b_calls": wave(base.call[mask])}
        if has_after:
            g["after"] = _rint(after[mask].sum(axis=0))
        return g

    changed = applied.changed
    everyone = np.ones(p.n_routes, dtype=bool)
    # Every route in the format of board.json, so the console can show the file's shift as it shows its own.
    routes = []
    for i in range(p.n_routes):
        r = {"id": int(p.routes[i]), "cap": round(float(cap[i])), "bl": round(float(new.dock.backlog[i] / cap[i]), 3),
             "way": int(new.dock.on_the_way[i]), "dock": int(new.dock.at_dock[i]), "call": int(new.call[i]),
             "h": _rint(y_new[i]), "p10": _rint(new.fc["p10"][i]), "p50": _rint(new.fc["p50"][i]),
             "p90": _rint(new.fc["p90"][i]), "c50": _rint(new.fc["inflow"][i]), "c90": _rint(new.fc["inflow_p90"][i]),
             "cr": _rint(new.fc["inflow_risk"][i])}
        if has_after:
            r["after"] = _rint(after[i])
        routes.append(r)
    # Without the file, in the same order: the forecast and what the call rule needs.
    base_routes = [{"bl": round(float(base.dock.backlog[i] / cap[i]), 3), "way": int(base.dock.on_the_way[i]),
                    "dock": int(base.dock.at_dock[i]), "cr": _rint(base.fc["inflow_risk"][i]),
                    "c90": _rint(base.fc["inflow_p90"][i]), "p50": _rint(base.fc["p50"][i]),
                    "c50": _rint(base.fc["inflow"][i])} for i in range(p.n_routes)] if in_hist else None

    later = earlier = held = 0
    if in_hist:
        b, n = base.call, new.call
        later = int(((n > b) & (b >= 0) | (n < 0) & (b >= 0)).sum())
        earlier = int(((n >= 0) & ((n < b) | (b < 0))).sum())
        held = int(((b == 0) & (n != 0)).sum())  # trucks the old plan called now and the new one does not

    return {
        "now": p.times[o].isoformat(), "step_min": int(STEP / pd.Timedelta(minutes=1)), "hist_n": HIST_N,
        "lead": svc.cfg.lead,
        "file": {"rows": None, "routes": applied.routes, "from": p.times[applied.first].isoformat(),
                 "to": p.times[o].isoformat(), "extended": applied.extended},
        "baseline": in_hist, "dock_from": "board" if in_hist and not applied.changed_y else "replay",
        "changed": int(changed.sum()), "changed_y": applied.changed_y,
        "calls": {"later": later, "earlier": earlier, "held": held},
        "all": group(everyone), "group": group(changed) if in_hist and changed.any() else None,
        "routes": routes, "base_routes": base_routes,
    }


def run(svc: Service, stored: Panel, cums: Cumsums, raw: bytes, base_at, band) -> dict:
    """The whole request: read, apply, forecast both ways, compare. ``base_at(origin)`` gives the cached base Run."""
    df = read_table(raw)
    applied = apply(stored, df)
    o = applied.origin
    base = base_at(o) if o <= stored.n_times - 1 else None
    # The dock state is a fact the warehouse reports, not a forecast: while the file leaves the shipments as they
    # were, the cargo at the dock and the trucks on the way are the same as on the console.
    same_dock = base is not None and not applied.changed_y
    new_cums = cumsums(applied.panel)
    new = forecast(svc, applied.panel, new_cums, o, base.dock if same_dock else None)
    out = compare(svc, stored, applied, new, base, band)
    out["file"]["rows"] = len(df)
    return out
