"""Check an uploaded file and repair it before the model reads it.

Real exports break in a few typical ways: other column names, Excel's semicolons and decimal commas, UTC instead of
Moscow time, seconds on the clock, rows repeated or missing, -1 or blanks for "no data", stray zeros, thousands
instead of items, half-hour volumes instead of the hourly sum. Each of these is repaired only when the file itself or
the stored history decides how; what cannot be decided is left as it is and said. Every step leaves a note, which
the screen and the assistant show, so a forecast on a repaired file says what it rests on.

    df, notes = check(raw, panel, x)     # x: the stored half-hour shipments (Cumsums.x)

Levels of a note: "fixed" -- repaired, nothing guessed; "filled" -- values replaced by an estimate (the history at
that moment, or the route's previous half-hour); "warn" -- left as it is, the forecast may suffer; "info" -- nothing
to repair, but worth knowing.
"""

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rwb.data import STATUS_COLS, STEP, Panel
from rwb.features import DAY, MAX_LOOKBACK, WEEK

COLUMNS = ["route_id", "timestamp", *STATUS_COLS, "target_1h"]
VALUES = COLUMNS[2:]
MAX_ROWS = 200_000
MAX_VALUE = 1e10  # far above any real half-hour volume; larger values only break the features
MAX_AHEAD = 96  # steps a file may run past the stored history: two days
SNAP = pd.Timedelta(minutes=5)  # a clock this close to a half-hour mark means that mark
OUTLIER = 10  # times the route's maximum over the last 4 weeks
MOSCOW = "Europe/Moscow"


class UploadError(ValueError):
    """What is wrong with the file, in words the user can act on."""


@dataclass
class Note:
    code: str
    level: str  # fixed | filled | warn | info
    text: str
    rows: int = 0

    def as_dict(self) -> dict:
        return {"code": self.code, "level": self.level, "text": self.text, "rows": self.rows}


def _n(v: float) -> str:
    """12 345 as the screen writes it."""
    return f"{v:,.0f}".replace(",", " ")


def _rows(n: int) -> str:
    tail = n % 100
    word = "строк" if 11 <= tail <= 14 else {1: "строка", 2: "строки", 3: "строки", 4: "строки"}.get(n % 10, "строк")
    return f"{_n(n)} {word}"


def _word(n: int, one: str, few: str, many: str) -> str:
    return many if 11 <= n % 100 <= 14 else {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def _values(n: int) -> str:
    tail = n % 100
    word = "значений" if 11 <= tail <= 14 else {1: "значение", 2: "значения", 3: "значения", 4: "значения"}.get(n % 10, "значений")
    return f"{_n(n)} {word}"


# --- Reading: bytes to a table with our column names --------------------------------------------------------------

_ALIASES = {
    "route_id": {"route_id", "routeid", "route", "route_no", "маршрут", "номер_маршрута", "id_маршрута", "маршрут_id"},
    "timestamp": {"timestamp", "time_stamp", "time", "datetime", "date_time", "ts", "время", "дата_время", "датавремя",
                  "дата_и_время", "дата"},
    "target_1h": {"target_1h", "target1h", "target", "y", "отгружено_за_час", "отгрузки_за_час", "отгрузка_за_час",
                  "отгружено", "отгрузки", "отгрузка"},
    **{f"status_{k}": {f"status_{k}", f"status{k}", f"статус_{k}", f"статус{k}", f"s_{k}", f"s{k}"} for k in range(1, 7)},
}
_CANON = {alias: name for name, aliases in _ALIASES.items() for alias in aliases}


def _norm(name) -> str:
    s = str(name).replace("﻿", "").strip().lower().replace("ё", "е")
    return re.sub(r"[\s\-.]+", "_", s).strip("_")


def _mapping(names: list) -> dict:
    """Original column name -> ours; an exact name wins over an alias of the same column."""
    out: dict = {}
    for exact in (True, False):
        for c in names:
            canon = _CANON.get(_norm(c))
            if canon and canon not in out.values() and c not in out and (not exact or _norm(c) == canon):
                out[c] = canon
    return out


def _missing(found: dict, names: list) -> UploadError:
    missing = [c for c in COLUMNS if c not in found.values()]
    if set(VALUES) <= set(missing) and len(missing) < len(COLUMNS):
        return UploadError("В файле только маршрут и время, без статусов и отгрузок: прогнозу нечего читать. "
                           f"Нужны колонки {', '.join(COLUMNS)}")
    shown = ", ".join(str(c) for c in names[:12])
    return UploadError(f"В файле нет колонок: {', '.join(missing)}. Есть: {shown}")


def _looks_like_data(cells: list) -> bool:
    """A header row made of values: a route number and a time where the names should be."""
    if len(cells) != len(COLUMNS):
        return False
    first = str(cells[0]).strip()
    when = pd.to_datetime(pd.Series([str(cells[1]).strip()]), errors="coerce", format="mixed", dayfirst=True)
    return bool(re.fullmatch(r"\d+(?:\.0+)?", first)) and when.notna().all()


def read(raw: bytes) -> tuple[pd.DataFrame, list[Note]]:
    """Parquet or CSV in any of the usual spellings, as text or typed columns named as ours, plus what was done."""
    notes: list[Note] = []
    if not raw.strip():
        raise UploadError("Файл пустой")
    too_many = f"Строк больше {_n(MAX_ROWS)}: загрузите последние двое суток"
    unreadable = "Не получилось прочитать файл: нужен CSV или Parquet"
    if raw[:4] == b"PK\x03\x04":
        raise UploadError("Это Excel (xlsx): сохраните его как CSV (Файл → Сохранить как → CSV) и загрузите снова")
    if raw[:4] == b"PAR1":
        # A few MB of compressed Parquet can unpack into gigabytes: check the size before reading, and read only our columns.
        import pyarrow.parquet as pq

        try:
            f = pq.ParquetFile(io.BytesIO(raw))
        except Exception:
            raise UploadError(unreadable)
        names = list(f.schema_arrow.names)
        found = _mapping(names)
        if len(set(found.values())) < len(COLUMNS):
            raise _missing(found, names)
        if f.metadata.num_rows > MAX_ROWS:
            raise UploadError(too_many)
        try:
            df = f.read(columns=list(found)).to_pandas()
        except Exception:
            raise UploadError(unreadable)
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1251", errors="replace")
            notes.append(Note("encoding", "fixed", "Кодировка Windows-1251 (так сохраняет Excel): прочитана"))
        lines = [ln for ln in text[:4000].splitlines() if ln.strip()]
        if not lines:
            raise UploadError("Файл пустой")
        sep = max([",", ";", "\t", "|"], key=lines[0].count)
        try:
            head = pd.read_csv(io.StringIO(text), sep=sep, nrows=0, dtype=str)
        except Exception:
            raise UploadError(unreadable)
        names = list(head.columns)
        found = _mapping(names)
        header = 0
        if not found and _looks_like_data(names):
            header, found = None, dict(enumerate(COLUMNS))
            notes.append(Note("no_header", "fixed", "Строки заголовка нет: колонки взяты в порядке образца, "
                                                   f"{', '.join(COLUMNS)}"))
        if len(set(found.values())) < len(COLUMNS):
            raise _missing(found, names)
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, header=header, dtype=str, keep_default_na=False,
                             nrows=MAX_ROWS + 1, usecols=list(found), skip_blank_lines=True)
        except Exception:
            raise UploadError(unreadable)
        if sep != ",":
            notes.append(Note("separator", "fixed", f"Разделитель «{'Tab' if sep == chr(9) else sep}» вместо запятой: прочитан"))
    renamed = {c: n for c, n in found.items() if isinstance(c, str) and c != n}
    df = df.rename(columns=found)[COLUMNS].copy()
    if renamed:
        shown = ", ".join(f"{str(c).strip()} → {n}" for c, n in list(renamed.items())[:4])
        notes.append(Note("columns", "fixed", f"Колонки названы иначе, сопоставлены: {shown}{' и другие' if len(renamed) > 4 else ''}"))
    if len(names) > len(found):
        extra = [str(c).strip() for c in names if c not in found]
        notes.append(Note("extra_columns", "info", f"Лишние колонки не используются: {', '.join(extra[:5])}"))
    if not len(df):
        raise UploadError("В файле нет строк")
    if len(df) > MAX_ROWS:
        raise UploadError(too_many)
    return df, notes


# --- Parsing: text to numbers and times, unreadable cells marked -------------------------------------------------

_EMPTY, _UNREADABLE, _NEGATIVE, _INFINITE, _OUTLIER, _ABSENT = 1, 2, 3, 4, 5, 6


def parse_ids(col: pd.Series) -> tuple[pd.Series, int]:
    """Route numbers; 471.0, R-471 and № 471 are read as 471. Returns the numbers (NaN where none) and how many were respelled."""
    if pd.api.types.is_numeric_dtype(col):
        v = col.astype(np.float64)
        return v.where(np.isfinite(v) & (v == np.round(v))), 0
    text = col.astype(str).str.strip()
    digits = text.str.extract(r"^(?:[A-Za-zА-Яа-я№#]+[\s\-_]*)?(\d+)(?:[.,]0+)?$")[0]
    return pd.to_numeric(digits, errors="coerce"), int((digits.notna() & (digits != text)).sum())


def parse_times(ts: pd.Series, notes: list[Note] | None = None) -> pd.Series:
    """Moscow time without a zone. ISO and dotted day-first dates as they are; times with an offset, Unix seconds or
    milliseconds (UTC) moved to Moscow; Excel serial days as local time."""
    notes = notes if notes is not None else []
    if pd.api.types.is_datetime64_any_dtype(ts):
        if ts.dt.tz is not None:
            notes.append(Note("timezone", "fixed", "Время с часовым поясом: переведено в московское"))
            return ts.dt.tz_convert(MOSCOW).dt.tz_localize(None)
        return ts.astype("datetime64[ns]")
    num = pd.to_numeric(ts, errors="coerce") if not pd.api.types.is_numeric_dtype(ts) else ts.astype(np.float64)
    if num.notna().mean() > 0.9:
        med = float(num.median())
        for lo, hi, unit, what in [(1e9, 1e11, "s", "секунд"), (1e12, 1e14, "ms", "миллисекунд")]:
            if lo <= med < hi:
                notes.append(Note("epoch", "fixed", f"Время числом {what} Unix (UTC): переведено в московское"))
                return pd.to_datetime(num, unit=unit, errors="coerce", utc=True).dt.tz_convert(MOSCOW).dt.tz_localize(None)
        if 2e4 <= med < 8e4:
            notes.append(Note("excel_days", "fixed", "Время числом дней Excel: переведено в дату и время"))
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(num, unit="D")).dt.round("s")
    text = ts.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    zoned = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if zoned.any():
        z = pd.to_datetime(text[zoned], format="ISO8601", errors="coerce", utc=True)
        out[zoned] = z.dt.tz_convert(MOSCOW).dt.tz_localize(None)
        offsets = sorted(set(text[zoned].str.extract(r"(Z|[+-]\d{2}:?\d{2})$")[0].replace("Z", "UTC")))
        if offsets != ["+03:00"] and offsets != ["+0300"]:
            notes.append(Note("timezone", "fixed", f"Время с часовым поясом {', '.join(offsets[:3])}: переведено в московское"))
    plain = ~zoned
    if plain.any():
        out[plain] = pd.to_datetime(text[plain], format="ISO8601", errors="coerce")
        rest = plain & out.isna()
        if rest.any():
            out[rest] = pd.to_datetime(text[rest], format="mixed", dayfirst=True, errors="coerce")
    return out


def parse_numbers(col: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Values, the reason a cell is unusable (0 if it is fine), and how many were spelled the Excel way (150 418,5)."""
    why = np.zeros(len(col), dtype=np.int8)
    if pd.api.types.is_numeric_dtype(col):
        v = col.to_numpy(np.float64)
        why[np.isnan(v)] = _EMPTY
        respelled = 0
    else:
        text = col.astype(str).str.strip()
        clean = text.str.replace(r"[\s  ']", "", regex=True)
        both = clean.str.contains(",", regex=False) & clean.str.contains(".", regex=False)
        comma_last = clean.str.rfind(",") > clean.str.rfind(".")
        clean = clean.where(~both, clean.where(~comma_last, clean.str.replace(".", "", regex=False)))
        clean = clean.where(~(both & ~comma_last), clean.str.replace(",", "", regex=False))
        clean = clean.str.replace(",", ".", regex=False)
        v = pd.to_numeric(clean, errors="coerce").to_numpy(np.float64)
        empty = text.isin(["", "nan", "NaN", "NA", "null", "None", "-", "—"]).to_numpy()
        why[np.isnan(v) & empty] = _EMPTY
        why[np.isnan(v) & ~empty] = _UNREADABLE
        respelled = int(((clean != text) & ~np.isnan(v)).sum())
    why[np.isinf(v)] = _INFINITE
    why[(why == 0) & (v < 0)] = _NEGATIVE
    v = np.where(why == 0, v, np.nan)
    return v, why, respelled


# --- Repair against the stored history ----------------------------------------------------------------------------


def _shift(r: np.ndarray, t: np.ndarray, y: np.ndarray, panel: Panel) -> tuple[int, float, float]:
    """Whole hours that make the file's shipments meet the stored ones: (steps, share at 0, share at that shift)."""
    last = panel.n_times - 1

    def share(k: int) -> tuple[float, int]:
        tk = t + k
        ok = (tk >= 0) & (tk <= last) & np.isfinite(y)
        if ok.sum() < 30:
            return 0.0, int(ok.sum())
        stored = panel.y[r[ok], tk[ok]]
        return float((np.abs(y[ok] - stored) <= 0.5 + 1e-6 * np.abs(stored)).mean()), int(ok.sum())

    at0, n0 = share(0)
    if at0 >= 0.5:
        return 0, at0, at0
    best, best_share = 0, at0
    for k in [s * h for h in range(1, 13) for s in (2, -2)]:
        sh, n = share(k)
        if sh > best_share and n >= max(30, 0.5 * len(t)):
            best, best_share = k, sh
    return (best, at0, best_share) if best_share >= 0.9 else (0, at0, at0)


def _reference(panel: Panel, col: int, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """What the history says the value should be: the stored value inside it, the mean of the same half-hour over the
    last 4 weeks after it. col 0..5 is a status, 6 is target_1h."""
    a = panel.s[col] if col < 6 else panel.y
    last = panel.n_times - 1
    inside = t <= last
    out = np.full(len(t), np.nan)
    out[inside] = a[r[inside], t[inside]]
    if (~inside).any():
        acc, cnt = np.zeros((~inside).sum()), np.zeros((~inside).sum())
        for k in range(1, 5):
            tk = t[~inside] - k * WEEK
            ok = (tk >= 0) & (tk <= last)
            acc[ok] += a[r[~inside][ok], tk[ok]]
            cnt[ok] += 1
        out[~inside] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return out


def repair(df: pd.DataFrame, panel: Panel, x: np.ndarray, notes: list[Note]) -> pd.DataFrame:
    """The parsed file made fit for the model, judged against the stored history. ``x``: stored half-hour shipments."""
    last = panel.n_times - 1

    # Rows the model cannot place.
    lost = df["route_id"].isna() | df["timestamp"].isna()
    if lost.all():
        raise UploadError("Ни в одной строке не читается номер маршрута: нужны целые числа, как 471")
    if lost.any():
        notes.append(Note("unplaced", "fixed", f"{_rows(int(lost.sum()))} без номера маршрута или с нечитаемым временем "
                                               "не используются", int(lost.sum())))
        df = df[~lost]

    ids = df["route_id"].to_numpy(np.int64)
    r = np.minimum(np.searchsorted(panel.routes, ids), panel.n_routes - 1)
    known = panel.routes[r] == ids
    if not known.any():
        listed = ", ".join(str(v) for v in np.unique(ids)[:5])
        raise UploadError(f"Ни одного маршрута из истории: {listed}. Номера маршрутов от {panel.routes[0]} до {panel.routes[-1]}")
    if not known.all():
        unknown = np.unique(ids[~known])
        listed = ", ".join(str(v) for v in unknown[:5]) + (" и другие" if len(unknown) > 5 else "")
        notes.append(Note("unknown_routes", "fixed", f"Маршрутов {listed} нет в истории: их {_rows(int((~known).sum()))} "
                                                     "не используются", int((~known).sum())))
        df, r = df[known], r[known]

    # The clock: seconds and a few minutes off the half-hour mark are the same half-hour.
    delta = df["timestamp"] - panel.times[0]
    off = (delta % STEP).to_numpy()
    off = np.where(off > STEP / 2, off - STEP.to_timedelta64(), off)
    near = np.abs(off) <= SNAP.to_timedelta64()
    snapped = near & (off != np.timedelta64(0, "ns"))
    if (~near).mean() > 0.1:
        stamps = np.sort(df["timestamp"].unique())
        gaps = pd.Series(np.diff(stamps)).value_counts()
        step = f": похоже, шаг {int(gaps.index.min() / pd.Timedelta(minutes=1))} мин" if len(gaps) else ""
        raise UploadError(f"Время не ложится на сетку 30 минут{step}. Нужны отметки 10:00, 10:30, 11:00")
    if snapped.any():
        worst = pd.Timedelta(np.abs(off[snapped]).max())
        n = int(snapped.sum())
        notes.append(Note("clock", "fixed", f"У {_n(n)} {_word(n, 'строки', 'строк', 'строк')} время не ровно на получасе (до "
                                            f"{int(worst.total_seconds() // 60)} мин {int(worst.total_seconds() % 60)} с): "
                                            "округлено до ближайшего получаса", n))
    if (~near).any():
        notes.append(Note("off_grid", "fixed", f"{_rows(int((~near).sum()))} со временем между получасами (10:15, 10:45) "
                                               "не используются", int((~near).sum())))
        df, r, off, delta = df[near], r[near], off[near], delta[near]
    t = ((delta.to_numpy() - off) // STEP.to_timedelta64()).astype(np.int64)
    vals = np.array(df[VALUES], dtype=np.float64)
    why = np.array(df[[f"why:{c}" for c in VALUES]], dtype=np.int8)

    # A file a few hours off the history it repeats was written in another time zone.
    k, at0, at_k = _shift(r, t, vals[:, 6], panel)
    if k:
        t = t + k
        notes.append(Note("time_shift", "fixed", f"Время сдвинуто на {abs(k) // 2} ч: так отгрузки совпали с историей в "
                                                 f"{at_k:.0%} строк, без сдвига в {at0:.0%}. Похоже, время записано в UTC или "
                                                 f"другом часовом поясе; {'прибавлено' if k > 0 else 'вычтено'} {abs(k) // 2} ч", len(t)))

    # The model's window: two days past the history at most, 4 weeks and a day before the file's end at least.
    ahead = t > last + MAX_AHEAD
    if ahead.any():
        if ahead.mean() > 0.1:
            raise UploadError(f"Данные заходят дальше двух суток после истории ({panel.times[last]:%d.%m.%Y %H:%M})")
        first = (panel.times[0] + STEP * int(t[ahead].min())).strftime("%d.%m.%Y")
        notes.append(Note("far_future", "fixed", f"{_rows(int(ahead.sum()))} с датой дальше двух суток после истории "
                                                 f"(например, {first}) не используются", int(ahead.sum())))
    keep = ~ahead
    origin = int(t[keep].max())
    early = keep & (t < origin - MAX_LOOKBACK - DAY)
    if early.any():
        since = (panel.times[0] + STEP * (origin - MAX_LOOKBACK - DAY)).strftime("%d.%m.%Y")
        notes.append(Note("old_rows", "fixed", f"{_rows(int(early.sum()))} раньше {since}: модель смотрит назад на 4 недели "
                                               "и день, на прогноз они не влияют и не используются", int(early.sum())))
        keep &= ~early
    r, t, vals, why = r[keep], t[keep], vals[keep], why[keep]

    # The same route and half-hour twice: one row stays, each cell from the lowest row in the file that has a number.
    key = r.astype(np.int64) * (last + MAX_AHEAD + 2) + t
    if len(np.unique(key)) < len(key):
        # Sorted, so a pair is contiguous and the first and last rows of the groups come in the order of the groups.
        order = np.argsort(key, kind="stable")
        r, t, vals, why, key = r[order], t[order], vals[order], why[order], key[order]
        frame = pd.DataFrame(vals).assign(key=key)
        groups = frame.groupby("key", sort=False)
        merged = groups.last()  # the last non-empty value of each column
        size = groups.size().to_numpy()
        differ = (groups.nunique() > 1).any(axis=1).to_numpy()
        firsts = groups.head(1).index.to_numpy()  # positions: the frame has a plain 0..n-1 index
        lasts = groups.tail(1).index.to_numpy()
        extra = int(size.sum() - len(size))
        clash = int((size[differ] - 1).sum())
        if extra - clash:
            notes.append(Note("duplicates", "fixed", f"{_rows(extra - clash)} повторяются: лишние копии убраны", extra - clash))
        if clash:
            notes.append(Note("conflicts", "warn", f"{_rows(clash)}: тот же маршрут и время, но другие числа; взята строка, "
                                                   "которая в файле ниже", clash))
        vals = np.array(merged, dtype=np.float64)  # a copy: pandas hands out read-only views
        why = np.where(np.isnan(vals), why[lasts], 0).astype(np.int8)
        r, t = r[firsts], t[firsts]

    # Units: the whole column a thousand times off the history of the same routes at the same half-hours.
    floor = 10.0
    for j, c in enumerate(VALUES):
        ref = _reference(panel, j, r, t)
        ok = np.isfinite(vals[:, j]) & (vals[:, j] > 0) & (ref >= floor)
        if ok.sum() < 30:
            continue
        lg = np.log10(vals[ok, j] / ref[ok])
        med = float(np.median(lg))
        for p in (-6, -3, 3, 6):
            if abs(med - p) < 0.2 and (np.abs(lg - p) < 0.3).mean() > 0.6:
                vals[:, j] *= 10.0 ** -p
                word = {-3: "тысячах", -6: "миллионах", 3: "тысячных", 6: "миллионных"}[p]
                notes.append(Note("units", "fixed", f"{c} в {word}: в {_n(10 ** abs(p))} раз {'меньше' if p < 0 else 'больше'} "
                                                    f"истории тех же маршрутов в те же получасы; {'умножено' if p < 0 else 'разделено'} "
                                                    f"на {_n(10 ** abs(p))}", len(t)))
                break

    # target_1h is the sum of two half-hours; a file with the half-hours themselves is half as large.
    ref = _reference(panel, 6, r, t)
    ok = np.isfinite(vals[:, 6]) & (ref >= floor)
    if ok.sum() >= 30:
        ratio = float(np.median(vals[ok, 6] / ref[ok]))
        if 0.3 <= ratio <= 0.7:
            prev = np.full(len(t), np.nan)
            pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(r, t))}
            for i, (a, b) in enumerate(zip(r, t)):
                j = pos.get((int(a), int(b) - 1))
                prev[i] = vals[j, 6] if j is not None else (x[a, b - 1] if 1 <= b <= last + 1 else np.nan)
            pair = vals[:, 6] + prev
            inside = (t <= last) & np.isfinite(pair)
            stored = panel.y[r[inside], t[inside]]
            meets = float((np.abs(pair[inside] - stored) <= np.maximum(2, 0.02 * stored)).mean()) if inside.sum() >= 30 else 0.0
            if meets >= 0.8:
                vals[:, 6] = pair
                notes.append(Note("half_hour", "fixed", "target_1h за полчаса, а модели нужна сумма за час: сложены соседние "
                                                        f"получасы, сумма совпала с историей в {meets:.0%} строк", len(t)))
            else:
                notes.append(Note("half_hour", "warn", f"target_1h в среднем в {1 / ratio:.1f} раза меньше истории тех же "
                                                       "маршрутов: если это объём за полчаса, нужна сумма за час", len(t)))

    # Outliers: far above anything the route did in the last 4 weeks, as a stray zero or a broken counter gives.
    a, b = max(0, min(origin, last) - 4 * WEEK + 1), min(origin, last) + 1
    hist = [panel.s[j][:, a:b] for j in range(6)] + [panel.y[:, a:b]]
    examples = []
    for j, c in enumerate(VALUES):
        top = np.nanmax(hist[j], axis=1)
        limit = OUTLIER * np.maximum(top, 0.05 * np.nanmedian(top)) + OUTLIER
        big = np.isfinite(vals[:, j]) & ((vals[:, j] > limit[r]) | (vals[:, j] > MAX_VALUE))
        if big.any():
            i = int(np.argmax(np.where(big, vals[:, j] / np.maximum(limit[r], 1), 0)))
            examples.append((int(big.sum()), f"№ {panel.routes[r[i]]} в {panel.times[0] + STEP * int(t[i]):%H:%M}: "
                                             f"{c} {_n(vals[i, j])} при максимуме за 4 недели {_n(top[r[i]])}"))
            why[big, j] = _OUTLIER
            vals[big, j] = np.nan

    # After the history every route is needed at every step: a step with most routes is completed, one with few is cut.
    R = panel.n_routes
    if origin > last:
        counts = np.bincount(t[t > last] - last - 1, minlength=origin - last)
        cut = next((i for i, n in enumerate(counts) if n < 0.5 * R), None)
        if cut is not None:
            at = panel.times[last] + STEP * (cut + 1)
            new_origin = last + cut
            notes.append(Note("cut", "fixed", f"В {at:%H:%M} {'нет данных' if counts[cut] == 0 else f'есть только {counts[cut]} {_word(int(counts[cut]), "маршрут", "маршрута", "маршрутов")} из {R}'}: "
                                              f"этот получас{' и дальше' if cut + 1 < len(counts) else ''} не используется, "
                                              f"прогноз от {panel.times[0] + STEP * new_origin:%H:%M}", int((t > new_origin).sum())))
            keep = t <= new_origin
            r, t, vals, why = r[keep], t[keep], vals[keep], why[keep]
            origin = new_origin
        holes = []
        for step in range(last + 1, origin + 1):
            have = np.zeros(R, dtype=bool)
            have[r[t == step]] = True
            if not have.all():
                holes.append((step, np.flatnonzero(~have)))
        if holes:
            add_r = np.concatenate([h for _, h in holes])
            add_t = np.concatenate([np.full(len(h), s) for s, h in holes])
            r, t = np.r_[r, add_r], np.r_[t, add_t]
            vals = np.r_[vals, np.full((len(add_r), len(VALUES)), np.nan)]
            why = np.r_[why, np.full((len(add_r), len(VALUES)), _ABSENT, dtype=np.int8)]
            shown = ", ".join(f"{panel.times[0] + STEP * s:%H:%M} нет {len(h)}" for s, h in holes[:3])
            notes.append(Note("absent", "filled", f"После истории не хватает маршрутов ({shown} из {R}): для них взят "
                                                  "предыдущий получас того же маршрута", len(add_r)))

    # Unusable cells: inside the history its own value stays, after it the route's previous half-hour.
    gone = np.isnan(vals)
    if gone.any():
        inside = t <= last
        for j in range(len(VALUES)):
            a_ = panel.s[j] if j < 6 else panel.y
            m = gone[:, j] & inside
            vals[m, j] = a_[r[m], t[m]]
        if (gone & ~inside[:, None]).any():
            ext = origin - last
            grid = np.full((len(VALUES), R, ext + 1), np.nan)
            grid[:6, :, 0], grid[6, :, 0] = panel.s[:, :, last], panel.y[:, last]
            out = ~inside
            grid[:, r[out], t[out] - last] = vals[out].T
            grid = pd.DataFrame(grid.reshape(-1, ext + 1).T).ffill().to_numpy().T.reshape(len(VALUES), R, ext + 1)
            m = out[:, None] & gone
            rows_, cols_ = np.nonzero(m)
            vals[rows_, cols_] = grid[cols_, r[rows_], t[rows_] - last]
        reasons = [(_EMPTY, "пустые"), (_UNREADABLE, "нечитаемые"), (_NEGATIVE, "отрицательные (−1 и т. п.)"),
                   (_INFINITE, "бесконечные")]
        parts = [f"{word} {_n(int((why == code).sum()))}" for code, word in reasons if (why == code).any()]
        where = []
        if (gone & inside[:, None]).any():
            where.append("значение из истории")
        if (gone & ~inside[:, None] & (why != _ABSENT)).any():
            where.append("предыдущий получас маршрута (после конца истории)")
        if parts:
            n = int(np.isin(why, [c for c, _ in reasons]).sum())
            notes.append(Note("cells", "filled", f"Ячейки без числа: {', '.join(parts)}. Вместо них {' или '.join(where)}", n))
        if examples:
            n = sum(k for k, _ in examples)
            notes.append(Note("outliers", "filled", f"{_values(n)} больше обычного для маршрута в {OUTLIER} с лишним раз, "
                                                    f"похоже на ошибку выгрузки (например, {examples[0][1]}). Вместо них "
                                                    f"{' или '.join(where)}", n))

    # What the file covers: the rest is the stored history.
    routes_in = len(np.unique(r))
    steps_in = np.unique(t[t <= last])
    if routes_in < R:
        notes.append(Note("partial_routes", "info", f"В файле {routes_in} {_word(routes_in, 'маршрут', 'маршрута', 'маршрутов')} из {R}: остальные берутся из истории "
                                                    "без изменений"))
    holes_in = routes_in * len(steps_in) - int((t <= last).sum())
    if holes_in > 0:
        notes.append(Note("missing_rows", "info", f"Внутри истории в файле нет {_rows(holes_in)} маршрут × время: "
                                                  "там остаются данные истории", holes_in))
    if len(t) == 0:
        raise UploadError("После проверки в файле не осталось строк")

    out = pd.DataFrame({"route_id": panel.routes[r], "timestamp": panel.times[0] + STEP * t})
    for j, c in enumerate(VALUES):
        out[c] = vals[:, j]
    return out[COLUMNS].sort_values(["timestamp", "route_id"], kind="stable").reset_index(drop=True)


def parse(df: pd.DataFrame, notes: list[Note]) -> pd.DataFrame:
    """Typed columns; an unusable value becomes NaN and its reason goes along in a ``why:<column>`` column."""
    out = pd.DataFrame(index=df.index)
    out["route_id"], respelled = parse_ids(df["route_id"])
    if respelled:
        notes.append(Note("route_ids", "fixed", f"Номера маршрутов записаны как «471.0» или «R-471»: прочитаны как номера",
                          respelled))
    out["timestamp"] = parse_times(df["timestamp"], notes)
    bad = out["timestamp"].isna()
    if bad.mean() > 0.2:
        sample = str(df["timestamp"][bad].iloc[0]).strip()[:40]
        raise UploadError(f"Время не читается в {bad.mean():.0%} строк (например, «{sample}»): нужен формат "
                          "2025-11-01 10:30 или 01.11.2025 10:30")
    total = 0
    for c in VALUES:
        v, why, respelled = parse_numbers(df[c])
        if (why > 0).mean() > 0.5:  # a whole column of blanks or words is a wrong column, not gaps to fill
            sample = next((str(x) for x, w in zip(df[c], why) if w == _UNREADABLE), "")
            raise UploadError(f"Колонка {c}: в {(why > 0).mean():.0%} строк нет числа"
                              f"{f' (например, «{sample[:30]}»)' if sample else ''}")
        out[c], out[f"why:{c}"] = v, why
        total += respelled
    if total:
        notes.append(Note("numbers", "fixed", f"{_values(total)} записаны с пробелами или запятой (150 418,0): прочитаны",
                          total))
    return out


def check(raw: bytes, panel: Panel, x: np.ndarray) -> tuple[pd.DataFrame, list[Note]]:
    """Read, parse and repair a file against the stored history: the table the model reads and what was done to it."""
    df, notes = read(raw)
    return repair(parse(df, notes), panel, x, notes), notes
