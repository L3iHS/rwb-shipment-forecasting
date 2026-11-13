"""The check of an uploaded file: every repair gives back the clean file, and what cannot be repaired is refused."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.checks import COLUMNS, UploadError, check  # noqa: E402
from rwb.data import STATUS_COLS, Panel  # noqa: E402
from rwb.features import DAY, MAX_LOOKBACK, cumsums  # noqa: E402

R = 40
T = MAX_LOOKBACK + DAY + 60


@pytest.fixture(scope="module")
def stored():
    """Half-hours x with zeros now and then (so the hourly sums decompose exactly), y[t] = x[t] + x[t-1]."""
    rng = np.random.default_rng(0)
    x = rng.integers(200, 2000, (R, T)).astype(np.float64)
    x[rng.random((R, T)) < 0.1] = 0
    y = x + np.c_[np.zeros(R), x[:, :-1]]
    s = np.concatenate([rng.integers(0, 50, (3, R, T)), rng.integers(1000, 5000, (3, R, T))]).astype(np.float32)
    p = Panel(pd.date_range("2025-08-01", periods=T, freq="30min"), np.arange(R), y.astype(np.float32), s, pd.DataFrame())
    return p, cumsums(p).x


def rows(p: Panel, steps) -> pd.DataFrame:
    """The stored history at these steps, as a file would carry it."""
    out = []
    for t in steps:
        out.append(pd.DataFrame({"route_id": p.routes, "timestamp": p.times[t],
                                 **{c: p.s[k, :, t] for k, c in enumerate(STATUS_COLS)}, "target_1h": p.y[:, t]}))
    return pd.concat(out, ignore_index=True)[COLUMNS]


def last4(p: Panel) -> pd.DataFrame:
    return rows(p, range(p.n_times - 4, p.n_times))


def run(df: pd.DataFrame | bytes, p: Panel, x) -> tuple[pd.DataFrame, dict]:
    raw = df if isinstance(df, bytes) else df.to_csv(index=False).encode()
    out, notes = check(raw, p, x)
    return out, {n.code: n for n in notes}


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a, b = (d.sort_values(["route_id", "timestamp"]).reset_index(drop=True) for d in (a, b))
    return len(a) == len(b) and (a["timestamp"] == b["timestamp"]).all() and np.allclose(a[COLUMNS[2:]], b[COLUMNS[2:]].astype(float))


def test_a_clean_file_passes_without_notes(stored):
    p, x = stored
    out, notes = run(last4(p), p, x)
    assert same(out, last4(p)) and not notes


def test_russian_excel_export_reads_as_the_clean_file(stored):
    p, x = stored
    d = last4(p)
    d.columns = ["Маршрут", "Время", *[f"Статус {i}" for i in range(1, 7)], "Отгружено за час"]
    d["Время"] = d["Время"].dt.strftime("%d.%m.%Y %-H:%M")
    for c in d.columns[2:]:
        d[c] = d[c].map(lambda v: f"{v:,.1f}".replace(",", " ").replace(".", ","))
    out, notes = run(d.to_csv(index=False, sep=";").encode("cp1251"), p, x)
    assert same(out, last4(p)) and {"encoding", "separator", "columns", "numbers"} <= set(notes)


def test_a_file_without_a_header_takes_the_columns_in_order(stored):
    p, x = stored
    out, notes = run(last4(p).to_csv(index=False, header=False).encode(), p, x)
    assert same(out, last4(p)) and "no_header" in notes


def test_seconds_off_the_half_hour_are_rounded_and_quarter_hours_refused(stored):
    p, x = stored
    d = last4(p)
    d["timestamp"] += pd.to_timedelta(np.random.default_rng(1).integers(-200, 200, len(d)), unit="s")
    out, notes = run(d, p, x)
    assert same(out, last4(p)) and "clock" in notes
    q = last4(p)
    q["timestamp"] += pd.to_timedelta(np.tile([0, 15], len(q) // 2), unit="min")
    with pytest.raises(UploadError, match="шаг 15 мин"):
        run(q, p, x)


def test_utc_without_a_mark_is_found_by_the_history_it_repeats(stored):
    p, x = stored
    d = last4(p)
    d["timestamp"] -= pd.Timedelta(hours=3)
    out, notes = run(d, p, x)
    assert same(out, last4(p)) and "time_shift" in notes
    z = last4(p)
    z["timestamp"] = (z["timestamp"] - pd.Timedelta(hours=3)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out, notes = run(z, p, x)
    assert same(out, last4(p)) and "time_shift" not in notes


def test_unix_seconds_are_utc(stored):
    p, x = stored
    d = last4(p)
    d["timestamp"] = (d["timestamp"] - pd.Timedelta(hours=3) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    out, notes = run(d, p, x)
    assert same(out, last4(p)) and "epoch" in notes


def test_repeated_rows_go_and_a_conflict_keeps_the_lower_row(stored):
    p, x = stored
    d = last4(p)
    out, notes = run(pd.concat([d, d.iloc[:10]]), p, x)
    assert same(out, d) and notes["duplicates"].rows == 10 and "conflicts" not in notes
    later = d.iloc[:1].assign(target_1h=d["target_1h"].iloc[0] + 1)
    out, notes = run(pd.concat([d, later]), p, x)
    assert notes["conflicts"].level == "warn"
    assert out.loc[(out.route_id == d.route_id.iloc[0]) & (out.timestamp == d.timestamp.iloc[0]), "target_1h"].item() == later.target_1h.item()


def test_unknown_routes_and_rows_too_old_to_matter_are_dropped(stored):
    p, x = stored
    d = last4(p)
    extra = pd.concat([d.iloc[:3].assign(route_id=[500, 501, 502]), d.iloc[:5].assign(timestamp=p.times[0])])
    out, notes = run(pd.concat([d, extra]), p, x)
    assert same(out, d) and notes["unknown_routes"].rows == 3 and notes["old_rows"].rows == 5
    with pytest.raises(UploadError, match="Ни одного маршрута"):
        run(d.assign(route_id=d.route_id + 1000), p, x)


def test_blanks_and_minus_one_inside_the_history_take_its_values(stored):
    p, x = stored
    d = last4(p).astype({c: float for c in COLUMNS[2:]})
    d.loc[3, "status_2"] = np.nan
    d.loc[7, "target_1h"] = -1
    out, notes = run(d, p, x)
    assert same(out, last4(p)) and notes["cells"].level == "filled" and notes["cells"].rows == 2


def test_stray_zeros_are_outliers_and_take_the_history(stored):
    p, x = stored
    d = last4(p)
    d.loc[5, "target_1h"] *= 1000
    out, notes = run(d, p, x)
    assert same(out, last4(p)) and notes["outliers"].rows == 1


def test_thousands_are_scaled_back(stored):
    p, x = stored
    d = last4(p)
    d["target_1h"] /= 1000
    out, notes = run(d, p, x)
    assert same(out, last4(p)) and "units" in notes


def test_half_hours_become_the_hourly_sum(stored):
    p, x = stored
    d = last4(p)
    t = np.repeat(np.arange(p.n_times - 4, p.n_times), R)
    d["target_1h"] = x[d["route_id"].to_numpy(), t]
    out, notes = run(d, p, x)
    assert same(out, last4(p)) and notes["half_hour"].level == "fixed"


def test_after_the_history_missing_routes_are_carried_and_a_thin_step_is_cut(stored):
    p, x = stored
    last = p.n_times - 1
    nxt = rows(p, [last - 1]).assign(timestamp=p.times[last] + pd.Timedelta(minutes=30))
    d = pd.concat([last4(p), nxt.iloc[5:]])  # 35 of 40 routes after the history
    out, notes = run(d, p, x)
    assert notes["absent"].rows == 5 and len(out) == len(last4(p)) + R
    missing = out[(out.timestamp == nxt.timestamp.iloc[0]) & (out.route_id < 5)]
    assert np.allclose(missing["target_1h"], p.y[:5, last])  # the route's previous half-hour
    out, notes = run(pd.concat([last4(p), nxt.iloc[:10]]), p, x)  # 10 of 40: too few, the step is cut
    assert "cut" in notes and out["timestamp"].max() == p.times[last]


def test_files_that_cannot_be_repaired_say_why(stored):
    p, x = stored
    with pytest.raises(UploadError, match="Excel"):
        check(b"PK\x03\x04zip", p, x)
    with pytest.raises(UploadError, match="только маршрут и время"):
        check(b"id,route_id,timestamp\n0,1,2025-11-01 11:00\n", p, x)
    with pytest.raises(UploadError, match="status_1: в 100% строк нет числа"):
        run(last4(p).assign(status_1="abc"), p, x)
    with pytest.raises(UploadError, match="Время не читается"):
        run(last4(p).assign(timestamp="вчера"), p, x)
