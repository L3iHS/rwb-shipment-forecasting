import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.data import STATUS_COLS, Panel
from rwb.features import DAY, MAX_LOOKBACK
from rwb.upload import COLUMNS, UploadError, apply, read_table


def _panel(R=3, T=MAX_LOOKBACK + DAY + 20):
    times = pd.date_range("2025-01-01", periods=T, freq="30min")
    y = np.arange(R * T, dtype=np.float32).reshape(R, T)
    s = np.ones((6, R, T), dtype=np.float32)
    return Panel(times, np.array([10, 20, 30]), y, s, pd.DataFrame())


def _rows(p: Panel, steps, routes=(10, 20, 30)):
    out = []
    for t in steps:
        for rid in routes:
            r = int(np.searchsorted(p.routes, rid))
            out.append({"route_id": rid, "timestamp": p.times[0] + pd.Timedelta(minutes=30 * t),
                        **{c: float(p.s[k, r, t]) if t < p.n_times else 1.0 for k, c in enumerate(STATUS_COLS)},
                        "target_1h": float(p.y[r, t]) if t < p.n_times else 5.0})
    return pd.DataFrame(out)[COLUMNS]


def test_read_table_reads_semicolon_csv_and_names_missing_columns():
    df = read_table(b"route_id;timestamp;status_1;status_2;status_3;status_4;status_5;status_6;target_1h\n"
                    b"10;2025-01-01 00:00;1;1;1;1;1;1;7\n")
    assert df["target_1h"].iloc[0] == 7
    with pytest.raises(UploadError, match="status_1"):
        read_table(b"route_id,timestamp\n1,2025-01-01\n")


def test_file_that_repeats_history_changes_nothing():
    p = _panel()
    a = apply(p, _rows(p, range(p.n_times - 4, p.n_times)))
    assert a.origin == p.n_times - 1 and not a.changed.any() and not a.changed_y and a.extended == 0


def test_status_change_marks_only_that_route_and_keeps_shipments():
    p = _panel()
    df = _rows(p, [p.n_times - 1])
    df.loc[df.route_id == 20, "status_1"] = 0
    a = apply(p, df)
    assert a.changed.tolist() == [False, True, False] and not a.changed_y
    assert a.panel.s[0, 1, -1] == 0 and p.s[0, 1, -1] == 1  # the stored history is not touched


def test_rows_after_history_extend_it_only_when_complete():
    p = _panel()
    a = apply(p, _rows(p, [p.n_times, p.n_times + 1]))
    assert a.extended == 2 and a.panel.n_times == p.n_times + 2 and a.origin == p.n_times + 1
    with pytest.raises(UploadError, match="всех маршрутов"):
        apply(p, _rows(p, [p.n_times], routes=(10, 20)))


def test_rejects_unknown_routes_off_grid_times_and_too_early_data():
    p = _panel()
    with pytest.raises(UploadError, match="99"):
        apply(p, _rows(p, [p.n_times - 1], routes=(10,)).assign(route_id=99))
    off = _rows(p, [p.n_times - 1])
    off["timestamp"] += pd.Timedelta(minutes=10)
    with pytest.raises(UploadError, match="30 минут"):
        apply(p, off)
    with pytest.raises(UploadError, match="4 недели"):
        apply(p, _rows(p, [5]))


def test_dotted_dates_are_day_first_and_offsets_become_moscow_time():
    from rwb.upload import parse_times

    got = parse_times(pd.Series(["10.09.2025 09:00", "2025-11-01T07:30:00Z", "2025-11-01 10:30"])).tolist()
    assert got == [pd.Timestamp("2025-09-10 09:00"), pd.Timestamp("2025-11-01 10:30"), pd.Timestamp("2025-11-01 10:30")]


def test_rejects_infinite_and_huge_values_and_rows_that_land_on_one_step():
    head = ",".join(COLUMNS) + "\n"
    for v in ["inf", "1e30"]:
        with pytest.raises(UploadError, match="слишком большое"):
            read_table((head + f"10,2025-01-01 00:00,1,1,1,1,1,1,{v}\n").encode())
    p = _panel()
    df = _rows(p, [p.n_times - 1], routes=(10,))
    with pytest.raises(UploadError, match="30 минут"):
        apply(p, df.assign(timestamp=df.timestamp + pd.Timedelta(minutes=1)))


def test_file_ending_inside_the_first_replay_day_is_rejected_before_the_model():
    p = _panel()
    with pytest.raises(UploadError, match="конец не раньше"):
        apply(p, _rows(p, [MAX_LOOKBACK + 3]))
