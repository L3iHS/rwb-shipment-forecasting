"""How the "Свой файл" tab copes with dirty files: the same data written the ways real exports get it wrong.

    python scripts/dirty_upload.py            # every case through the /api/forecast engine, a table in the terminal
    python scripts/dirty_upload.py --write    # also the files: data/processed/dirty/*.csv and the demo of all at once,
                                              # frontend/public/demo/gryaznyi_0900-1030.csv

Every case is the jam demo (zator_0900-1030.csv: the last 2 hours of history, sorting down to 5 % on 200 routes from
09:30) spoiled one way. A good answer to a spoiled file is the clean file's forecast: the check finds what is wrong,
repairs what can be repaired without guessing, and says so. The table compares each case with the clean one by the
network's 4-hour volume and by the routes' 4-hour volumes (sum |case - clean| / sum clean).

Two more cases run past the stored history (11:00 and 11:30 for every route, made from the same half-hours a week
earlier): a full one, and the same with 300 routes missing at 11:30. The last case, "mess", is most of the others in
one file: the Russian Excel export of the jam, in UTC, with the clock off by seconds, repeated rows, -1 for blanks,
stray zeros, routes that do not exist and a column too many. It is the "Грязная выгрузка" demo of the tab.
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.data import ROOT, STATUS_COLS, STEP, load_panel  # noqa: E402
from rwb.features import WEEK  # noqa: E402
from rwb.upload import COLUMNS, UploadError  # noqa: E402

DEMO = ROOT / "frontend/public/demo"
OUT = ROOT / "data/processed/dirty"
MESS = DEMO / "gryaznyi_0900-1030.csv"
RU = ["Маршрут", "Время", "Статус 1", "Статус 2", "Статус 3", "Статус 4", "Статус 5", "Статус 6", "Отгружено за час"]


def csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def cases(panel, x: np.ndarray) -> list[tuple[str, str, bytes]]:
    """(key, what is wrong, file bytes); the first is the clean file every other case is compared with."""
    clean = pd.read_csv(DEMO / "zator_0900-1030.csv", parse_dates=["timestamp"])
    rng = np.random.default_rng(0)
    out: list[tuple[str, str, bytes]] = [("clean", "чистый файл: затор на 200 маршрутах", csv(clean))]

    def add(key: str, what: str, df: pd.DataFrame | None = None, raw: bytes | None = None, **kw) -> None:
        out.append((key, what, raw if raw is not None else df.to_csv(index=False, **kw).encode("utf-8")))

    def cells(frac: float, cols=COLUMNS[2:]) -> list[tuple[int, str]]:
        n = int(len(clean) * frac)
        return [(int(i), str(c)) for i, c in zip(rng.choice(len(clean), n, replace=False), rng.choice(cols, n))]

    # Format: what an export from Excel or a database does to the same numbers.
    d = clean.rename(columns=dict(zip(COLUMNS, ["Route ID", " TimeStamp ", *[f"Status {i}" for i in range(1, 7)], "TARGET_1H"])))
    add("headers", "колонки в другом регистре и с пробелами", d)
    d = clean.copy()
    d.columns = RU
    d["Время"] = d["Время"].dt.strftime("%d.%m.%Y %-H:%M")
    for c in RU[2:]:
        d[c] = d[c].map(lambda v: f"{v:,.1f}".replace(",", " ").replace(".", ","))
    add("excel_ru", "Excel по-русски: «;», cp1251, русские колонки, «150 418,0», 01.11.2025 9:00",
        raw=d.to_csv(index=False, sep=";").encode("cp1251"))
    d = clean.copy().sample(frac=1, random_state=1)
    d.insert(0, "id", range(len(d)))
    d["склад"] = "Коледино"
    add("extra_cols", "лишние колонки, строки перемешаны", d)
    add("no_header", "без строки заголовка", raw=clean.to_csv(index=False, header=False).encode())
    add("bom_crlf", "BOM, переводы строк Windows, пустые строки в конце",
        raw=b"\xef\xbb\xbf" + clean.to_csv(index=False, lineterminator="\r\n").encode() + b"\r\n\r\n")
    d = clean.copy()
    d["route_id"] = d["route_id"].astype(float)
    add("float_ids", "route_id дробными числами: 471.0", d)

    # Time.
    d = clean.copy()
    d["timestamp"] = (d["timestamp"] - pd.Timedelta(hours=3)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    add("utc_z", "время в UTC с пометкой Z", d)
    d = clean.copy()
    d["timestamp"] = d["timestamp"] - pd.Timedelta(hours=3)
    add("utc_naive", "время в UTC без пометки: всё на 3 часа раньше", d)
    d = clean.copy()
    d["timestamp"] = d["timestamp"] + pd.to_timedelta(rng.integers(-180, 181, len(d)), unit="s")
    add("jitter", "время с секундами и сбитыми минутами: 09:28:41, 10:02:15", d)
    d = clean.copy()
    d["timestamp"] = (d["timestamp"] - pd.Timedelta(hours=3) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    add("epoch", "время числом секунд Unix (UTC)", d)
    d = clean.copy()
    d["timestamp"] = d["timestamp"].dt.strftime("%Y/%m/%d %H:%M")
    add("slashes", "дата через косую черту: 2025/11/01 10:30", d)

    # Rows.
    d = pd.concat([clean, clean.sample(frac=0.05, random_state=2)]).sort_values(["route_id", "timestamp"])
    add("dupes", "5 % строк повторены дважды", d)
    d = clean.drop(clean.sample(frac=0.10, random_state=3).index)
    add("missing_rows", "10 % строк пропали", d)
    d = pd.concat([clean, clean[clean.route_id < 5].assign(route_id=lambda x: x.route_id + 1000)])
    add("unknown_routes", "5 маршрутов, которых нет в истории (1000–1004)", d)
    d = pd.concat([clean, clean[clean.route_id < 50].assign(timestamp=lambda x: x.timestamp - pd.Timedelta(days=90))])
    add("old_rows", "строки трёхмесячной давности, раньше начала истории", d)

    # Values.
    d = clean.copy().astype({c: float for c in COLUMNS[2:]})
    for i, c in cells(0.02):
        d.loc[i, c] = np.nan
    add("empty_cells", "2 % ячеек пустые", d)
    d = clean.copy()
    for i, c in cells(0.01):
        d.loc[i, c] = -1
    add("minus_one", "1 % ячеек равны −1 (так выгрузка пишет «нет данных»)", d)
    d = clean.copy()
    for i, c in cells(0.002, ["target_1h"]):
        d.loc[i, c] = d.loc[i, c] * 1000
    add("typo_x1000", "8 значений target_1h с тремя лишними нулями", d)
    d = clean.copy()
    d["target_1h"] = d["target_1h"] / 1000
    add("thousands", "target_1h в тысячах товаров", d)
    d = clean.copy()
    o = panel.n_times - 1
    d["target_1h"] = np.round(x[d["route_id"].to_numpy(), o - 3 + (d["timestamp"] - d["timestamp"].min()) // STEP])
    add("half_hour", "target_1h за полчаса, а не сумма за час", d)

    # Past the stored history: 11:00 and 11:30 for every route, the same half-hours a week earlier.
    o = panel.n_times - 1
    fut = []
    for k in (1, 2):
        t = o + k - WEEK
        fut.append(pd.DataFrame({"route_id": panel.routes, "timestamp": panel.times[o] + k * STEP,
                                 **{c: np.round(panel.s[i, :, t]) for i, c in enumerate(STATUS_COLS)},
                                 "target_1h": np.round(panel.y[:, t])}))
    full = pd.concat([clean, *fut], ignore_index=True)
    out.append(("future", "чистый файл до 11:30: два получаса после истории", csv(full)))
    drop = full[(full.timestamp == panel.times[o] + 2 * STEP) & (full.route_id.isin(rng.choice(panel.routes, 300, replace=False)))]
    add("future_partial", "то же, но в 11:30 нет 300 маршрутов", full.drop(drop.index))

    # All at once: what one bad export can look like.
    d = pd.concat([clean, clean.sample(frac=0.03, random_state=4),
                   clean[clean.route_id < 3].assign(route_id=lambda x: x.route_id + 1000)], ignore_index=True)
    for i, c in cells(0.005):
        d.loc[i, c] = -1
    for i, c in cells(0.0015, ["target_1h"]):
        d.loc[i, c] = d.loc[i, c] * 1000
    d["timestamp"] = d["timestamp"] - pd.Timedelta(hours=3) + pd.to_timedelta(rng.integers(-90, 91, len(d)), unit="s")
    d = d.sample(frac=1, random_state=5)
    d.insert(0, "Склад", "Коледино")
    d.columns = ["Склад", *RU]
    d["Время"] = d["Время"].dt.strftime("%d.%m.%Y %-H:%M:%S")
    for c in RU[2:]:
        d[c] = d[c].map(lambda v: f"{v:,.0f}".replace(",", " "))
    add("mess", "всё сразу: Excel по-русски, UTC, секунды, повторы, −1, лишние нули, чужие маршруты, лишняя колонка",
        raw=d.to_csv(index=False, sep=";").encode("cp1251"))
    return out


def total(res: dict) -> np.ndarray:
    return np.array([r["c50"][-1] for r in res["routes"]], dtype=np.float64)


def engine():
    """The /api/forecast engine itself (backend/forecast.py), loaded in this process."""
    sys.path.insert(0, str(ROOT / "backend"))
    from forecast import Engine

    e = Engine()
    e._load()
    if e.error:
        raise SystemExit(e.error)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    t0 = time.time()
    eng = engine()
    print(f"engine {time.time() - t0:.0f} s")
    rows, ref = [], {}
    for key, what, raw in cases(eng.panel, eng.cums.x):
        if args.only and key not in args.only and key not in ("clean", "future"):
            continue
        t1 = time.time()
        try:
            res = eng.run(raw)
        except UploadError as e:
            rows.append((key, what, "отказ", str(e), None, None, None))
            continue
        except Exception as e:  # the screen would show a 500
            rows.append((key, what, "сбой", f"{type(e).__name__}: {e}", None, None, None))
            continue
        tot = total(res)
        if key in ("clean", "future"):
            ref[key] = tot
        r = ref["future" if key.startswith("future") else "clean"]
        net = tot.sum() / r.sum() - 1
        wape = np.abs(tot - r).sum() / r.sum()
        notes = "; ".join(f"{c['text']}" for c in res.get("checks", []))
        rows.append((key, what, "прогноз", notes, pd.Timestamp(res["now"]).strftime("%d.%m %H:%M"), net, wape))
        print(f"{key:15s} {time.time() - t1:5.1f} s  {res['now']}  net {net:+.4f}  routes {wape:.4f}")

    print()
    for key, what, status, notes, now, net, wape in rows:
        num = f"от {now}, сеть {net:+.1%}, маршруты {wape:.1%}" if now else ""
        print(f"{key:15s} {status:8s} {num:42s} {notes[:150]}")

    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for key, what, raw in cases(eng.panel, eng.cums.x):
            (OUT / f"{key}.csv").write_bytes(raw)
            if key == "mess":
                MESS.write_bytes(raw)
        print(f"files: {OUT}, {MESS}")


if __name__ == "__main__":
    main()
