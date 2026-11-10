"""Assistant tools over board.json. Same numbers as frontend/src/tools.ts (see docs/web.md)."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

BOARD_PATH = Path(os.environ.get("BOARD", Path(__file__).resolve().parents[1] / "frontend/public/board.json"))
B = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
R = B["routes"]
NOW = datetime.fromisoformat(B["now"])
STEP = B["step_min"]
BY_ID = {r["id"]: r for r in R}


def clock(k: int) -> str:
    return (NOW + timedelta(minutes=k * STEP)).strftime("%H:%M")


def band(p10: list[float], p50: list[float], p90: list[float]) -> float:
    """Half-width of the 80% band relative to the forecast over the horizon: 0.44 means ±44%."""
    return sum(hi - lo for lo, hi in zip(p10, p90)) / 2 / max(1, sum(p50))


def peak(p50: list[float]) -> int:
    return max(range(len(p50)), key=p50.__getitem__)


def route_forecast(route_id: int) -> dict:
    r = BY_ID.get(route_id)
    if r is None:
        return {"error": f"маршрута {route_id} нет, номера 0..{len(R) - 1}"}
    k = peak(r["p50"])
    return {
        "route_id": route_id, "volume_4h": r["c50"][-1], "horizon_end": clock(len(r["p50"])),
        "peak_at": clock(k + 1), "peak_per_hour": r["p50"][k], "band_80": round(band(r["p10"], r["p50"], r["p90"]), 3),
        "history_8h": {clock(i - len(r["h"]) + 1): v for i, v in enumerate(r["h"])},
        "forecast": [{"at": clock(i + 1), "p10": lo, "p50": mid, "p90": hi}
                     for i, (lo, mid, hi) in enumerate(zip(r["p10"], r["p50"], r["p90"]))],
        "check_week": {"error_wape": r.get("wape"), "bias": r.get("bias")},
    }


ORDERS = ["largest", "least_accurate", "most_accurate", "widest_band"]


def find_routes(order: str, limit: int) -> dict:
    key = {
        "largest": lambda r: -r["c50"][-1],
        "least_accurate": lambda r: -(r.get("wape") or 0),
        "most_accurate": lambda r: r.get("wape") or 0,
        "widest_band": lambda r: -band(r["p10"], r["p50"], r["p90"]),
    }[order]
    top = sorted(R, key=lambda r: (key(r), r["id"]))[:limit]
    return {"order": order, "routes": [{"route_id": r["id"], "volume_4h": r["c50"][-1], "error_wape": r.get("wape"),
                                        "band_80": round(band(r["p10"], r["p50"], r["p90"]), 3)} for r in top]}


def network_forecast() -> dict:
    n = B["network"]
    k = peak(n["p50"])
    return {
        "volume_4h": sum(r["c50"][-1] for r in R), "routes": len(R), "now": clock(0), "horizon_end": clock(len(n["p50"])),
        "peak_at": clock(k + 1), "peak_per_hour": n["p50"][k], "band_80": round(band(n["lo"], n["p50"], n["hi"]), 3),
        "forecast": [{"at": clock(i + 1), "p50": mid, "lo": lo, "hi": hi}
                     for i, (lo, mid, hi) in enumerate(zip(n["lo"], n["p50"], n["hi"]))],
    }


def model_quality() -> dict:
    bt = B["backtest"]
    return {"week": bt["week"], "metric_case_from_1030": bt.get("point_1030"), "metric_case_all_origins": bt["point"],
            "band_80_coverage": bt["intervals"].get("80%"), "per_day": B["monitor"],
            "metric": "WAPE + |относительный сдвиг суммы|, чем меньше, тем лучше"}


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    {"name": "route_forecast", "strict": True,
     "description": "Прогноз маршрута на 4 часа: объём, пик, коридор 80 %, история за 8 ч, прогноз по получасам "
                    "(p10, p50, p90), ошибка и сдвиг на проверочной неделе.",
     "input_schema": _schema({"route_id": {"type": "integer"}}, ["route_id"])},
    {"name": "find_routes", "strict": True,
     "description": "Маршруты по признаку: largest — больше всего товаров за 4 ч, least_accurate — самая большая "
                    "ошибка на проверке, most_accurate — самая маленькая, widest_band — самый широкий коридор.",
     "input_schema": _schema({"order": {"type": "string", "enum": ORDERS}, "limit": {"type": "integer"}},
                             ["order", "limit"])},
    {"name": "network_forecast", "strict": True,
     "description": "Прогноз по всем маршрутам вместе: товаров за 4 ч, пик, коридор 80 %, прогноз по получасам.",
     "input_schema": _schema({}, [])},
    {"name": "model_quality", "strict": True,
     "description": "Точность модели на неделе, которую она не видела: метрика кейса, попадание в коридор 80 %, "
                    "ошибка и сдвиг по дням.",
     "input_schema": _schema({}, [])},
]


def run(name: str, args: dict, view: dict) -> tuple[dict, dict | None]:
    """Execute a tool; return its result and the screen action it implies (open a route)."""
    if name == "route_forecast":
        out = route_forecast(args["route_id"])
        return out, None if "error" in out else {"type": "open_route", "id": args["route_id"]}
    if name == "find_routes":
        out = find_routes(args["order"], max(1, min(args["limit"], 20)))
        first = out["routes"][0]["route_id"] if out["routes"] else None
        return out, None if first is None else {"type": "open_route", "id": first}
    if name == "network_forecast":
        return network_forecast(), None
    if name == "model_quality":
        return model_quality(), None
    return {"error": f"нет функции {name}"}, None
