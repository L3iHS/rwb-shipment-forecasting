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


def lead_steps(lead_h: float) -> int:
    return round(lead_h * 60 / STEP)


def call_step(r: dict, lead: int, risk: str) -> int:
    """Steps until the next truck call (0 = now, -1 = not within the horizon); rwb.dispatch.call_plan."""
    inflow = r["c90"] if risk == "p90" else r["cr"]
    backlog = r["bl"] * r["cap"]
    need = r["cap"] * (1 + r["way"] + r["dock"])
    if backlog >= need:
        return 0
    for k in range(len(inflow) - lead + 1):
        if backlog + inflow[k + lead - 1] >= need:
            return k
    return -1


def plan(lead_h: float, risk: str) -> list[int]:
    return [call_step(r, lead_steps(lead_h), risk) for r in R]


def spread(r: dict) -> float:
    """Relative width of the 80% band."""
    return sum(hi - lo for hi, lo in zip(r["p90"], r["p10"])) / max(1, sum(r["p50"]))


LOW_CONFIDENCE = sorted(spread(r) for r in R)[len(R) * 2 // 3]


def route_status(route_id: int, lead_h: float, risk: str) -> dict:
    r = BY_ID.get(route_id)
    if r is None:
        return {"error": f"маршрута {route_id} нет, номера 0..{len(R) - 1}"}
    call = call_step(r, lead_steps(lead_h), risk)
    return {
        "route_id": route_id, "call_at": None if call < 0 else clock(call), "call_now": call == 0,
        "backlog_loads": r["bl"], "trucks_called": r["way"] + r["dock"], "capacity": r["cap"],
        "history_8h": {clock(i - len(r["h"]) + 1): v for i, v in enumerate(r["h"])},
        "forecast": [{"at": clock(k + 1), "p10": lo, "p50": mid, "p90": hi}
                     for k, (lo, mid, hi) in enumerate(zip(r["p10"], r["p50"], r["p90"]))],
        "inflow_4h": r["c50"][-1], "inflow_4h_p90": r["c90"][-1],
        "confidence": "низкая" if spread(r) > LOW_CONFIDENCE else "обычная",
    }


def find_routes(condition: str, limit: int, lead_h: float, risk: str) -> dict:
    def matches(r: dict, call: int) -> bool:
        if condition == "call_now":
            return call == 0
        if condition == "call_within_hour":
            return 0 <= call <= 2
        if condition == "waiting_without_truck":
            return r["bl"] >= 1 and r["way"] + r["dock"] == 0
        return 0 <= call <= 2 and spread(r) > LOW_CONFIDENCE  # low_confidence

    hit = sorted(((r, c) for r, c in zip(R, plan(lead_h, risk)) if matches(r, c)), key=lambda t: -t[0]["bl"])
    return {"count": len(hit), "routes": [{"route_id": r["id"], "backlog_loads": r["bl"],
                                           "call_at": None if c < 0 else clock(c)} for r, c in hit[:limit]]}


def network_summary(lead_h: float, risk: str) -> dict:
    calls = plan(lead_h, risk)
    p50 = B["network"]["p50"]
    peak = max(range(len(p50)), key=p50.__getitem__)
    return {
        "call_now": sum(c == 0 for c in calls), "call_within_hour": sum(0 < c <= 2 for c in calls),
        "new_trucks": sum(max(0, int(r["bl"] + r["c50"][-1] / r["cap"]) - r["way"] - r["dock"]) for r in R),
        "new_trucks_cautious": sum(max(0, int(r["bl"] + r["c90"][-1] / r["cap"]) - r["way"] - r["dock"]) for r in R),
        "volume_4h": sum(r["c50"][-1] for r in R),
        "waiting_without_truck": sum(r["bl"] >= 1 and r["way"] + r["dock"] == 0 for r in R),
        "peak_at": clock(peak + 1), "peak_volume_per_hour": p50[peak], "horizon_end": clock(len(p50)),
    }


def what_if(lead_h: float, risk: str) -> dict:
    base, now = plan(B["model"]["lead_h"], "p50"), plan(lead_h, risk)
    return {"lead_h": lead_h, "risk": risk, "call_now": sum(c == 0 for c in now),
            "call_now_service_plan": sum(c == 0 for c in base), "routes_changed": sum(a != b for a, b in zip(now, base))}


def evidence() -> dict:
    bt = B["backtest"]
    return {"week": bt["week"], "loss_minutes_per_truck": bt["sensitivity"], "per_day": bt["per_day"],
            "band_coverage": bt["intervals"], "quality_log": B["monitor"], "alerts": B["alerts"],
            "note": "вместимость кузова и время подачи — параметры, в данных их нет"}


LEADS = [0.5, 1, 2]
RISKS = ["p50", "p90"]
CONDITIONS = ["call_now", "call_within_hour", "waiting_without_truck", "low_confidence"]


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    {"name": "route_status", "strict": True,
     "description": "Всё о маршруте при текущем сценарии пульта: когда вызывать машину, груз у ворот (в кузовах), "
                    "вызванные машины, история за 8 ч, прогноз отгрузок с диапазоном 10–90 %, приток за 4 ч, уверенность.",
     "input_schema": _schema({"route_id": {"type": "integer"}}, ["route_id"])},
    {"name": "find_routes", "strict": True,
     "description": "Маршруты по условию, по убыванию груза у ворот: call_now — вызвать сейчас, call_within_hour — "
                    "в течение часа, waiting_without_truck — полный кузов у ворот без машины, low_confidence — "
                    "вызов в ближайший час при неуверенном прогнозе.",
     "input_schema": _schema({"condition": {"type": "string", "enum": CONDITIONS}, "limit": {"type": "integer"}},
                             ["condition", "limit"])},
    {"name": "network_summary", "strict": True,
     "description": "Сводка по складу при текущем сценарии: вызовы сейчас и в течение часа, новых машин до конца "
                    "горизонта, объём за 4 ч, груз без машины, пик отгрузок сети.",
     "input_schema": _schema({}, [])},
    {"name": "what_if", "strict": True,
     "description": "Пересчёт очереди при другом времени подачи машины (часы) и сценарии прогноза (p50 обычный, "
                    "p90 осторожный) против плана сервиса. Пульт переключится на этот сценарий.",
     "input_schema": _schema({"lead_h": {"type": "number", "enum": LEADS}, "risk": {"type": "string", "enum": RISKS}},
                             ["lead_h", "risk"])},
    {"name": "evidence", "strict": True,
     "description": "Проверка на истории: потери на машину (минуты) для правил вызова при разных подаче и кузове, "
                    "машины за день, честность диапазонов, журнал качества по дням.",
     "input_schema": _schema({}, [])},
]


def run(name: str, args: dict, view: dict) -> tuple[dict, dict | None]:
    """Execute a tool; return its result and the console action it implies."""
    lead, risk = view["lead_h"], view["risk"]
    if name == "route_status":
        out = route_status(args["route_id"], lead, risk)
        return out, None if "error" in out else {"type": "open_route", "id": args["route_id"]}
    if name == "find_routes":
        out = find_routes(args["condition"], max(1, min(args["limit"], 20)), lead, risk)
        f = {"call_now": "now", "call_within_hour": "hour", "waiting_without_truck": "jam"}.get(args["condition"])
        return out, None if f is None else {"type": "filter", "value": f}
    if name == "network_summary":
        return network_summary(lead, risk), None
    if name == "what_if":
        return what_if(args["lead_h"], args["risk"]), {"type": "scenario", "lead_h": args["lead_h"], "risk": args["risk"]}
    if name == "evidence":
        return evidence(), None
    return {"error": f"нет функции {name}"}, None
