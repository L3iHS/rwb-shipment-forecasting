"""Dispatcher console backend: POST /api/agent (Claude with tools over board.json) and the built frontend.

    uvicorn app:app --port 8000          # from backend/, needs ANTHROPIC_API_KEY
"""

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# backend/.env (KEY=value lines) for local runs; real environment variables win.
_env = Path(__file__).with_name(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

import tools  # noqa: E402  (reads BOARD from the environment)

MODEL = os.environ.get("MODEL", "claude-opus-5-5")
RATE = int(os.environ.get("RATE_PER_10MIN", "30"))  # requests per IP per 10 minutes
STATIC = Path(os.environ.get("STATIC", Path(__file__).resolve().parents[1] / "frontend/dist"))

SYSTEM = f"""Ты ассистент диспетчера склада в пульте вызова машин. Склад отгружает товары по 1000 маршрутам.
Сейчас {tools.clock(0)}, прогноз на 4 часа вперёд, до {tools.clock(8)}. Шаг данных 30 минут.

Решение, когда вызывать машину, принимает правило сервиса: следующая машина нужна, когда груз у ворот плюс ожидаемый
приток покрывают на кузов больше, чем уже вызвано; вызывают её за время подачи до этого момента. Ты объясняешь это
решение и пересчитываешь сценарии, но не меняешь его.

Отвечай по-русски, 1–3 коротких предложения, без markdown-разметки и без вступлений. Сначала ответ, потом одна причина.
Каждое число бери только из результатов функций; если нужного числа нет, вызови функцию или скажи, что не знаешь.
Округляй: груз у ворот — до десятых кузова, объёмы — до тыс. или млн товаров. Правило вызова пересказывай, только если
о нём спросили. Вместимость кузова и время подачи — параметры сервиса, в данных их нет; не выдавай их за факт.
Функции, которые ты вызываешь, сами переключают экран пульта (открывают маршрут, фильтруют список, меняют условия);
не сообщай об этом."""

client = anthropic.Anthropic()
app = FastAPI(title="Пульт диспетчера")


class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class View(BaseModel):
    tab: str = "shift"
    route: int | None = None
    lead_h: float = tools.B["model"]["lead_h"]
    risk: Literal["p50", "p90"] = "p50"


class AgentRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=20)
    view: View = View()


_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(ip: str) -> None:
    q, now = _hits[ip], time.time()
    while q and now - q[0] > 600:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "Слишком много вопросов, попробуйте через несколько минут")
    q.append(now)


@app.post("/api/agent")
def agent(req: AgentRequest, request: Request) -> dict:
    rate_limit(request.client.host if request.client else "?")
    view = req.view.model_dump()
    forecast = "осторожный" if view["risk"] == "p90" else "обычный"
    opened = f", открыт маршрут № {view['route']}" if view["route"] is not None else ""
    state = f"[Пульт: подача машины {view['lead_h']} ч, прогноз {forecast}{opened}]"
    messages = [m.model_dump() for m in req.messages]
    messages[-1]["content"] = f"{state}\n{messages[-1]['content']}"

    actions: dict[str, dict] = {}  # last action of each type
    used = []
    for _ in range(6):
        try:
            resp = client.beta.messages.create(
                model=MODEL, max_tokens=16000, system=SYSTEM, tools=tools.TOOLS, messages=messages,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except anthropic.RateLimitError:
            raise HTTPException(503, "Модель перегружена, попробуйте позже")
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"Ошибка модели: {e.status_code}")
        except anthropic.APIConnectionError:
            raise HTTPException(502, "Нет связи с моделью")
        except TypeError:  # the SDK raises it when no API key or auth token is configured
            raise HTTPException(503, "Ассистент не настроен: нет ключа API")

        if resp.stop_reason == "refusal":
            return {"reply": "Не могу ответить на этот вопрос. Спросите о маршрутах, машинах или прогнозе.", "actions": [], "tools": used}
        if resp.stop_reason != "tool_use":
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            return {"reply": text, "actions": list(actions.values()), "tools": used}

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            out, act = tools.run(b.name, b.input, view)
            used.append({"name": b.name, "args": b.input})
            if act:
                actions[act["type"]] = act
                if act["type"] == "scenario":  # later calls see the switched scenario
                    view.update(lead_h=act["lead_h"], risk=act["risk"])
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, ensure_ascii=False),
                            "is_error": "error" in out})
        messages.append({"role": "user", "content": results})
    return {"reply": "Не успел собрать ответ, переформулируйте вопрос.", "actions": list(actions.values()), "tools": used}


if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
