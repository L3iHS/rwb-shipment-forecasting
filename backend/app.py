"""Dispatcher console backend: POST /api/agent (Claude with tools over board.json) and the built frontend.

    uvicorn app:app --port 8000          # from backend/, needs ANTHROPIC_API_KEY

The endpoint is public, so every question has a fixed cost ceiling: short history, short answers, few model calls,
few tool calls, a deadline, per-IP and global limits, and a cap on questions answered at the same time.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

# backend/.env (KEY=value lines) for local runs; real environment variables win.
_env = Path(__file__).with_name(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

import tools  # noqa: E402  (reads BOARD from the environment)

MODEL = os.environ.get("MODEL", "claude-opus-5-5")
RATE = int(os.environ.get("RATE_PER_10MIN", "30"))  # questions per IP per 10 minutes
RATE_HOUR = int(os.environ.get("RATE_PER_HOUR", "300"))  # questions per hour from all visitors together
CONCURRENCY = int(os.environ.get("AGENT_CONCURRENCY", "4"))  # questions answered at the same time
STATIC = Path(os.environ.get("STATIC", Path(__file__).resolve().parents[1] / "frontend/dist"))

STEPS = 4  # model calls per question; the last one may not call tools, so it has to answer
TOOLS_PER_STEP = 4
MAX_TOKENS = 1024
DEADLINE = 45.0  # seconds per question
KEEP = 8  # latest messages of the conversation sent to the model
MAX_MSG = 1000  # characters of one message the model sees
MAX_CHARS = 6000  # their total length
MAX_BODY = 64_000  # bytes of a request to /api/
ACTION_ORDER = ["scenario", "filter", "open_route", "tab"]  # a filter must not replace the opened route; the tab wins

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
не сообщай об этом. На вопросы не о работе склада и пульта отвечай одной фразой, что помогаешь только со сменой."""

# The base URL is fixed: an ANTHROPIC_BASE_URL left in the shell would otherwise receive the key.
# No SDK retries: they sleep for the server's retry-after and would break the deadline.
client = anthropic.Anthropic(base_url="https://api.anthropic.com", timeout=20, max_retries=0)
log = logging.getLogger("uvicorn.error")
app = FastAPI(title="Пульт диспетчера")


class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)  # past replies can be long; history() cuts them


class View(BaseModel):
    tab: Literal["shift", "evidence"] = "shift"
    route: int | None = Field(None, ge=0, le=1_000_000)
    lead_h: float = tools.B["model"]["lead_h"]
    risk: Literal["p50", "p90"] = "p50"

    @field_validator("lead_h")
    @classmethod
    def known_lead(cls, v: float) -> float:
        if v not in tools.LEADS:
            raise ValueError(f"подача машины одна из {tools.LEADS}")
        return v


class AgentRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=50)
    view: View = View()

    @model_validator(mode="after")
    def ends_with_question(self) -> "AgentRequest":
        if self.messages[-1].role != "user":
            raise ValueError("последнее сообщение — вопрос пользователя")
        if any(m.role == "user" and len(m.content) > MAX_MSG for m in self.messages):
            raise ValueError(f"вопрос длиннее {MAX_MSG} знаков")
        return self


def history(msgs: list[Msg]) -> list[dict]:
    """The latest messages within KEEP and MAX_CHARS, starting with a question, each cut to MAX_MSG."""
    kept = [Msg(role=m.role, content=m.content[:MAX_MSG]) for m in msgs[-KEEP:]]
    while sum(len(m.content) for m in kept) > MAX_CHARS and len(kept) > 1:
        kept = kept[1:]
    while kept[0].role != "user":  # ends with a question, so this stops
        kept = kept[1:]
    return [m.model_dump() for m in kept]


_lock = threading.Lock()
_hits: dict[str, deque] = {}
_all: deque = deque()
_busy = threading.BoundedSemaphore(CONCURRENCY)


def admit(ip: str) -> None:
    now = time.monotonic()
    with _lock:
        if len(_hits) > 10_000:  # forget visitors idle for 10 minutes
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > 600]:
                del _hits[k]
        q = _hits.setdefault(ip, deque())
        while q and now - q[0] > 600:
            q.popleft()
        while _all and now - _all[0] > 3600:
            _all.popleft()
        if len(q) >= RATE:
            raise HTTPException(429, "Слишком много вопросов, попробуйте через несколько минут")
        if len(_all) >= RATE_HOUR:
            raise HTTPException(429, "Ассистент исчерпал лимит вопросов на этот час")
        q.append(now)
        _all.append(now)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/"):
        n = request.headers.get("content-length", "")
        if not n.isdigit() or int(n) > MAX_BODY:
            return JSONResponse({"detail": "Слишком большой запрос"}, 413)
    return await call_next(request)


def ask_model(messages: list[dict], last: bool, left: float):
    try:
        return client.with_options(timeout=min(20.0, left)).beta.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, tools=tools.TOOLS, messages=messages,
            tool_choice={"type": "none"} if last else {"type": "auto"},
            cache_control={"type": "ephemeral"}, output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    except anthropic.EnforcedSpendLimitReachedError:
        raise HTTPException(503, "Ассистент исчерпал лимит расходов")
    except (anthropic.RateLimitError, anthropic.OverloadedError, anthropic.ServiceUnavailableError):
        raise HTTPException(503, "Модель перегружена, попробуйте позже")
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        raise HTTPException(503, "Ассистент не настроен: ключ API не подходит")
    except anthropic.APITimeoutError:
        raise HTTPException(504, "Модель не ответила вовремя")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Ошибка модели: {e.status_code}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Нет связи с моделью")
    except anthropic.AnthropicError:
        raise HTTPException(502, "Ошибка модели")


@app.post("/api/agent")
def agent(req: AgentRequest, request: Request) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "Ассистент не настроен: нет ключа API")
    if not _busy.acquire(blocking=False):
        raise HTTPException(503, "Ассистент занят, спросите через минуту")
    try:
        admit(request.client.host if request.client else "?")
        return answer(req)
    finally:
        _busy.release()


def answer(req: AgentRequest) -> dict:
    view = req.view.model_dump()
    forecast = "осторожный" if view["risk"] == "p90" else "обычный"
    opened = f", открыт маршрут № {view['route']}" if view["route"] is not None else ""
    state = f"[Пульт: подача машины {view['lead_h']} ч, прогноз {forecast}{opened}]"
    messages = history(req.messages)
    messages[-1]["content"] = f"{state}\n{messages[-1]['content']}"

    actions: dict[str, dict] = {}  # last action of each type
    used = []

    def reply(text: str) -> dict:
        return {"reply": text, "actions": [actions[k] for k in ACTION_ORDER if k in actions], "tools": used}

    deadline = time.monotonic() + DEADLINE
    for step in range(STEPS):
        left = deadline - time.monotonic()
        if left < 3:
            break
        resp = ask_model(messages, last=step == STEPS - 1, left=left)
        log.info("agent step %d: %s, tokens in %d out %d", step + 1, resp.stop_reason,
                 resp.usage.input_tokens, resp.usage.output_tokens)

        if resp.stop_reason == "refusal":
            actions.clear()
            return reply("Не могу ответить на этот вопрос. Спросите о маршрутах, машинах или прогнозе.")
        if resp.stop_reason != "tool_use":
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            return reply(text or "Не получилось ответить, переформулируйте вопрос.")

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for i, b in enumerate(b for b in resp.content if b.type == "tool_use"):
            if i >= TOOLS_PER_STEP:
                out = {"error": f"не больше {TOOLS_PER_STEP} функций за раз, ответь по уже полученным данным"}
            else:
                try:
                    out, act = tools.run(b.name, b.input, view)
                except (KeyError, TypeError, ValueError):  # arguments outside the schema
                    out, act = {"error": "неверные аргументы функции"}, None
                used.append({"name": b.name, "args": b.input})
                if act:
                    actions[act["type"]] = act
                    if act["type"] == "scenario":  # later calls see the switched scenario
                        view.update(lead_h=act["lead_h"], risk=act["risk"])
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, ensure_ascii=False),
                            "is_error": "error" in out})
        messages.append({"role": "user", "content": results})
    return reply("Не успел собрать ответ, переформулируйте вопрос.")


if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
