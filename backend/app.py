"""Dispatcher console backend: POST /api/agent (RouterAI with tools over board.json), POST /api/forecast (the service
model on an uploaded file) and the built frontend.

    uvicorn app:app --port 8000          # from backend/, assistant needs ROUTERAI_API_KEY

The endpoint is public, so every question has a fixed cost ceiling: short history, short answers, few model calls,
few tool calls, a deadline, per-IP and global limits, and a cap on questions answered at the same time.
"""

import json
import logging
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
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
from forecast import ENGINE  # noqa: E402

MODEL = os.environ.get("MODEL", "deepseek/deepseek-v4.1-flash-1017")
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
MAX_UPLOAD = 16_000_000  # bytes of a file to /api/forecast
UPLOADS = int(os.environ.get("UPLOADS_PER_10MIN", "20"))  # forecasts per IP per 10 minutes
UPLOAD_CONCURRENCY = int(os.environ.get("UPLOAD_CONCURRENCY", "1"))  # each peaks near 900 MB over the 1.5 GB engine
ACTION_ORDER = ["open_route"]

SYSTEM = f"""Ты ассистент на экране прогноза отгрузок склада. Склад отгружает товары по 1000 маршрутам.
Сейчас {tools.clock(0)}, прогноз на 4 часа вперёд, до {tools.clock(8)}, шаг 30 минут. Модель (LightGBM) прогнозирует,
сколько товаров маршрут отгрузит за час, и даёт диапазон прогноза: в 8 случаях из 10 факт внутри него. Называй его «диапазон», не «коридор». По одному
маршруту диапазон широкий, потому что товары уходят партиями; по всем маршрутам вместе он узкий.

Отвечай по-русски, 1–3 коротких предложения, без markdown-разметки и без вступлений. Длинное тире не ставь:
пиши запятую или двоеточие. Сначала ответ, потом одна причина.
Каждое число бери только из результатов функций; если нужного числа нет, вызови функцию или скажи, что не знаешь.
Округляй объёмы до тыс. или млн товаров. Доли пиши процентами: band_80 0,435 — «диапазон ±44 %», error_wape 0,386 — «ошибка 39 %»,
bias −0,003 — «сдвиг −0,3 %». Только метрику кейса (metric_case_*) пиши числом до тысячных: 0,328.
Машин, их вместимости и времени подачи в данных нет: на вопросы о машинах так и отвечай. Функции сами открывают
маршрут на экране; не сообщай об этом. На вопросы не о прогнозе отгрузок
отвечай одной фразой, что помогаешь только с прогнозом."""

# The destination is fixed; unrelated SDK credentials/base URLs in the shell are never used.
client = httpx.Client(base_url="https://routerai.ru/api/v1/", timeout=httpx.Timeout(20.0, connect=5.0), trust_env=False)
log = logging.getLogger("uvicorn.error")
app = FastAPI(title="Пульт диспетчера")
ENGINE.start()


@app.get("/api/health")
def health():
    ready = ENGINE.ready.is_set() and ENGINE.error is None
    return JSONResponse({"status": "ok" if ready else "starting", "model_ready": ready,
                         "agent_configured": bool(os.environ.get("ROUTERAI_API_KEY")),
                         "release": os.environ.get("RWB_RELEASE", "local")}, status_code=200 if ready else 503)


class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)  # past replies can be long; history() cuts them


class View(BaseModel):
    tab: Literal["shift", "evidence"] = "shift"
    route: int | None = Field(None, ge=0, le=1_000_000)  # the route open on screen


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
        limit = MAX_UPLOAD if request.url.path == "/api/forecast" else MAX_BODY
        if not n.isdigit():
            return JSONResponse({"detail": "Нужен заголовок Content-Length"}, 411)
        if int(n) > limit:
            return JSONResponse({"detail": "Файл больше 16 МБ" if limit == MAX_UPLOAD else "Слишком большой запрос"}, 413)
    return await call_next(request)


_uploads: dict[str, deque] = {}
_crunch = threading.BoundedSemaphore(UPLOAD_CONCURRENCY)


@app.post("/api/forecast")
async def forecast_file(request: Request) -> dict:
    """Body: the file itself (CSV or Parquet with the train columns). Answer: the forecast before and after it."""
    ip = request.client.host if request.client else "?"
    now = time.monotonic()
    with _lock:
        q = _uploads.setdefault(ip, deque())
        while q and now - q[0] > 600:
            q.popleft()
        if len(q) >= UPLOADS:
            raise HTTPException(429, "Слишком много расчётов, попробуйте через несколько минут")
    raw = await request.body()
    if not await run_in_threadpool(ENGINE.ready.wait, 60):
        raise HTTPException(503, "Модель ещё загружается, попробуйте через минуту")
    if ENGINE.error:
        raise HTTPException(503, "Модель не загрузилась на сервере")
    if not _crunch.acquire(blocking=False):
        raise HTTPException(503, "Сервер занят другим расчётом, попробуйте через минуту")
    with _lock:  # only a request that gets to compute counts against the limit
        _uploads.setdefault(ip, deque()).append(now)
    try:
        from rwb.upload import UploadError

        try:
            return await run_in_threadpool(ENGINE.run, raw)
        except UploadError as e:
            raise HTTPException(422, str(e))
        except Exception:
            log.exception("forecast on an uploaded file failed")
            raise HTTPException(500, "Не получилось посчитать прогноз по этому файлу")
    finally:
        _crunch.release()


_net = {"down_until": 0.0}  # while the API host does not resolve, answer without the model at once


def reachable() -> bool:
    """Whether the API host resolves within 3 s. A dead network hangs in DNS for 15 s, which no client timeout cuts;
    after a failure the next 30 s skip the check, so later questions get the answer without the model at once."""
    if time.monotonic() < _net["down_until"]:
        return False
    found = threading.Event()

    def lookup() -> None:
        try:
            socket.getaddrinfo("routerai.ru", 443)
            found.set()
        except OSError:
            pass

    threading.Thread(target=lookup, daemon=True).start()
    if found.wait(3.0):
        return True
    _net["down_until"] = time.monotonic() + 30
    return False


def ask_model(messages: list[dict], last: bool, left: float):
    if not reachable():
        raise HTTPException(502, "Нет связи с моделью")
    try:
        response = client.post("chat/completions", headers={"Authorization": "Bearer " + os.environ["ROUTERAI_API_KEY"]},
            timeout=httpx.Timeout(min(20.0, left), connect=5.0), json={
                "model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0.2,
                "messages": [{"role": "system", "content": SYSTEM}, *messages],
                "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                           "parameters": t["input_schema"]}} for t in tools.TOOLS],
                "tool_choice": "none" if last else "auto",
            })
        if response.status_code in (401, 403):
            raise HTTPException(503, "Ассистент не настроен: ключ RouterAI не подходит")
        if response.status_code == 402:
            raise HTTPException(503, "Ассистент исчерпал баланс RouterAI")
        if response.status_code == 429 or response.status_code >= 500:
            raise HTTPException(503, "Модель перегружена, попробуйте позже")
        if not response.is_success:
            raise HTTPException(502, "Ошибка модели: " + str(response.status_code))
        message = response.json()["choices"][0]["message"]
        if not isinstance(message, dict):
            raise ValueError("Expected a message object")
        return message
    except httpx.TimeoutException:
        raise HTTPException(504, "Модель не ответила вовремя")
    except httpx.RequestError:
        raise HTTPException(502, "Нет связи с моделью")
    except (ValueError, KeyError, IndexError, TypeError):
        raise HTTPException(502, "Некорректный ответ модели")


@app.post("/api/agent")
def agent(req: AgentRequest, request: Request) -> dict:
    if not os.environ.get("ROUTERAI_API_KEY"):
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
    state = f"[Экран: открыт маршрут № {view['route']}]" if view["route"] is not None else "[Экран: прогноз по всем маршрутам]"
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
        if resp.get("refusal"):
            actions.clear()
            return reply("Не могу ответить на этот вопрос. Спросите о маршрутах, прогнозе или его точности.")
        calls = resp.get("tool_calls") or []
        if not calls:
            text = (resp.get("content") or "").strip()
            return reply(text or "Не получилось ответить, переформулируйте вопрос.")

        assistant = {"role": "assistant", "content": resp.get("content"), "tool_calls": calls}
        for field in ("reasoning_content", "reasoning_details"):
            if field in resp:
                assistant[field] = resp[field]
        messages.append(assistant)
        for i, call in enumerate(calls):
            function = call.get("function", {})
            name = function.get("name", "")
            if i >= TOOLS_PER_STEP:
                out = {"error": f"не больше {TOOLS_PER_STEP} функций за раз, ответь по уже полученным данным"}
            else:
                try:
                    args = json.loads(function.get("arguments", "{}"))
                    out, act = tools.run(name, args, view)
                except (KeyError, TypeError, ValueError):  # arguments outside the schema
                    out, act = {"error": "неверные аргументы функции"}, None
                    args = {}
                used.append({"name": name, "args": args})
                if act:
                    actions[act["type"]] = act
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(out, ensure_ascii=False)})
    return reply("Не успел собрать ответ, переформулируйте вопрос.")


if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
