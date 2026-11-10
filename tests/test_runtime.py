"""Small release gate: actual saved model, uploaded demo and RouterAI tool protocol (mocked, no paid calls)."""

import json
import os
from pathlib import Path
import sys

import httpx
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "src")]
os.environ.pop("ROUTERAI_API_KEY", None)

from rwb.data import Panel
from rwb.features import cumsums
from runtime import mapped_cumsums
import app as backend
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api():
    assert backend.ENGINE.ready.wait(90), "Model startup timed out"
    assert backend.ENGINE.error is None, backend.ENGINE.error
    with TestClient(backend.app) as client:
        yield client


def test_disk_backed_features_equal_original_features(tmp_path):
    rng = np.random.default_rng(7)
    y = rng.uniform(0, 100, (3, 80)).astype("float32")
    s = rng.uniform(0, 100, (6, 3, 80)).astype("float32")
    y[0, 10] = np.nan
    panel = Panel(pd.date_range("2025-01-01", periods=80, freq="30min"), np.arange(3), y, s, pd.DataFrame())
    expected, actual = cumsums(panel), mapped_cumsums(panel, tmp_path, batch=2)
    for name in ("y", "y2", "s", "x", "xc"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_health_requires_loaded_model_and_reports_release(api):
    response = api.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": True, "agent_configured": False,
                               "release": os.environ.get("RWB_RELEASE", "local")}


def test_uploaded_demo_runs_real_model_and_bad_file_is_rejected(api):
    response = api.post("/api/forecast", content=b"route_id,timestamp\n1,2025-01-01\n")
    assert response.status_code == 422 and "status_1" in response.json()["detail"]
    raw = (ROOT / "tests/demo/zator_0900-1030.csv").read_bytes()
    response = api.post("/api/forecast", content=raw)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["routes"]) == 1000 and data["changed"] == 200
    assert len(data["all"]["p50"]) == 8 and all(x >= 0 for x in data["all"]["p50"])
    assert data["baseline"] and data["group"]["routes"] == 200


def test_routerai_tool_roundtrip_and_missing_key(api, monkeypatch):
    question = {"messages": [{"role": "user", "content": "Какой прогноз по маршруту 471?"}], "view": {"route": 471}}
    assert api.post("/api/agent", json=question).status_code == 503
    seen = []

    def provider(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://routerai.ru/api/v1/chat/completions"
        assert body["model"].startswith("deepseek/")
        seen.append(body)
        if len(seen) == 1:
            message = {"content": None, "tool_calls": [{"id": "route-1", "type": "function",
                "function": {"name": "route_forecast", "arguments": '{"route_id":471}'}}]}
        else:
            assert body["messages"][-1]["role"] == "tool"
            message = {"content": "Прогноз маршрута рассчитан по данным пульта."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    monkeypatch.setenv("ROUTERAI_API_KEY", "test-only")
    monkeypatch.setattr(backend, "reachable", lambda: True)
    with httpx.Client(base_url="https://routerai.ru/api/v1/", transport=httpx.MockTransport(provider)) as client:
        monkeypatch.setattr(backend, "client", client)
        response = api.post("/api/agent", json=question)
    assert response.status_code == 200, response.text
    assert len(seen) == 2 and response.json()["tools"][0]["name"] == "route_forecast"
