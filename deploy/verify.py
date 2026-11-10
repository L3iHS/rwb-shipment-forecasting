"""Public release smoke check, including actual ML inference, without a paid LLM request."""
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request

commit, previous_web, previous_api, state_dir = sys.argv[1:]
url = "https://rwb.ochev.net"


def get(path, body=None, expected=200):
    request = urllib.request.Request(url + path, data=body, headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(request, timeout=70) as response:
            assert response.status == expected
            return response.read()
    except urllib.error.HTTPError as error:
        assert error.code == expected, (path, error.code, expected)
        return error.read()


def container(path):
    return subprocess.check_output(["docker", "exec", "rwb-web", "cat", "/srv/" + path])


assert get("/healthz").strip() == b"ok"
health = json.loads(get("/api/health"))
assert health["model_ready"] and health["release"] == commit, health
index = get("/")
assert index == container("index.html"), "HTML differs from the release"
assets = re.findall(rb'(?:src|href)="(/assets/[^\"]+)"', index)
assert assets
for asset in assets:
    path = asset.decode()
    assert get(path) == container(path.lstrip("/")), path
board = get("/board.json")
assert board == container("board.json") and len(json.loads(board)["routes"]) == 1000
get("/assets/missing.js", expected=404)
bad = json.loads(get("/api/forecast", b"route_id,timestamp\n1,2025-01-01\n", expected=422))
assert "status_1" in bad["detail"]
forecast = json.loads(get("/api/forecast", container("demo/zator_0900-1030.csv")))
assert len(forecast["routes"]) == 1000 and forecast["changed"] == 200
assert len(forecast["all"]["p50"]) == 8 and all(x >= 0 for x in forecast["all"]["p50"])
state_path = pathlib.Path(state_dir) / "deployment.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state.update(deployed_commit=commit, previous_web_image=previous_web, previous_api_image=previous_api,
             verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             public_https_verified=True, public_assets_match_local_build=True,
             frontend_demo_only=False, backend_model_ready=True, public_api_verified=True,
             agent_configured=health["agent_configured"], web_runtime="caddy",
             board_sha256=hashlib.sha256(board).hexdigest())
state_path.write_text(json.dumps(state, indent=2) + "\n")
state_path.chmod(0o600)
print(f"HTTPS, {len(assets)} assets and actual ML file forecast verified: {commit}")
