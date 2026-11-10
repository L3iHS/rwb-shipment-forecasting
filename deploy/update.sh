#!/usr/bin/env bash
# Run as the server administrator after pulling the release branch.
set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${RWB_DEPLOY_ENV:-$repo_dir/../deploy.env}"
public_url="https://rwb.ochev.net"
release_branch="${RWB_DEPLOY_BRANCH:-main}"

if [[ $EUID -ne 0 ]]; then
  echo 'Run with sudo or from the administrator shell.' >&2
  exit 1
fi
[[ -f "$env_file" ]] || { echo "Missing $env_file" >&2; exit 1; }
git_owner="$(stat -c %U "$repo_dir")"
git_read=(runuser -u "$git_owner" -- git -C "$repo_dir")
[[ "$("${git_read[@]}" branch --show-current)" == "$release_branch" ]] || { echo "Expected $release_branch." >&2; exit 1; }
[[ -z "$("${git_read[@]}" status --porcelain)" ]] || { echo 'Working tree has local changes; leaving the running site unchanged.' >&2; exit 1; }
release_sha="$("${git_read[@]}" rev-parse HEAD)"
export RWB_IMAGE_TAG="$release_sha"
compose=(docker compose --env-file "$env_file" -f "$repo_dir/deploy/compose.yml")
previous_image="$(docker inspect -f '{{.Config.Image}}' rwb-web 2>/dev/null || true)"
previous_sha="${previous_image#rwb-shipment-web:}"

"${compose[@]}" config --quiet
# Build first: the old container keeps serving until the new image is ready.
"${compose[@]}" build web

rollback() {
  trap - ERR INT TERM HUP
  echo 'Release verification failed.' >&2
  if [[ "$previous_image" == rwb-shipment-web:* && "$previous_sha" =~ ^[a-f0-9]{40}$ ]]; then
    echo "Restoring $previous_sha" >&2
    RWB_IMAGE_TAG="$previous_sha" "${compose[@]}" up -d --no-build --no-deps --wait --wait-timeout 90 web || true
  fi
  exit 1
}
trap rollback ERR INT TERM HUP
"${compose[@]}" up -d --no-build --no-deps --wait --wait-timeout 90 web

# Verify HTTPS, the data snapshot and every asset referenced by this exact build.
python3 - "$public_url" "$release_sha" "$previous_sha" "$repo_dir/.." <<'PY'
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request

url, commit, previous, state_dir = sys.argv[1:]
def get(path):
    with urllib.request.urlopen(url + path, timeout=20) as response:
        return response.read()
def container(path):
    return subprocess.check_output(['docker', 'exec', 'rwb-web', 'cat', '/srv/' + path])
assert get('/healthz').strip() == b'ok', 'Health endpoint failed'
index = get('/')
assert index == container('index.html'), 'Public HTML does not match the release'
assets = re.findall(rb'(?:src|href)="(/assets/[^\"]+)"', index)
assert assets, 'No build assets in HTML'
for asset in assets:
    path = asset.decode()
    assert get(path) == container(path.lstrip('/')), f'Asset mismatch: {path}'
board = get('/board.json')
assert board == container('board.json'), 'Snapshot mismatch'
assert json.loads(board)['routes'], 'Empty snapshot'
for path in ['/api/agent', '/assets/missing.js']:
    try:
        get(path)
    except urllib.error.HTTPError as error:
        assert error.code == 404, (path, error.code)
    else:
        raise AssertionError(f'Expected 404: {path}')
state_path = pathlib.Path(state_dir) / 'deployment.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state.update(deployed_commit=commit, previous_commit=previous,
             verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             public_https_verified=True, public_assets_match_local_build=True,
             frontend_demo_only=True, theme_modes=['light', 'dark'], web_runtime='caddy',
             board_sha256=hashlib.sha256(board).hexdigest())
state_path.write_text(json.dumps(state, indent=2) + '\n')
state_path.chmod(0o600)
print(f'HTTPS, snapshot and {len(assets)} assets verified: {commit}')
PY
trap - ERR INT TERM HUP
"${compose[@]}" ps web
echo "Updated $public_url to $release_sha"
