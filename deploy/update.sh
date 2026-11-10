#!/usr/bin/env bash
# Download tested images; build/training never runs on the production server.
set -Eeuo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${RWB_DEPLOY_ENV:-$repo_dir/../deploy.env}"
[[ $EUID -eq 0 && -f "$env_file" ]] || { echo 'Run as administrator with deploy.env.' >&2; exit 1; }
git_owner="$(stat -c %U "$repo_dir")"
git_read=(runuser -u "$git_owner" -- git -C "$repo_dir")
[[ "$("${git_read[@]}" branch --show-current)" == "${RWB_DEPLOY_BRANCH:-main}" ]]
[[ -z "$("${git_read[@]}" status --porcelain)" ]] || { echo 'Server working tree has local changes.' >&2; exit 1; }
release_sha="$("${git_read[@]}" rev-parse HEAD)"
export RWB_IMAGE_TAG="$release_sha"
export RWB_WEB_IMAGE="ghcr.io/l3ihs/rwb-shipment-forecasting-web:$release_sha"
export RWB_API_IMAGE="ghcr.io/l3ihs/rwb-shipment-forecasting-api:$release_sha"
compose=(docker compose --env-file "$env_file" -f "$repo_dir/deploy/compose.yml")
previous_web="$(docker inspect -f '{{.Config.Image}}' rwb-web 2>/dev/null || true)"
previous_api="$(docker inspect -f '{{.Config.Image}}' rwb-api 2>/dev/null || true)"
"${compose[@]}" config --quiet
# Old containers keep serving while completed runner images are downloaded.
"${compose[@]}" pull web api

rollback() {
  trap - ERR INT TERM HUP
  echo 'Release verification failed; restoring the previous images.' >&2
  if [[ -n "$previous_web" ]]; then
    export RWB_WEB_IMAGE="$previous_web"
    if [[ -n "$previous_api" ]]; then
      export RWB_API_IMAGE="$previous_api"
      "${compose[@]}" up -d --no-build --no-deps --wait --wait-timeout 120 api || true
    else
      "${compose[@]}" stop api || true
    fi
    "${compose[@]}" up -d --no-build --no-deps --wait --wait-timeout 60 web || true
  fi
  exit 1
}
trap rollback ERR INT TERM HUP
"${compose[@]}" up -d --no-build --no-deps --wait --wait-timeout 120 api
"${compose[@]}" up -d --no-build --no-deps --wait --wait-timeout 60 web
python3 "$repo_dir/deploy/verify.py" "$release_sha" "$previous_web" "$previous_api" "$repo_dir/.."
trap - ERR INT TERM HUP
"${compose[@]}" ps api web
echo "Updated https://rwb.ochev.net to $release_sha (frontend + backend + saved ML model)"
