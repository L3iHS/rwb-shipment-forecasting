#!/usr/bin/env bash
# Install root-owned as /usr/local/sbin/rwb-deploy-main. No arbitrary refs or commands.
set -Eeuo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Expected one full commit SHA.' >&2
  exit 64
fi
if [[ $EUID -ne 0 ]]; then
  echo 'This command must run through the restricted deploy account.' >&2
  exit 77
fi
release_sha="$1"
repo_dir=/srv/rwb-shipment-forecasting/app
exec 9>/run/lock/rwb-main-release.lock
flock -w 180 9
git_read=(runuser -u rwb -- env GIT_TERMINAL_PROMPT=0 git -C "$repo_dir")
[[ "$("${git_read[@]}" branch --show-current)" == main ]] || { echo 'Expected main.' >&2; exit 1; }
[[ -z "$("${git_read[@]}" status --porcelain)" ]] || { echo 'Server working tree has local changes.' >&2; exit 1; }
"${git_read[@]}" fetch --no-tags origin refs/heads/main:refs/remotes/origin/main
latest_sha="$("${git_read[@]}" rev-parse refs/remotes/origin/main)"
if [[ "$release_sha" != "$latest_sha" ]]; then
  echo "Superseded release $release_sha; main is already at $latest_sha. Skipping."
  exit 0
fi
"${git_read[@]}" merge --ff-only "$release_sha"
[[ "$("${git_read[@]}" rev-parse HEAD)" == "$release_sha" ]]
started_at=$SECONDS
RWB_DEPLOY_BRANCH=main bash "$repo_dir/deploy/update.sh"
echo "Server deployment completed in $((SECONDS - started_at))s."
