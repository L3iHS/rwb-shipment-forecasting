#!/usr/bin/env bash
# Root-owned forced command for the dedicated GitHub Actions SSH key.
set -euo pipefail
if [[ "${SSH_ORIGINAL_COMMAND:-}" =~ ^deploy\ ([0-9a-f]{40})$ ]]; then
  exec /usr/bin/sudo -n /usr/local/sbin/rwb-deploy-main "${BASH_REMATCH[1]}"
fi
echo 'Only deploy <40-character commit SHA> is allowed.' >&2
exit 64
