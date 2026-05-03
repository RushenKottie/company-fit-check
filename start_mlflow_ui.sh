#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

BACKEND_STORE_URI="${MLFLOW_TRACKING_URI:-file://$SCRIPT_DIR/.mlruns}"

exec "$SCRIPT_DIR/.venv/bin/mlflow" ui \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --host "${MLFLOW_UI_HOST:-127.0.0.1}" \
  --port "${MLFLOW_UI_PORT:-5000}"
