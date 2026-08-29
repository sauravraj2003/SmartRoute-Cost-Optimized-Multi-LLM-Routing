#!/usr/bin/env bash
set -euo pipefail
uvicorn gateway.api:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
