#!/usr/bin/env bash
set -euo pipefail
python -m benchmark.run_model_matrix "$@"
