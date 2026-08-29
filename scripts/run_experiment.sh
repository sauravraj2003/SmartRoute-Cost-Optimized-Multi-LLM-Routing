#!/usr/bin/env bash
set -euo pipefail

# Arguments are passed to benchmark collection (for example: --dataset mbpp --limit 100).
python -m benchmark.run_model_matrix "$@" --output benchmark/results/matrix.csv
python -m router.train benchmark/results/matrix.csv
python -m eval.evaluate_router benchmark/results/matrix.csv
python -m eval.ablations benchmark/results/matrix.csv
