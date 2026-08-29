$ErrorActionPreference = "Stop"

# Arguments are passed to benchmark collection, for example:
# .\scripts\run_experiment.ps1 --dataset mbpp --limit 100
$benchmarkArgs = @args + @("--output", "benchmark/results/matrix.csv")
python -m benchmark.run_model_matrix @benchmarkArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m router.train benchmark/results/matrix.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m eval.evaluate_router benchmark/results/matrix.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m eval.ablations benchmark/results/matrix.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
