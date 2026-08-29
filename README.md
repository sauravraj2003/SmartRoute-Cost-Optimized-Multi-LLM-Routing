# SmartRoute

SmartRoute is a configurable gateway for coding assistants. It estimates request difficulty, selects the least expensive configured model whose predicted success clears a policy threshold, and records model usage, latency, and estimated cost. It supports OpenAI Chat Completions compatible providers (including Groq and compatible local servers) and Anthropic's Messages API.

## What this does and does not guarantee

Routing probabilities come from a classifier trained on benchmark outcomes. Before training, the predictor uses a neutral 0.5 fallback, so the policy normally selects the highest configured tier. Tier rank is configuration metadata and must be checked against held-out empirical results; it is not a universal ranking of models. A probability is not a guarantee that a response is correct. The gateway can verify a coding response only when the caller supplies `tests`; those tests and generated code then run inside the Docker sandbox. It cannot verify arbitrary UI behavior, explanations, or DSA answers without suitable tests.

The model entries and tier ranks are configurable examples. Rates in `configs/pricing.yaml` were checked on 2026-08-20 and may change; confirm current provider pricing and model availability before making calls. No provider is contacted during installation or startup.

## Install

Requires Python 3.10+. Install dependencies and set provider keys in `.env` (or your process environment):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY`, `GROQ_API_KEY`, and/or `ANTHROPIC_API_KEY` for the configured providers. For a local OpenAI-compatible server, update its model endpoint and set provider to `local`; assign nonzero normalized input/output rates so local inference is not treated as free. Protect a publicly reachable gateway with `SMARTROUTE_API_KEY` and TLS at the proxy.

## Start the API

```bash
uvicorn gateway.api:app --host 127.0.0.1 --port 8000
```

Check `GET /health`, inspect configured models at `GET /v1/models`, or send a routed request:

```bash
curl http://127.0.0.1:8000/route -H "Content-Type: application/json" -d '{"prompt":"Write a Python function that reverses a string", "mode":"balanced"}'
```

`POST /v1/chat/completions` accepts text chat messages and the `max_tokens`/`max_completion_tokens` limit, returning a non-streaming chat-completion response with additional `smartroute` metadata. Tool calls, vision content, and streaming are not implemented. `/route` returns the normalized response contract including the selected provider, predicted probability, cost estimate, routing reason, and request ID. Pass `mode` as `cost_first`, `balanced`, or `quality_first`. An explicit `model` pins a request to a configured model. Add `tests` to request isolated test-based escalation; provide `entry_point` when the test suite defines `check(candidate)`. The Docker sandbox must be available for that request.

Configuration paths can be overridden using `SMARTROUTE_MODELS_CONFIG`, `SMARTROUTE_PRICING_CONFIG`, `SMARTROUTE_ROUTING_CONFIG`, `SMARTROUTE_MODE`, and `SMARTROUTE_MODEL_DIR`. The model directory defaults to `router/models`; absent trained artifacts use the neutral predictor fallback. Install `pip install -e ".[embeddings]"` and set `SMARTROUTE_USE_SENTENCE_TRANSFORMER=true` to use sentence-transformer features instead of deterministic feature hashing; use the same setting for training and serving.

## Benchmark and train

The benchmark requires provider credentials and Docker. It loads public datasets using Hugging Face `datasets` by default; a local JSONL file can be supplied with `--data`. Each JSONL record should contain `prompt`, `tests`, and optionally `prompt_id`, `entry_point`, and `dataset`.

```bash
python -m benchmark.run_model_matrix --dataset mbpp --split train --limit 100
python -m router.train benchmark/results/matrix.csv
python -m eval.evaluate_router benchmark/results/matrix.csv
```

Use `--dataset mbpp`, `--output benchmark/results/matrix.parquet`, or other runner flags as needed. Benchmark rows include one task/model result, response, test outcome, token usage, latency, and estimated cost. Tasks without tests are reported as unscored failures and must not be used as supervised correctness labels.

Training fits one success classifier per model. The training command creates a deterministic 70/15/15 task split, grouping variants that share a `group_id`; fits only on train tasks; calibrates with validation rows when there are enough examples; and saves the test task IDs in `router/models/split.json`. The evaluation command uses those held-out task rows when they are present in its input. This project-specific split is not a canonical HumanEval or MBPP score. For final research results, evaluate on a separate untouched task set. Threshold tuning must use validation data, never the final test split.

The optional latency budget filters models only when each candidate has a measured `typical_latency_ms.p95` value in `models.yaml`; the sample pool leaves this unset. The configured tiers are ordering hints, while the trained predictors and validation results should determine whether the ordering matches your workload.

## Safety and operations

The benchmark runner and optional request verifier execute model output as untrusted code only inside a disposable Docker container with networking disabled, a read-only root, a non-root user, dropped Linux capabilities, and resource limits. Docker must be installed. Never set `require_docker=False` for untrusted output; that mode runs code on the host. API credentials are read from environment variables and should never be committed.

HumanEval uses the [MIT-licensed OpenAI dataset](https://huggingface.co/datasets/openai/openai_humaneval); MBPP is [CC-BY-4.0](https://huggingface.co/datasets/google-research-datasets/mbpp). Keep the original dataset attribution when redistributing derived benchmark artifacts.

Costs are estimated from the dated `configs/pricing.yaml` per-million-token ledger, merged over the model definitions. If a provider omits token usage, the gateway estimates counts from text length and reports `usage_estimated=true`. The estimator supports the configured long-context multiplier but does not model every cache, batch, regional, or provider billing modifier. Keep prices dated and verified. Telemetry emits structured logs and console spans by default; set `OTEL_EXPORTER_OTLP_ENDPOINT` to export traces, or `SMARTROUTE_TRACING_ENABLED=false` to disable tracing.

## Ablations and semantic cache

Run the full experiment flow with `bash scripts/run_experiment.sh` or `scripts/run_experiment.ps1` (PowerShell). Arguments go to benchmark collection, for example `./scripts/run_experiment.ps1 --dataset mbpp --limit 100`. The workflow collects model/task outcomes, trains on the saved train split, evaluates on the held-out test split, and runs the seven planned comparison groups using those same split IDs. It writes the matrix to `benchmark/results/matrix.csv`, evaluation metrics and measured comparisons to `eval/results.json`, and ablations to `eval/ablations.json`; reports include sample counts and success, cost, and latency for each baseline/policy. `eval/results.json` also contains the cost/quality Pareto frontier; add `--pareto-plot eval/pareto.png` to `python -m eval.evaluate_router` to render it.

The SQLite response cache is opt-in under `features.semantic_cache` in `configs/routing.yaml`. Set `enabled: true` to activate it. It stores only responses that passed the caller's supplied tests and scopes matches to identical tests, prior conversation, generation settings, policy mode, and configured model pool. Without optional sentence-transformer embeddings, it safely supports exact prompt matches only; install `.[embeddings]` and set `SMARTROUTE_USE_SENTENCE_TRANSFORMER=true` to enable approximate semantic matches at the configured threshold. Requests without tests are never read from or written to the cache. Cache TTL, maximum size, and file path can be set with `SMARTROUTE_CACHE_TTL_SECONDS`, `SMARTROUTE_CACHE_MAX_ENTRIES`, and `SMARTROUTE_CACHE_PATH`.

Benchmark results and performance claims are generated only after you run the workflow with provider credentials and Docker. The repository does not contain fabricated quality, cost-saving, or latency numbers.

## Project map

- `gateway/`: FastAPI endpoints, request validation, middleware, telemetry and provider adapters.
- `features/` and `router/`: deterministic prompt features, optional embeddings, training, calibration, policy and escalation helpers.
- `benchmark/` and `data/`: dataset loaders, provider matrix collection, Docker test runner and cost calculations.
- `eval/`: baseline metrics, router evaluation and Pareto analysis.
- `configs/`: model pool, example prices, routing modes and telemetry settings.
