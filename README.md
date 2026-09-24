# RouteWise
## Cost-Aware Multi-LLM Gateway for Coding Assistants

> **Resume integrity note:** Provider prices change, benchmark results depend on the exact models and hardware, and the cost/quality numbers in your current resume draft are only targets until you run this project.

## 1. Project in one sentence

Build a provider-agnostic middleware layer for coding assistants that learns which available LLM is likely to solve each request, routes the request to the lowest-cost model satisfying a quality threshold, and escalates uncertain or failed requests to stronger models while measuring quality, latency, tokens, and cost.

---

# 2. Product definition

A normal coding assistant is configured like:

```text
Coding App
    ↓
One fixed LLM
```

RouteWise changes this to:

```text
Coding App
    ↓
RouteWise Gateway
    ↓
Request Features
    ↓
Learned Success Predictor
    ↓
Cost / Latency Policy
    ↓
+-------------+-------------+-------------+
|             |             |             |
Small Local   Mid Model     Strong Model
|             |             |             |
+-------------+-------------+-------------+
              ↓
        Verification
              ↓
       return / escalate
```

The user can configure the providers/models that **you actually support**.

Do not claim “works with any LLM” unless you truly provide a universal-compatible interface.

---

# 3. Core question

For prompt `x` and model `m`, estimate:

```text
P(success | x, m)
```

Then solve:

```text
choose lowest-cost model
subject to predicted success >= quality threshold
```

Optionally include latency:

```text
utility =
quality benefit
- λ1 * cost
- λ2 * latency
```

This makes RouteWise a real routing/optimization project rather than a set of `if/else` rules.

---

# 4. Datasets

## 4.1 HumanEval

Official repository:

https://github.com/openai/human-eval

Use for programming-function generation with automated correctness tests.

Important: model-generated code is untrusted. Execute it only in an isolated sandbox/container with:
- strict timeout,
- CPU/memory limits,
- no unnecessary network access,
- no host filesystem access.

## 4.2 MBPP

Official Google Research repository:

https://github.com/google-research/google-research/tree/master/mbpp

Use as an additional set of short programming tasks.

## 4.3 Coding instruction tasks

Add a public instruction dataset whose license you verify, or create your own derived task set from public coding exercises.

Useful categories:

```text
code generation
bug fixing
explanation
refactoring
algorithm selection
test generation
code review
```

Start the learned routing benchmark with **testable coding tasks** because correctness can be automatically verified.

---

# 5. Model pool

Start with three tiers:

```text
Tier 1: small local/open model
Tier 2: medium model
Tier 3: strongest hosted/open model you can access
```

Each model configuration:

```yaml
name: model-a
provider: local
endpoint: http://...
context_window: ...
input_cost_per_million: ...
output_cost_per_million: ...
supports_tools: false
```

Hosted-provider prices should be versioned with a date because pricing changes.

API keys belong in:

```text
environment variables
```

or a secret manager, never Git.

---

# 6. Why you need an empirical routing dataset

You cannot reliably label prompts as “easy” or “hard” from intuition alone.

Run each benchmark prompt through every candidate model.

Store:

```text
prompt_id
model_id
response
correct
quality_score
input_tokens
output_tokens
latency_ms
cost
```

This creates a performance matrix:

| Prompt | Small | Medium | Strong |
|---|---:|---:|---:|
| P1 | pass | pass | pass |
| P2 | fail | pass | pass |
| P3 | fail | fail | pass |

For P1, the optimal model may be Small.
For P2, Medium.
For P3, Strong.

That becomes supervised routing data.

---

# 7. Cost calculation

For hosted models:

```text
request cost =
(input_tokens / 1,000,000) * input_rate
+
(output_tokens / 1,000,000) * output_rate
```

For local models, decide how you represent cost.

Options:

```text
GPU-seconds
estimated cloud GPU cost
normalized cost unit
```

Do not pretend local inference is literally free.

---

# 8. Code-execution sandbox

For HumanEval/MBPP correctness, execute generated code in a sandbox.

Recommended design:

```text
generation
   ↓
temporary isolated container
   ↓
copy only task + candidate
   ↓
run tests with timeout
   ↓
capture pass/fail
   ↓
destroy container
```

Security constraints:
- no privileged container,
- no Docker socket mounted,
- network disabled unless required,
- strict memory/CPU limits,
- execution timeout.

The official HumanEval repository itself warns about executing untrusted model-generated code, so sandboxing is part of a correct implementation.

---

# 9. Repository structure

```text
routewise/
├── README.md
├── configs/
│   ├── models.yaml
│   ├── pricing.yaml
│   └── routing.yaml
├── data/
│   ├── load_humaneval.py
│   ├── load_mbpp.py
│   └── taxonomy.py
├── benchmark/
│   ├── run_model_matrix.py
│   ├── sandbox.py
│   ├── evaluators.py
│   └── results/
├── features/
│   ├── embeddings.py
│   ├── scalar_features.py
│   └── task_type.py
├── router/
│   ├── train.py
│   ├── calibrate.py
│   ├── policy.py
│   └── escalation.py
├── gateway/
│   ├── api.py
│   ├── schemas.py
│   └── adapters/
├── eval/
│   ├── evaluate_router.py
│   └── pareto.py
└── tests/
```

---

# 10. Phase 1 — Build provider adapters

Define a common interface:

```python
class ModelAdapter:
    async def generate(self, messages, **kwargs):
        ...

    def estimate_cost(self, usage):
        ...
```

Implement only adapters you can test.

Examples:
- local OpenAI-compatible server,
- hosted provider A,
- hosted provider B.

Normalize result:

```json
{
  "text": "...",
  "input_tokens": 123,
  "output_tokens": 88,
  "latency_ms": 930,
  "model": "..."
}
```

---

# 11. Phase 2 — Benchmark every model

For each task:

```text
prompt
   ↓
model A → evaluate
model B → evaluate
model C → evaluate
```

Use deterministic or carefully controlled decoding for the core benchmark.

Record raw result rows in Parquet/CSV.

Never train the router while benchmark collection is incomplete or silently changing.

---

# 12. Correct train/validation/test split

Split at the **prompt/task level**.

If multiple variants come from the same underlying programming problem, keep them in the same split to avoid leakage.

Suggested:

```text
70% router train
15% router validation
15% router test
```

Keep official benchmark test semantics in mind if you also intend to report canonical benchmark numbers.

For a project-specific routing study, clearly state how you constructed the split.

---

# 13. Baseline A — Strongest model only

All prompts:

```text
→ strongest model
```

Record:

```text
success rate
cost/request
cost/successful request
p50
p95
```

This is the quality-first baseline.

---

# 14. Baseline B — Cheapest model only

All prompts:

```text
→ cheapest model
```

This gives the opposite extreme.

You expect:
- low cost,
- lower quality.

---

# 15. Baseline C — Rule-based router

Example heuristic:

```text
short explanation → small
simple function generation → medium
complex debugging/long context → strong
```

The learned router needs to outperform or improve the cost-quality trade-off over this simple baseline.

---

# 16. Prompt feature extraction

Use three families.

## 16.1 Embedding features

Encode prompt text with a small embedding model.

You can:
- use full embedding with XGBoost if dimensions are manageable,
- reduce dimensions with PCA if desired.

## 16.2 Scalar features

Examples:

```text
prompt token count
code token count
number of code blocks
number of functions/classes in context
stack trace present?
number of files/snippets
requested output length
test-related keywords
debugging keywords
```

## 16.3 Task category

Labels such as:

```text
generation
debugging
explanation
refactoring
testing
algorithm
```

This can be:
- dataset metadata,
- a small classifier,
- a deterministic taxonomy.

---

# 17. Define model success

For testable tasks:

```text
success = all required tests pass
```

You can also use partial score:

```text
passed_tests / total_tests
```

For non-executable explanations, either:
- exclude them initially,
- use human/rubric evaluation,
- add a separate judge with clearly documented limitations.

A strong first version focuses on objectively verifiable coding tasks.

---

# 18. Router target construction

For each prompt, find all models that pass.

Then target:

```text
cheapest passing model
```

If no model passes:

```text
strongest model
```

or label as “none”.

Store:

```json
{
  "prompt_id": "P100",
  "optimal_model": "medium",
  "small_success": 0,
  "medium_success": 1,
  "strong_success": 1
}
```

---

# 19. XGBoost success prediction

A cleaner approach than directly predicting model name:

Train:

```text
P(success of Small | prompt)
P(success of Medium | prompt)
P(success of Strong | prompt)
```

Either:
- one classifier per model,
- one classifier using model ID as an input.

Then policy selects based on cost.

Why this is better:

If prices change, you do not necessarily need to retrain the success estimator. The policy can recompute which capable model is cheapest.

---

# 20. Probability calibration

Routing uses thresholds such as:

```text
P(success) >= 0.80
```

Therefore the probabilities should be calibrated.

Evaluate:
- reliability diagram,
- Brier score.

If needed use:
- isotonic calibration,
- Platt/sigmoid calibration.

Do calibration only on validation data.

---

# 21. Routing policy

Pseudo-logic:

```python
pred = predict_success(prompt)

eligible = [
    m for m in models
    if pred[m] >= threshold
]

if eligible:
    selected = min(eligible, key=expected_cost)
else:
    selected = strongest_model
```

Optional latency constraint:

```text
expected p95 <= latency budget
```

---

# 22. Confidence-based escalation

The first routed model may fail.

For coding tasks you have a powerful verifier:

```text
run tests
```

Flow:

```text
Route to selected model
        ↓
execute tests
        ↓
Pass? -------- yes → return
 |
 no
 ↓
next stronger model
        ↓
test
        ↓
return / escalate again
```

Track:
- first-route success,
- escalation rate,
- final success,
- extra cost due to escalation.

---

# 23. Budget modes

Expose:

```text
quality_first
balanced
cost_first
```

Each policy can change:
- success threshold,
- maximum cost,
- escalation limit.

Do not claim one universal threshold.

Tune thresholds on validation data.

---

# 24. Gateway API

Possible endpoint:

```text
POST /v1/chat/completions
```

or:

```text
POST /route
```

Input:

```json
{
  "messages": [...],
  "routing_policy": "balanced",
  "max_cost": 0.02
}
```

Internal steps:

```text
validate
features
success estimates
policy
provider call
verification
optional escalation
response
```

Return optional routing metadata in debug mode:

```json
{
  "selected_model": "medium",
  "predicted_success": 0.86,
  "estimated_cost": 0.003
}
```

---

# 25. Key metrics

## Final task quality

```text
pass@1 / task success
```

## Economics

```text
average cost/request
total benchmark cost
cost/successful request
cost reduction vs strongest-only
```

## Latency

```text
p50
p95
```

## Routing

```text
first-route success
escalation rate
model usage distribution
```

## Router classification

```text
routing label accuracy
```

Useful, but not the headline.

The real question is:

> How much cost/latency did you save for a given quality level?

---

# 26. Pareto curve

Vary routing threshold.

Plot:

```text
y = task success
x = average cost/request
```

Add:
- cheapest-only,
- strongest-only,
- heuristic router,
- learned router.

A learned router that lies on a better cost-quality frontier is compelling.

---

# 27. Latency measurement

Separate:

```text
router decision latency
provider/model generation latency
sandbox verification latency
end-to-end latency
```

A cheap model may answer fast but frequent escalation can increase overall p95.

Measure the complete workflow.

---

# 28. Benchmark size

Do not force `10K+ prompts`.

Start:

```text
300–500 prompts × 3 models
```

Then:

```text
1,000–2,000
```

Scale only if:
- API credits,
- local GPU time,
- evaluation time

allow it.

Dataset availability is not the major bottleneck; **model-evaluation cost** is.

---

# 29. Ablations

1. heuristic router vs learned.
2. embeddings only.
3. scalar features only.
4. embeddings + scalars.
5. calibrated vs uncalibrated.
6. no escalation vs escalation.
7. quality-first vs balanced vs cost-first.

---

# 30. Optional semantic cache

This can be added later, but do not let it blur the core project.

If implemented, cache:
- exact repeated requests,
- very high-similarity requests with compatible context.

Measure:
- hit rate,
- wrong-cache rate,
- cost reduction.

Keep routing as the project’s primary contribution.

---

# 31. OpenTelemetry

Trace:

```text
gateway request
feature extraction
router prediction
selected provider
generation
verification
escalation
```

Attributes:

```text
model
predicted success
actual success
cost
latency
escalated
```

This gives you a clean LLM infrastructure story.

---

# 32. Final comparison table

| Policy | Task Success | Cost/Req | Cost/Success | p95 | Escalation |
|---|---:|---:|---:|---:|---:|
| Cheapest only | ... | ... | ... | ... | - |
| Strongest only | ... | ... | ... | ... | - |
| Heuristic | ... | ... | ... | ... | ... |
| RouteWise | ... | ... | ... | ... | ... |

This table should directly support the final resume bullet.

---

# 33. Resume claim mapping

### “provider-agnostic”
You need a shared adapter interface and multiple tested provider/local adapters.

### “XGBoost-based routing”
You need a trained model using held-out router test data.

### “46.3% lower cost”
Calculate from the same benchmark as the strongest-only baseline.

### “1.1 pp quality degradation”
Use:
```text
RouteWise success - strongest-only success
```

on the identical test prompts.

### “31.8% lower p95”
Measure end-to-end p95 consistently.

---

# 34. Build order

1. Load HumanEval/MBPP.
2. Build sandbox.
3. Add 2 model adapters.
4. Run 100-prompt matrix.
5. Build strongest/cheapest baselines.
6. Add third model.
7. Extract features.
8. Train XGBoost success estimators.
9. Add calibration.
10. Add routing policy.
11. Add test-based escalation.
12. Build gateway API.
13. Run full held-out benchmark.
14. Create Pareto plot.
15. Add telemetry.

---

# 35. References

- HumanEval: https://github.com/openai/human-eval
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- XGBoost: https://xgboost.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/
- OpenTelemetry: https://opentelemetry.io/docs/
