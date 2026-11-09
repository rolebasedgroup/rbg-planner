# Metrics Query Configuration

The planner queries Prometheus to observe the live TTFT/ITL of the serving
workload. Because each inference engine exposes metrics under different names
and shapes, the planner does not hardcode them: it resolves **one PromQL query
per logical metric slot**, either from a built-in engine preset or from
user-supplied queries in the AutoScaler CR.

## Metric slots

| Slot | Meaning | Used by the planner |
|------|---------|---------------------|
| `ttft` | Mean time to first token (seconds) | Prefill scaling signal |
| `itl` | Mean inter-token latency (seconds) | Decode scaling signal |
| `requestsTotal` | Observed request rate (requests/second) | Load estimation |
| `requestDuration` | Mean end-to-end request duration (seconds) | Load estimation |
| `isl` | Mean input sequence length (tokens) | Load prediction |
| `osl` | Mean output sequence length (tokens) | Load prediction |

Each slot is a complete, self-contained PromQL expression. When a query
returns multiple series, the adapter averages them.

## Engine presets

`spec.implementation.dynamoPlanner.metrics.engine` selects a preset that
provides all six queries:

| Engine | ISL/OSL query shape | Model label |
|--------|---------------------|-------------|
| `sglang` (default) | counter ratio (`sglang:*` metrics with colon prefix) | `model_name` |
| `vllm` | counter ratio | `model_name` |
| `dynamo` | histogram ratio (`_sum` / `_count` of `dynamo_frontend_*`) | `model` |
| `patio` | counter ratio | `model_name` |

Presets live in `python/planner/rbg_planner/metrics/defaults.py`
(`BUILTIN_ENGINES`). The engine names track the inference engine, not the
bench backend: every `sglang-*` backend (`sglang-oai`, `sglang-pd-native`,
...) shares the `sglang` preset.

## Placeholders

Preset templates (and user queries) may reference two placeholders, expanded
when the adapter is constructed:

- `{modelFilter}` -- the model label matcher, including braces. Filled from
  `spec.implementation.dynamoPlanner.modelName` using the preset's model
  label: `sglang` + `Qwen/Qwen3.5-4B` expands to `{model_name="Qwen/Qwen3.5-4B"}`.
  When no model name is configured it expands to `{}` (match all).
- `{interval}` -- the planner's adjustment interval as a Prometheus duration
  string (e.g. `180s`), so rate/increase windows track the planner loop.

For example, the `sglang` `ttft` template:

```text
rate(sglang:time_to_first_token_seconds_sum{modelFilter}[{interval}])
  / rate(sglang:time_to_first_token_seconds_count{modelFilter}[{interval}])
```

## Per-slot overrides

`spec.implementation.dynamoPlanner.metrics.queries` overrides individual
slots with complete PromQL; omitted slots fall back to the engine preset:

```yaml
spec:
  implementation:
    dynamoPlanner:
      modelName: Qwen/Qwen3.5-4B
      metrics:
        engine: sglang
        queries:
          ttft: >-
            sum(rate(my_ttft_seconds_sum{modelFilter}[{interval}]))
            / sum(rate(my_ttft_seconds_count{modelFilter}[{interval}]))
          isl: >-
            sum(increase(my_prompt_tokens{modelFilter}[{interval}]))
            / sum(increase(my_requests{modelFilter}[{interval}]))
```

Query keys are validated by CEL rules in the CRD and must be one of the six
slot names.

## Custom engines

Set `metrics.engine` to any non-builtin name and provide **all six** queries
via `metrics.queries` -- there is no preset to fall back to. The planner
rejects a custom engine with missing slots at startup.

## Planner environment variables

The operator injects the resolved configuration into the planner Deployment:

| Env var | Source |
|---------|--------|
| `METRIC_ENGINE` | `spec.implementation.dynamoPlanner.metrics.engine` |
| `METRIC_QUERIES_JSON` | JSON encoding of `metrics.queries` (unset when empty) |
