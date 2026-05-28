# RBG Planner: SLA-Driven Autoscaling for Prefill-Decode Disaggregated Inference

*May 2025*

## TL;DR

We introduce **RBG Planner**, a standalone SLA-based autoscaler for Prefill-Decode (PD) disaggregated LLM inference workloads running on Kubernetes. By combining offline performance profiling, online load prediction, and real-time correction factors, the planner automatically scales prefill and decode replicas to meet TTFT (Time to First Token) and ITL (Inter-Token Latency) SLA targets while minimizing GPU usage.

Key results:
- Automatically maintains SLA compliance under dynamic load patterns
- Reduces GPU waste by 40-60% compared to static over-provisioning
- Supports SGLang, vLLM, and engine-agnostic (Patio) metric sources
- Full profiling-to-deployment pipeline via `inference-ext-cli`

---

## Background: The PD Disaggregation Scaling Problem

### Why Disaggregation?

Modern LLM serving systems increasingly adopt **Prefill-Decode disaggregation** to optimize hardware utilization. The insight is simple: prefill (prompt encoding) and decode (token generation) have fundamentally different computational profiles.

| Phase | Compute Pattern | Bottleneck | Latency Metric |
|-------|----------------|------------|----------------|
| Prefill | Compute-bound, batched matrix ops | FLOPs | TTFT |
| Decode | Memory-bound, sequential KV cache reads | Memory bandwidth | ITL |

By separating these phases into dedicated GPU pools, each can be independently optimized and scaled — prefill nodes maximize throughput per GPU, while decode nodes minimize per-token latency.

### The Scaling Challenge

However, disaggregation introduces a **multi-dimensional scaling problem** that static provisioning cannot solve:

```
                    Request Load
                         │
              ┌──────────┴──────────┐
              │                     │
         Prefill Load          Decode Load
              │                     │
    ┌─────────┴─────────┐   ┌──────┴──────┐
    │                   │   │             │
  ISL × Rate      Rate × OSL     Concurrency × Context
    │                   │   │             │
    v                   v   v             v
  TTFT ≤ SLA?      Throughput?  ITL ≤ SLA?  KV Cache?
```

The optimal number of prefill and decode replicas depends on:
1. **Request rate** — how many requests per second
2. **Input sequence length (ISL)** — determines prefill compute cost
3. **Output sequence length (OSL)** — determines decode duration
4. **Concurrency** — active decodes sharing KV cache memory
5. **SLA targets** — the TTFT and ITL thresholds users require

These dimensions interact non-linearly. A 2x increase in ISL might require 3x prefill throughput (due to quadratic attention), while the same traffic might only need 1.2x decode capacity. Static replica counts either waste GPUs during valleys or violate SLAs during peaks.

### Existing Approaches Fall Short

| Approach | Problem |
|----------|---------|
| HPA on CPU/memory | Irrelevant for GPU inference workloads |
| HPA on request rate | Ignores ISL/OSL distribution changes |
| HPA on latency (TTFT/ITL) | Reactive — SLA violated before scale-up |
| Manual GPU budgeting | Cannot adapt to diurnal/weekly patterns |

What we need is a **proactive, SLA-aware** scaler that:
- Understands the performance characteristics of the specific model + hardware
- Predicts upcoming load (not just reacts to current state)
- Independently scales prefill and decode to their respective SLA targets
- Respects a total GPU budget constraint

---

## Solution: RBG Planner

### Architecture

The RBG Planner runs as a lightweight sidecar role within a [RoleBasedGroup](https://github.com/rolebasedgroup/rbg) — the Kubernetes CRD that manages multi-role inference deployments.

```
┌─────────────────────────────────────────────────────────┐
│                    RoleBasedGroup                        │
│                                                         │
│  ┌─────────┐     ┌─────────────┐     ┌──────────────┐  │
│  │ Prefill │     │   Planner   │     │    Decode    │  │
│  │ Role    │     │   Role      │     │    Role      │  │
│  │ (N GPU) │◄────│             │────►│   (M GPU)    │  │
│  └─────────┘     │  ┌───────┐  │     └──────────────┘  │
│                  │  │Profile│  │                        │
│                  │  │ Data  │  │                        │
│                  │  └───────┘  │                        │
│                  └──────┬──────┘                        │
│                         │                               │
└─────────────────────────┼───────────────────────────────┘
                          │ query
                          ▼
                   ┌──────────────┐
                   │  Prometheus  │
                   │  (SGLang/    │
                   │   vLLM/Patio)│
                   └──────────────┘
```

### The Planning Loop

Every `ADJUSTMENT_INTERVAL` seconds (default: 180s), the planner executes a four-phase cycle:

#### Phase 1: Observe

Query Prometheus for the current state of the inference engines:
- **TTFT** — average time to first token (ms)
- **ITL** — average inter-token latency (ms)
- **Request count** — total requests in the interval
- **ISL/OSL** — average input/output sequence lengths

```python
# Metrics are engine-agnostic via configurable metric source
interval = f"{adjustment_interval}s"
ttft = metrics_client.get_avg_time_to_first_token(interval, model)
itl = metrics_client.get_avg_inter_token_latency(interval, model)
num_req = metrics_client.get_request_count(interval, model)
isl = metrics_client.get_avg_input_sequence_tokens(interval, model)
osl = metrics_client.get_avg_output_sequence_tokens(interval, model)
```

#### Phase 2: Predict

Feed observed metrics into a time-series predictor to estimate the **next** interval's load. This is what makes the planner proactive rather than reactive.

Supported predictors:
- **Constant** — assumes next interval equals current (baseline)
- **ARIMA** — auto-regressive integrated moving average (default)
- **Prophet** — Facebook's forecasting library (for longer patterns)

```python
next_num_req = num_req_predictor.predict_next()  # e.g., 1200
next_isl = isl_predictor.predict_next()          # e.g., 2048
next_osl = osl_predictor.predict_next()          # e.g., 256
```

#### Phase 3: Compute

This is the core intelligence. Using **offline profiling data**, the planner translates predicted load into required replicas.

**Prefill replicas:**
```
predicted_prefill_throughput = next_num_req × next_isl / interval × correction_factor
engine_capacity = interpolate_thpt_per_gpu(next_isl) × gpus_per_engine
num_prefill = ⌈predicted_prefill_throughput / engine_capacity⌉
```

**Decode replicas:**
```
corrected_itl_target = itl_sla / d_correction_factor
decode_thpt_per_gpu = find_best_throughput(itl ≤ corrected_itl_target, context_length)
predicted_decode_throughput = next_num_req × next_osl / interval
num_decode = ⌈predicted_decode_throughput / (decode_thpt_per_gpu × gpus_per_engine)⌉
```

The profiling interpolators provide the mapping from workload characteristics to hardware performance — this is what makes the decisions model-specific and hardware-aware.

#### Phase 4: Scale

Apply the computed replica counts via the Kubernetes API:
1. Try scaling via **RBGSA** (RoleBasedGroupScalingAdapter) `/scale` subresource — compatible with HPA
2. Fall back to direct **RBG patch** on `spec.roles[].replicas`

```python
target_replicas = [
    TargetReplica(role_name="prefill", desired_replicas=num_prefill),
    TargetReplica(role_name="decode", desired_replicas=num_decode),
]
await connector.set_replicas(target_replicas)
```

### Correction Factors

Profiling data represents idealized, isolated performance. Real-world deployments experience queuing delays, network overhead, and resource contention. The planner compensates using **correction factors**:

```
p_correction = observed_ttft / expected_ttft(current_isl)
d_correction = observed_itl / expected_itl(current_concurrency, current_context_length)
```

When `p_correction > 1.0`, the system is performing worse than profiled — the planner increases replicas more aggressively. When `< 1.0`, the system is outperforming expectations — the planner can be more conservative.

This creates a feedback loop that adapts to the actual deployment environment without requiring re-profiling.

### GPU Budget Enforcement

The planner enforces a hard GPU ceiling to prevent runaway scaling:

```python
total_gpu = num_prefill × prefill_gpus + num_decode × decode_gpus
if total_gpu > max_gpu_budget:
    # Proportionally scale down both roles
    scale = max_gpu_budget / total_gpu
    num_prefill = max(min_replicas, round(num_prefill * scale))
    num_decode = max(min_replicas, ...)
```

---

## Profiling Pipeline

The quality of scaling decisions depends entirely on profiling data quality. We provide `inference-ext-cli` — a complete profiling pipeline that generates the performance model.

### Two-Phase Profiling

**Phase 1: Parallelization Sweep**

For each GPU count (1, 2, 4, ..., max_gpus), deploy temporary RBG instances with each candidate parallelization mapping (TP, TEP, DEP for MoE models) and measure TTFT/ITL at the target ISL/OSL:

```bash
inference-ext-cli profile \
  --engine sglang \
  --model "Qwen/Qwen3-0.6B" \
  --engine-image "lmsysorg/sglang:latest" \
  --min-gpus 1 --max-gpus 4 \
  --isl 3000 --osl 500 \
  --ttft-sla 200 --itl-sla 20
```

Selection criteria: **within SLA, maximize throughput/GPU**.

**Phase 2: Interpolation Sweep**

With the selected optimal configuration:
- **Prefill**: sweep ISL from 100 to max_context_length → build `ISL → (TTFT, throughput/GPU)` curve
- **Decode**: 2D sweep of `(ISL × concurrency)` → build `(KV_usage, context_length) → (ITL, throughput/GPU)` surface

### Profiling Data Format

The profiling produces two JSON files mounted as a ConfigMap:

```json
// prefill_raw_data.json — 1D interpolation
{
  "prefill_isl": [128, 256, 512, 1024, 2048, 4096, 8192],
  "prefill_ttft": [0.005, 0.009, 0.018, 0.035, 0.070, 0.140, 0.290],
  "prefill_thpt_per_gpu": [8000, 7000, 5500, 4000, 2800, 1800, 1000]
}
```

```json
// decode_raw_data.json — 2D scatter interpolation
{
  "x_kv_usage": [0.02, 0.05, 0.10, ...],      // KV cache utilization
  "y_context_length": [350, 350, 700, ...],     // context length at each point
  "z_itl": [0.006, 0.008, 0.010, ...],         // measured ITL (seconds)
  "z_thpt_per_gpu": [1500, 1350, 1200, ...],   // measured throughput
  "max_kv_tokens": 32768
}
```

The planner uses scipy cubic interpolation to query any point in this performance space at runtime.

---

## Deployment: One Command

The entire workflow — from existing RBG YAML to a planner-integrated deployment — is a single CLI command:

```bash
# Generate RBG with planner + profiling ConfigMap
inference-ext-cli generate \
  --rbg-yaml ./my-sglang-pd.yaml \
  --enable-planner \
  --planner-image ghcr.io/rolebasedgroup/rbg-planner:latest \
  --model-name "Qwen/Qwen3-0.6B" \
  --profiling-source json \
  --prefill-json ./prefill_raw_data.json \
  --decode-json ./decode_raw_data.json \
  --ttft-sla 200 --itl-sla 20 \
  --max-gpu-budget 8 \
  -o ./output/

# Deploy
kubectl apply -f ./output/profiling-configmap.yaml
kubectl apply -f ./output/rbg.yaml
```

The generated RBG YAML contains the original prefill/decode roles plus a new `planner` role with all configuration pre-wired:

```yaml
- name: planner
  replicas: 1
  standalonePattern:
    template:
      spec:
        containers:
          - name: rbg-planner
            image: ghcr.io/rolebasedgroup/rbg-planner:latest
            env:
              - name: RBG_NAME
                value: sglang-pd-inference
              - name: TTFT_SLA
                value: "200.0"
              - name: ITL_SLA
                value: "20.0"
              - name: LOAD_PREDICTOR
                value: arima
              ...
            volumeMounts:
              - name: profiling-data
                mountPath: /etc/rbg-planner/profiling
        volumes:
          - name: profiling-data
            configMap:
              name: sglang-pd-inference-profiling
```

---

## Example: Scaling Under Load

Here's what the planner produces under three traffic scenarios for Qwen3-0.6B on a cluster with 8 GPU budget:

| Scenario | Req/interval | ISL | OSL | Prefill | Decode | Total GPU |
|----------|-------------|-----|-----|---------|--------|-----------|
| Light | 100 | 512 | 128 | 1 | 1 | 2 |
| Medium | 500 | 1024 | 200 | 1 | 3 | 4 |
| Heavy | 2000 | 2048 | 512 | 2 | 6 | 8 (capped) |

The planner recognizes that:
- At light load, minimum replicas suffice for both SLAs
- At medium load, decode needs more replicas because ITL is the binding constraint at higher concurrency
- At heavy load, the GPU budget caps total allocation — the planner allocates more to decode (the tighter constraint) while keeping prefill at minimum viable

---

## Observability

The planner exposes its own Prometheus metrics for monitoring scaling behavior:

```
rbg_planner_observed_ttft_ms          # Are we meeting TTFT SLA?
rbg_planner_observed_itl_ms           # Are we meeting ITL SLA?
rbg_planner_predicted_request_rate    # What does the predictor expect?
rbg_planner_predicted_num_prefill     # Planned prefill replicas
rbg_planner_predicted_num_decode      # Planned decode replicas
rbg_planner_p_correction_factor       # How far off is profiling from reality?
rbg_planner_d_correction_factor       # (>1 means worse than expected)
rbg_planner_gpu_hours_total           # Cost tracking
```

A correction factor consistently above 1.5 signals that re-profiling is needed or that the deployment has systematic resource contention.

---

## Comparison with Alternatives

| Feature | HPA (k8s native) | KEDA | RBG Planner |
|---------|------------------|------|-------------|
| PD-aware | No | No | Yes |
| Proactive (predictive) | No | No | Yes (ARIMA/Prophet) |
| SLA-driven | No | Partial | Yes |
| Profiling-based | No | No | Yes |
| GPU budget enforcement | No | No | Yes |
| Correction factors | No | No | Yes |
| Engine-agnostic | N/A | N/A | Yes (SGLang/vLLM/Patio) |

---

## Future Work

- **Patio EngineRuntime integration** — unified `patio:*` metrics make the planner fully engine-agnostic without per-engine metric configuration
- **Multi-model support** — scaling decisions across multiple models sharing the same GPU pool
- **Cost-aware mode** — incorporate spot/on-demand pricing into scaling decisions
- **Gradual scale-down** — hysteresis to avoid thrashing during oscillating load

---

## Getting Started

```bash
# Install the CLI
pip install inference-ext-cli

# Run profiling (deploys temporary instances, benchmarks, collects data)
inference-ext-cli profile \
  --engine sglang \
  --model "your-model" \
  --engine-image "lmsysorg/sglang:latest" \
  --ttft-sla 200 --itl-sla 20

# Generate deployment with planner
inference-ext-cli generate \
  --rbg-yaml ./your-rbg.yaml \
  --enable-planner \
  --planner-image ghcr.io/rolebasedgroup/rbg-planner:latest \
  --profiling-source json \
  --prefill-json ./profiling-results/prefill_raw_data.json \
  --decode-json ./profiling-results/decode_raw_data.json \
  --model-name "your-model" \
  --ttft-sla 200 --itl-sla 20 \
  -o ./output/

# Deploy
kubectl apply -f ./output/
```

---

## Links

- [RBG Planner Repository](https://github.com/rolebasedgroup/rbg-planner)
- [inference-ext-cli Repository](https://github.com/rolebasedgroup/inference-ext-cli)
- [RoleBasedGroup (RBG)](https://github.com/rolebasedgroup/rbg)
- [Deployment SOP](./sop.md)
