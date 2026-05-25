# RBG Planner SOP (Standard Operating Procedure)

End-to-end guide for deploying the SLA-based autoscaler for RoleBasedGroup
Prefill/Decode disaggregated inference workloads.

## Overview

The RBG Planner automatically scales Prefill and Decode roles within a
RoleBasedGroup to meet TTFT (Time to First Token) and ITL (Inter-Token Latency)
SLA targets. It uses:

- **Performance profiling data** to map load characteristics to hardware requirements
- **Load prediction** (ARIMA/Constant/Prophet) to anticipate upcoming demand
- **Correction factors** to account for real-world deviations from profiled behavior

## Prerequisites

1. A Kubernetes cluster with the RBG controller installed
2. A RoleBasedGroup deployed with `prefill` and `decode` roles
3. Prometheus monitoring stack (e.g., kube-prometheus-stack)
4. `inference-ext-cli` installed (`pip install inference-ext-cli`)

## CLI Commands Overview

`inference-ext-cli` provides two commands:

| Command | Purpose |
|---------|---------|
| `generate` | Generate RBG YAML with planner role and profiling ConfigMap |
| `profile` | Run standalone SLA profiling pipeline (deploy, benchmark, collect data) |

## Step 1: Generate RBG with Planner

The `generate` command produces a deployable RBG YAML with the planner role
injected. It supports three ways to provide profiling data.

### Option A: From existing RBG YAML + local JSON profiling files

Use this when you already have an RBG YAML and profiling data files:

```bash
inference-ext-cli generate \
  --rbg-yaml ./my-rbg.yaml \
  --enable-planner \
  --planner-image ghcr.io/rolebasedgroup/rbg-planner:latest \
  --model-name "Qwen/Qwen3-0.6B" \
  --profiling-source json \
  --prefill-json ./prefill_raw_data.json \
  --decode-json ./decode_raw_data.json \
  --ttft-sla 200 \
  --itl-sla 20 \
  --max-gpu-budget 8 \
  -o ./output/
```

This generates:
- `output/rbg.yaml` - RBG with planner role added
- `output/profiling-configmap.yaml` - ConfigMap with profiling data

### Option B: From existing RBG YAML + existing ConfigMap

Use this when profiling data is already deployed as a ConfigMap in the cluster:

```bash
inference-ext-cli generate \
  --rbg-yaml ./my-rbg.yaml \
  --enable-planner \
  --planner-image ghcr.io/rolebasedgroup/rbg-planner:latest \
  --model-name "Qwen/Qwen3-0.6B" \
  --profiling-source configmap \
  --profiling-configmap profiling-data \
  --ttft-sla 200 \
  --itl-sla 20 \
  -o ./output/
```

This generates:
- `output/rbg.yaml` - RBG with planner role referencing the existing ConfigMap

### Option C: From existing RBG YAML + auto profiling

Use this to automatically run the profiling pipeline and generate everything:

```bash
inference-ext-cli generate \
  --rbg-yaml ./my-rbg.yaml \
  --enable-planner \
  --planner-image ghcr.io/rolebasedgroup/rbg-planner:latest \
  --model-name "Qwen/Qwen3-0.6B" \
  --profiling-source auto \
  --engine sglang \
  --engine-image "lmsysorg/sglang:latest" \
  --namespace profiling \
  --min-gpus 1 \
  --max-gpus 4 \
  --ttft-sla 200 \
  --itl-sla 20 \
  -o ./output/
```

This deploys temporary RBG instances, runs AIPerf benchmarks, collects data,
then generates:
- `output/rbg.yaml` - RBG with planner role
- `output/profiling-configmap.yaml` - ConfigMap with collected profiling data
- `output/profiling-artifacts/` - Raw profiling artifacts

### Option D: Generate full RBG from scratch

Use this when you don't have an existing RBG YAML:

```bash
inference-ext-cli generate \
  --engine sglang \
  --model "Qwen/Qwen3-0.6B" \
  --engine-image "lmsysorg/sglang:latest" \
  --prefill-tp 2 \
  --decode-tp 4 \
  --enable-planner \
  --planner-image ghcr.io/rolebasedgroup/rbg-planner:latest \
  --model-name "Qwen/Qwen3-0.6B" \
  --profiling-source json \
  --prefill-json ./prefill_raw_data.json \
  --decode-json ./decode_raw_data.json \
  --ttft-sla 200 \
  --itl-sla 20 \
  -o ./output/
```

This generates a complete PD-disaggregated RBG with prefill, decode, and planner roles.

## Step 2: Deploy

Apply the generated artifacts:

```bash
# Apply profiling ConfigMap (if generated)
kubectl apply -f ./output/profiling-configmap.yaml

# Apply the RBG with planner
kubectl apply -f ./output/rbg.yaml
```

Verify the roles are running:

```bash
kubectl get rbg sglang-pd-inference -o jsonpath='{.status.roleStatuses}'
```

## Step 3: Monitor the Planner

### Logs

```bash
kubectl logs -l role=planner -f
```

Expected log output pattern:
```
Workers: prefill=2, decode=3
Observed: num_req=150.00 isl=512.00 osl=128.00
Observed: ttft=180.00ms itl=35.00ms
Predicted: num_req=160.00 isl=520.00 osl=130.00
Prefill: 462.22(tok/s) / 4000.00(engine_cap) = 1(replicas)
Decode: 115.56(tok/s) / 500.00(engine_cap) = 1(replicas)
Target replicas: prefill=1, decode=1
```

### Prometheus Metrics

If `PLANNER_PROMETHEUS_PORT` is set (default: 9091), the planner exposes:

| Metric | Description |
|--------|-------------|
| `rbg_planner_num_prefill_workers` | Current prefill replica count |
| `rbg_planner_num_decode_workers` | Current decode replica count |
| `rbg_planner_observed_ttft_ms` | Last observed TTFT |
| `rbg_planner_observed_itl_ms` | Last observed ITL |
| `rbg_planner_observed_request_rate` | Observed request rate (req/s) |
| `rbg_planner_predicted_request_rate` | Predicted request rate |
| `rbg_planner_predicted_num_prefill` | Predicted prefill replicas |
| `rbg_planner_predicted_num_decode` | Predicted decode replicas |
| `rbg_planner_p_correction_factor` | TTFT correction factor |
| `rbg_planner_d_correction_factor` | ITL correction factor |
| `rbg_planner_gpu_hours_total` | Cumulative GPU hours used |

### Scaling Verification

Watch the RBG role replicas change:

```bash
kubectl get rbg sglang-pd-inference -w
```

Or check RBGSA status:

```bash
kubectl get rbgsa -l rbg.workloads.x-k8s.io/name=sglang-pd-inference
```

## Standalone Profiling

Use `inference-ext-cli profile` to run profiling independently (without generating
the final RBG YAML). This is useful for collecting profiling data to share or
iterate on before deploying.

```bash
inference-ext-cli profile \
  --engine sglang \
  --model "Qwen/Qwen3-0.6B" \
  --engine-image "lmsysorg/sglang:latest" \
  --namespace profiling \
  --min-gpus 1 \
  --max-gpus 4 \
  --isl 3000 \
  --osl 500 \
  --ttft-sla 200 \
  --itl-sla 20 \
  --max-context-length 32768 \
  --prefill-interpolation-granularity 16 \
  --decode-interpolation-granularity 6 \
  --output-dir ./profiling-results \
  --configmap-name profiling-data \
  --configmap-namespace default
```

The profiling pipeline:

1. **Phase 1 - Parallelization Sweep**: For each GPU count (powers of 2 from
   `--min-gpus` to `--max-gpus`), deploys a single-role RBG with each candidate
   parallelization mapping and measures TTFT (prefill) and ITL (decode).

2. **Config Selection**: Selects the parallelization that satisfies the SLA
   targets with highest throughput/GPU.

3. **Phase 2 - Interpolation Sweep**: With the selected config:
   - Prefill: sweeps ISL from 100 to max context length
   - Decode: 2D sweep of ISL x concurrency

4. **Output**: Generates `profiling-results/profiling-configmap.yaml` and raw JSON files.

### Supported Engines

- `sglang` - SGLang inference engine
- `vllm` - vLLM inference engine

### MoE Model Support

For Mixture-of-Experts models (DeepSeek V3, Qwen3-MoE), the profiler automatically
sweeps TEP (Tensor-Expert Parallel) and DEP (Data-Expert Parallel) in addition to
standard TP, selecting the best strategy for each phase.

## Profiling Data Format

**prefill_raw_data.json:**
```json
{
  "prefill_isl": [128, 256, 512, 1024, 2048],
  "prefill_ttft": [0.008, 0.015, 0.030, 0.060, 0.120],
  "prefill_thpt_per_gpu": [6000, 5000, 4000, 3000, 2000]
}
```

**decode_raw_data.json:**
```json
{
  "x_kv_usage": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
  "y_context_length": [256, 512, 1024, 2048, 4096],
  "z_itl": [0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.035],
  "z_thpt_per_gpu": [1200, 1100, 1000, 900, 800, 700, 500],
  "max_kv_tokens": 32768
}
```

## Configuration Reference

All planner configuration is via environment variables (set in the container env):

| Variable | Default | Description |
|----------|---------|-------------|
| `RBG_NAME` | (required) | Name of the RoleBasedGroup |
| `RBG_NAMESPACE` | `default` | Kubernetes namespace |
| `PREFILL_ROLE_NAME` | `prefill` | Prefill role name in the RBG |
| `DECODE_ROLE_NAME` | `decode` | Decode role name in the RBG |
| `PROMETHEUS_ENDPOINT` | `http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090` | Prometheus URL |
| `METRIC_SOURCE` | `sglang` | Metric source: `sglang`, `vllm`, or `patio` |
| `MODEL_NAME` | (empty) | Model name for Prometheus label filtering |
| `ADJUSTMENT_INTERVAL` | `180` | Seconds between scaling decisions |
| `MAX_GPU_BUDGET` | `8` | Maximum total GPUs (prefill + decode) |
| `MIN_REPLICAS` | `1` | Minimum replicas per role |
| `PREFILL_ENGINE_NUM_GPU` | `1` | GPUs per prefill engine instance |
| `DECODE_ENGINE_NUM_GPU` | `1` | GPUs per decode engine instance |
| `TTFT_SLA` | `500.0` | Target TTFT in milliseconds |
| `ITL_SLA` | `50.0` | Target ITL in milliseconds |
| `LOAD_PREDICTOR` | `arima` | Predictor: `constant`, `arima`, or `prophet` |
| `LOAD_PREDICTION_WINDOW_SIZE` | `50` | Data points in predictor window |
| `NO_CORRECTION` | `false` | Disable SLA correction factors |
| `PROFILE_RESULTS_DIR` | `/etc/rbg-planner/profiling` | Profiling data mount path |
| `PLANNER_PROMETHEUS_PORT` | `0` | Planner metrics port (0=disabled) |
| `NO_OPERATION` | `false` | Dry-run mode (observe only, no scaling) |

## Troubleshooting

### Planner not scaling

1. Check the planner logs for "Metrics contain None/NaN" - this means no traffic
2. Verify Prometheus is reachable from the planner pod
3. Confirm the model_name matches what the engine reports in metrics
4. Check `kubectl get rbg <name> -o jsonpath='{.status.conditions}'` for Ready state

### RBGSA not found (falling back to RBG patch)

This is normal if RoleBasedGroupScalingAdapters are not created for your roles.
The planner will fall back to directly patching RBG `spec.roles[].replicas`.
To use RBGSA (preferred for HPA compatibility), ensure your RBG roles have
`scalingAdapter` configured.

### High correction factors

If `p_correction_factor` >> 1.0, the observed TTFT is much higher than profiled,
indicating resource contention or queue buildup beyond what the profiling captured.
Consider:
- Re-running profiling under more realistic conditions
- Increasing `MIN_REPLICAS` to maintain a baseline capacity
- Reducing `ADJUSTMENT_INTERVAL` for faster reactions

## Architecture

```
                    +-----------------+
                    |   Prometheus    |
                    | (inference      |
                    |  metrics)       |
                    +--------+--------+
                             |
                             | query metrics
                             v
+----------+        +--------+--------+        +------------------+
|  Profiling|------->|   RBG Planner   |------->|  RBG Controller  |
|  ConfigMap|        |                 |        |                  |
+----------+        | 1. Observe      |        | scales pods via  |
                    | 2. Predict      |  RBGSA | StatefulSets/    |
                    | 3. Compute      |  patch | Deployments      |
                    | 4. Scale        |        |                  |
                    +-----------------+        +------------------+
```

## Future: Unified Metrics via Patio EngineRuntime

In Phase 2, the planner can be configured with `METRIC_SOURCE=patio` to use
unified `patio:*` metrics from the EngineRuntime sidecar. This makes the planner
engine-agnostic (works with both SGLang and vLLM without reconfiguration).

To enable this:
1. Deploy ClusterEngineRuntimeProfile with patio sidecar
2. Ensure ServiceMonitor scrapes patio on port 9091
3. Set `METRIC_SOURCE=patio` in the planner environment
