# RBG Planner

An engine-agnostic, framework-independent SLA-driven autoscaler for large model inference on Kubernetes.

RBG Planner works with **any inference engine** (SGLang, vLLM, NVIDIA Dynamo, ...) and **any orchestration framework** (RoleBasedGroup, custom controllers, ...) through its pluggable metrics adapter and connector architecture. It observes real-time latency metrics, predicts future load, and makes proactive scaling decisions to meet TTFT and ITL SLA targets -- all without coupling to a specific inference runtime.

### Key Features

- **Engine-agnostic**: Pluggable Metrics Adapter supports SGLang, vLLM, Dynamo, and any Prometheus-compatible metric source
- **Framework-independent**: Connector interface decouples scaling logic from the orchestration layer
- **Extensible scaling engine**: The `Implementation` union in the CRD supports multiple scaling engines -- `DynamoPlanner` ships built-in, and new engines can be added without modifying the core operator
- **SLA-driven**: Scales based on TTFT/ITL latency targets, not just resource utilization
- **Proactive**: ARIMA-based load prediction scales ahead of demand
- **Self-profiling**: Automatic benchmarking generates per-model performance curves for optimal replica calculation
- **PD-disaggregated**: First-class support for Prefill/Decode disaggregated inference patterns

### How It Works

Users create an `AutoScaler` custom resource targeting a [RoleBasedGroup](https://github.com/rolebasedgroup). The operator automatically runs SLA profiling, deploys the planner engine, and continuously scales prefill/decode roles to meet latency targets.

## Acknowledgements

The core planning algorithm, SLA profiling methodology, and performance interpolation approach in this project are derived from [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo). We are deeply grateful to the Dynamo team for their pioneering work on SLA-driven autoscaling for disaggregated inference. This project adapts and re-implements those ideas to work natively with the Kubernetes RoleBasedGroup API.

## Architecture

![Architecture](docs/images/architecture.png)

### Components

| Component | Image | Description |
|-----------|-------|-------------|
| **AutoScaler Operator** | `rbg-planner-operator` | Go CRD controller. Watches `AutoScaler` CRs, manages the lifecycle: validates the target RBG, creates RBAC, runs profiling Jobs, deploys the planner engine, and updates status. |
| **Planner Engine** | `rbg-planner` | Python SLA-based autoscaler (Dynamo planner algorithm). Observes metrics from Prometheus, predicts load, computes replica requirements from profiling data, and applies scaling decisions via RBGSA. |
| **Profiler** | `rbg-profiler` | Python profiling tool. Runs benchmarks against inference engines and generates profiling data (prefill/decode throughput curves) stored in a ConfigMap. |

### How It Works

1. User creates an `AutoScaler` CR with the same name as the target RBG
2. Operator validates the RBG exists and reads GPU-per-engine from the RBG's resource requests
3. Operator creates a ServiceAccount, ClusterRole, and ClusterRoleBinding for the planner
4. Operator runs a profiling Job to generate performance data (or uses existing ConfigMap)
5. Operator deploys the planner engine as a Deployment with profiling data mounted
6. Planner engine runs in a loop: observe metrics -> predict load -> compute replicas -> scale via RBGSA
7. Operator periodically updates status with current replica counts from the RBG

### Metrics Adapter

The planner uses a config-driven **Metrics Adapter** module to query Prometheus. Each inference engine exposes metrics in a different format -- the adapter normalizes them into a unified interface so the scaling logic is engine-agnostic.

Queries are resolved **per logical metric slot** (`ttft`, `itl`, `requestsTotal`, `requestDuration`, `isl`, `osl`). Built-in engine presets ship complete PromQL templates per slot; `metrics.queries` in the CRD overrides individual slots, KEDA-style.

| Engine | ISL/OSL query | Model label |
|--------|---------------|-------------|
| SGLang | counter ratio | `model_name` |
| vLLM | counter ratio | `model_name` |
| NVIDIA Dynamo | histogram | `model` |
| Patio | counter ratio | `model_name` |

Configured via `metrics.engine` in the AutoScaler CR. See [docs/metrics.md](docs/metrics.md) for the full query mapping guide -- slot semantics, placeholder expansion, per-slot overrides, and custom engines.

### Extensible Scaling Engine

The CRD `spec.implementation` is a discriminated union -- each field represents a distinct scaling engine:

```yaml
spec:
  implementation:
    dynamoPlanner: { ... }   # Built-in: SLA-driven planner derived from NVIDIA Dynamo
    # YourScaler: { ... }    # Add your own scaling engine here
```

`DynamoPlanner` is the built-in engine that ships with this project. To add a new scaling engine:

1. Define a new config struct in `api/v1alpha1/roleautoscaler_types.go` and add it to the `Implementation` union
2. Implement the engine's reconcile logic in the operator (profiling, deployment, scaling)
3. The operator dispatches to the correct engine based on which field is set

This design allows the same operator and CRD to support fundamentally different scaling strategies (e.g., rule-based, ML-based, external controller) without modifying the core framework.

### Lifecycle

![Lifecycle](docs/images/lifecycle.png)

### Planner Loop

![Planner Loop](docs/images/planner-loop.png)

## Quick Start

### Prerequisites

- Kubernetes cluster with [RBG controller](https://github.com/rolebasedgroup/rbg) installed
- `kubectl` configured to access the cluster
- Prometheus monitoring stack deployed

### Install with Helm

```bash
helm install rbg-planner ./charts/rbg-planner -n rbg-system --create-namespace
```

Override defaults as needed:

```bash
helm install rbg-planner ./charts/rbg-planner -n rbg-system --create-namespace \
  --set image.repository=<your-registry>/rbg-planner-operator \
  --set image.tag=v0.1.0 \
  --set planner.image=<your-registry>/rbg-planner:v0.1.0 \
  --set profiler.image=<your-registry>/rbg-profiler:v0.1.0 \
  --set prometheus.endpoint=http://prometheus.monitoring.svc:9090
```

Key configurable values (`charts/rbg-planner/values.yaml`):

| Value | Default | Description |
|-------|---------|-------------|
| `image.repository` | `ghcr.io/rolebasedgroup/rbg-planner-operator` | Operator image |
| `planner.image` | `ghcr.io/rolebasedgroup/rbg-planner:latest` | Planner image (deployed per CR) |
| `profiler.image` | `ghcr.io/rolebasedgroup/rbg-profiler:latest` | Profiler image (Job per CR) |
| `prometheus.endpoint` | `http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090` | Prometheus URL |
| `leaderElection.enabled` | `true` | Enable leader election for HA |

### Create an AutoScaler

Ensure you have a RoleBasedGroup already deployed (e.g., `sglang-pd-inference`), then create an `AutoScaler` with the **same name**:

```yaml
apiVersion: inference-extension.rolebasedgroup.io/v1alpha1
kind: AutoScaler
metadata:
  name: sglang-pd-inference    # must match the RBG name
  namespace: inference
spec:
  scalingInterval: 180    # seconds between scaling decisions
  backend: sglang-oai     # inference engine backend (profiling bench + metrics)

  # Scaling pattern: pdDisaggregated (supported) | unified (reserved, not yet implemented)
  pattern:
    pdDisaggregated:
      prefill:
        roleName: prefill
        maxReplicas: 10
        minReplicas: 1
      decode:
        roleName: decode
        maxReplicas: 10
        minReplicas: 1

  # Scaling engine
  implementation:
    dynamoPlanner:
      modelName: "Qwen/Qwen3-0.6B"

      # SLA targets
      sla:
        ttft: 200.0        # target TTFT in milliseconds
        itl: 20.0          # target ITL in milliseconds

      # Load prediction
      loadPredictor: arima # arima | constant | prophet
      predictionWindow: 50
      noCorrection: false  # disable SLA correction factors
      dryRun: false        # observe only, no actual scaling

      # Private registry credentials, applied to both the profiling Job
      # and the planner Deployment
      imagePullSecrets:
        - name: regcred

      profiling:
        image: "ghcr.io/rolebasedgroup/rbg-profiler:latest"
        # Prompt input form for bench requests: auto | text | token_ids
        inputFormat: "auto"
        # Optional: mount a PVC with pre-downloaded model weights and
        # tokenizer files. When omitted, the profiler downloads the
        # tokenizer from HuggingFace using modelName.
        modelSource:
          pvcName: "model-weights"
          modelPath: "deepseek-r1"
          mountPath: "/opt/model"

      # Metrics configuration
      metrics:
        engine: sglang
        port: 9091
        # Optional: per-CR override of the operator-level Prometheus query endpoint
        # endpoint: "https://prometheus.example.com"
        # Optional: credentials for the query endpoint (one mechanism only)
        # authentication:
        #   basicAuth:
        #     username: { name: prometheus-auth, key: username }
        #     password: { name: prometheus-auth, key: password }
        #   # or bearer token
        #   bearer:
        #     token: { name: prometheus-auth, key: token }
        # Optional: per-slot PromQL overrides. Slots: ttft, itl, requestsTotal,
        # requestDuration, isl, osl. Omitted slots use the engine preset.
        # queries:
        #   ttft: 'sum(increase(my_ttft_seconds_sum{modelFilter}[{interval}])) / sum(increase(my_ttft_seconds_count{modelFilter}[{interval}]))'
```

```bash
kubectl apply -f config/samples/roleautoscaler_full.yaml
```

### Step 4: Observe

```bash
# Watch the AutoScaler status
kubectl get as -w

# Expected output:
# NAME                  PHASE   PREFILL   DECODE   AGE
# sglang-pd-inference   Ready   1         1        5m

# Watch planner logs
kubectl logs -l app.kubernetes.io/name=rbg-planner -f
```

When load increases, the planner scales up:

```
INFO  Workers: prefill=1, decode=1
INFO  Observed: num_req=1200.00 isl=2048.00 osl=256.00
INFO  Predicted: num_req=1350.00 isl=2100.00 osl=260.00
INFO  Prefill: 15750.00(tok/s) / 2800.00(engine_cap) = 6(replicas)
INFO  Decode: 1950.00(tok/s) / 500.00(engine_cap) = 4(replicas)
INFO  Target replicas: prefill=4, decode=4
INFO  Scaling role prefill: 1 -> 4
INFO  Scaling role decode: 1 -> 4
```

## CRD Reference

### AutoScaler Spec

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `spec.scalingInterval` | int | `180` | Seconds between scaling decisions |
| `spec.backend` | string | `sglang-oai` | Inference engine backend used by both profiling (bench) and metrics collection, e.g. `sglang-oai`, `vllm`, `vllm-chat` |
| `spec.pattern.pdDisaggregated.prefill.roleName` | string | (required) | Prefill role name in the RBG |
| `spec.pattern.pdDisaggregated.prefill.maxReplicas` | int | (required) | Maximum prefill replicas |
| `spec.pattern.pdDisaggregated.prefill.minReplicas` | int | `1` | Minimum prefill replicas |
| `spec.pattern.pdDisaggregated.decode.roleName` | string | (required) | Decode role name in the RBG |
| `spec.pattern.pdDisaggregated.decode.maxReplicas` | int | (required) | Maximum decode replicas |
| `spec.pattern.pdDisaggregated.decode.minReplicas` | int | `1` | Minimum decode replicas |
| `spec.implementation.dynamoPlanner.modelName` | string | (required) | Model name for Prometheus label filtering and tokenizer resolution |
| `spec.implementation.dynamoPlanner.sla.ttft` | float | `500` | Target TTFT SLA (ms) |
| `spec.implementation.dynamoPlanner.sla.itl` | float | `50` | Target ITL SLA (ms) |
| `spec.implementation.dynamoPlanner.loadPredictor` | string | `arima` | Load predictor: `arima`, `constant`, `prophet` |
| `spec.implementation.dynamoPlanner.predictionWindow` | int | `50` | Data points in predictor window |
| `spec.implementation.dynamoPlanner.noCorrection` | bool | `false` | Disable SLA correction factors |
| `spec.implementation.dynamoPlanner.dryRun` | bool | `false` | Observe only, no actual scaling |
| `spec.implementation.dynamoPlanner.imagePullSecrets` | list | (empty) | Secrets for pulling profiling/planner images from private registries |
| `spec.implementation.dynamoPlanner.profiling.inputFormat` | string | `auto` | Prompt input form for bench requests: `auto`, `text`, `token_ids` |
| `spec.implementation.dynamoPlanner.profiling.image` | string | (hardcoded default) | Profiling tool container image |
| `spec.implementation.dynamoPlanner.profiling.modelSource.pvcName` | string | (empty) | PVC containing model weights and tokenizer files (same namespace as the AutoScaler) |
| `spec.implementation.dynamoPlanner.profiling.modelSource.modelPath` | string | (empty) | Path to the model/tokenizer directory within the PVC |
| `spec.implementation.dynamoPlanner.profiling.modelSource.mountPath` | string | `/opt/model` | Mount path for the PVC inside the profiler container |
| `spec.implementation.dynamoPlanner.metrics.engine` | string | `sglang` | Metrics engine preset: `sglang`, `vllm`, `dynamo`, `patio` |
| `spec.implementation.dynamoPlanner.metrics.port` | int | `9091` | Planner Prometheus metrics exposition port |
| `spec.implementation.dynamoPlanner.metrics.endpoint` | string | (operator-level Helm value) | Per-CR override of the Prometheus query endpoint |
| `spec.implementation.dynamoPlanner.metrics.queries` | map | (engine preset) | Per-slot PromQL overrides. Keys: `ttft`, `itl`, `requestsTotal`, `requestDuration`, `isl`, `osl`. Values may use `{modelFilter}` and `{interval}` placeholders; a custom engine must provide all six |
| `spec.implementation.dynamoPlanner.metrics.authentication.basicAuth` | object | (empty) | HTTP Basic credentials for the query endpoint (SecretKeySelector pair `username`/`password`) |
| `spec.implementation.dynamoPlanner.metrics.authentication.bearer` | object | (empty) | Bearer token for the query endpoint (SecretKeySelector `token`); mutually exclusive with `basicAuth` |

### AutoScaler Status

| Field | Description |
|-------|-------------|
| `status.phase` | Current phase: `Pending`, `Initializing`, `Ready`, `Failed` |
| `status.conditions` | Standard Kubernetes conditions |
| `status.prefillReplicas` | Current prefill replica count (from RBG status) |
| `status.decodeReplicas` | Current decode replica count (from RBG status) |
| `status.profilingConfigMap` | Name of the profiling ConfigMap |
| `status.plannerDeployment` | Name of the planner Deployment |

### GPU Budget Calculation

The operator reads GPU-per-engine from the target RBG's `resources.requests["nvidia.com/gpu"]` for each role. The maximum GPU budget is automatically computed as:

```
maxGpuBudget = prefill.maxReplicas * prefillGPUs + decode.maxReplicas * decodeGPUs
```

When the planner's computed replicas exceed this budget, it proportionally scales down both roles while respecting `minReplicas`.

## Prometheus Metrics

The planner engine exposes Prometheus metrics at the configured port (default: `9091`). All metrics are labeled with `namespace` and `rbg_name` for multi-planner filtering.

### Worker Counts & GPU Usage

| Metric | Description |
|--------|-------------|
| `rbg_planner_num_prefill_workers{namespace, rbg_name}` | Current prefill replica count |
| `rbg_planner_num_decode_workers{namespace, rbg_name}` | Current decode replica count |
| `rbg_planner_gpu_hours_total{namespace, rbg_name}` | Cumulative GPU hours used |

### Observed Metrics

| Metric | Description |
|--------|-------------|
| `rbg_planner_observed_ttft_ms{namespace, rbg_name}` | Observed average TTFT (ms) |
| `rbg_planner_observed_itl_ms{namespace, rbg_name}` | Observed average ITL (ms) |
| `rbg_planner_observed_request_rate{namespace, rbg_name}` | Observed request rate (req/s) |
| `rbg_planner_observed_request_duration_seconds{namespace, rbg_name}` | Observed average request duration (s) |
| `rbg_planner_observed_isl{namespace, rbg_name}` | Observed average input sequence length |
| `rbg_planner_observed_osl{namespace, rbg_name}` | Observed average output sequence length |

### Predicted Metrics

| Metric | Description |
|--------|-------------|
| `rbg_planner_predicted_request_rate{namespace, rbg_name}` | Predicted request rate (req/s) |
| `rbg_planner_predicted_isl{namespace, rbg_name}` | Predicted input sequence length |
| `rbg_planner_predicted_osl{namespace, rbg_name}` | Predicted output sequence length |
| `rbg_planner_predicted_num_prefill{namespace, rbg_name}` | Predicted prefill replicas needed |
| `rbg_planner_predicted_num_decode{namespace, rbg_name}` | Predicted decode replicas needed |

### Correction Factors

| Metric | Description |
|--------|-------------|
| `rbg_planner_p_correction_factor{namespace, rbg_name}` | Prefill correction factor (observed / expected TTFT) |
| `rbg_planner_d_correction_factor{namespace, rbg_name}` | Decode correction factor (observed / expected ITL) |

## Grafana Dashboard

A pre-built Grafana dashboard is provided at `deploy/grafana-planner-dashboard.yaml` with four sections:

1. **Worker Counts & GPU Usage** -- Stat panels + history timeseries
2. **Observed Metrics** -- TTFT/ITL latency, request rate/duration, ISL/OSL
3. **Predicted Metrics** -- Predicted request rate, sequence lengths, replica counts
4. **Correction Factors** -- Gauge panels with threshold coloring + history

The dashboard includes **Namespace** and **RBG Name** dropdown variables for filtering when running multiple planners.

```bash
kubectl apply -f deploy/grafana-planner-dashboard.yaml
```

## Project Structure

```
rbg-planner/
├── api/v1alpha1/                  # CRD type definitions (AutoScaler CRD)
├── cmd/main.go                    # Operator entrypoint
├── internal/controller/           # Reconciler implementation
├── config/
│   ├── crd/                       # Generated CRD manifests
│   ├── rbac/                      # Generated RBAC manifests
│   └── samples/                   # Example AutoScaler CR
├── python/
│   ├── planner/                   # Planner engine (Python)
│   │   ├── rbg_planner/           # Core planner package
│   │   │   └── metrics/           # Pluggable metrics adapters
│   │   ├── tests/                 # Planner tests
│   │   ├── Dockerfile             # Planner image
│   │   └── pyproject.toml
│   └── profiler/                  # Profiling tool (Python)
│       ├── rbg_profiler/          # Core profiler package
│       ├── Dockerfile             # Profiler image
│       └── pyproject.toml
├── charts/rbg-planner/            # Helm chart for operator deployment
├── docs/                          # Architecture and flow diagrams
├── deploy/                        # Grafana dashboard
├── Dockerfile                     # Operator image (Go)
├── Makefile                       # Build targets
└── go.mod
```

## Development

```bash
# Generate deepcopy methods
make generate

# Generate CRD and RBAC manifests
make manifests

# Build operator binary
make build

# Run operator locally (outside cluster)
make run

# Run Go tests
make test

# Run Python planner tests
make test-python

# Build all Docker images
make docker-build-all
```

### Docker Images

| Image | Dockerfile | Description |
|-------|-----------|-------------|
| `rbg-planner-operator` | `Dockerfile` | Go operator controller |
| `rbg-planner` | `python/planner/Dockerfile` | Python planner engine |
| `rbg-profiler` | `python/profiler/Dockerfile` | Python profiling tool |

```bash
# Build individually
make docker-build           # operator
make docker-build-planner   # planner
make docker-build-profiler  # profiler

# Build all
make docker-build-all
```

## License

Apache-2.0
