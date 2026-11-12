# End-to-end walkthrough: autoscaling SGLang PD-disaggregated inference

This guide walks you through deploying a complete, working setup: an SGLang
prefill/decode disaggregated workload on a RoleBasedGroup (RBG), metrics
scraping into Prometheus, and an `AutoScaler` that profiles the engines and
scales them to meet TTFT/ITL SLA targets. All manifests referenced here are
sanitized examples in [`docs/examples/`](examples/); replace every
`<placeholder>` value before applying.

Estimated time: 30-60 minutes, plus image pull and model load times.

## Before you begin

Make sure you have the following:

- A Kubernetes cluster with the [RBG controller](https://github.com/rolebasedgroup/rbg)
  installed, and at least one GPU node per engine replica you plan to run.
- `kubectl` configured to access the cluster.
- A Prometheus stack that discovers scrape targets through
  `ServiceMonitor` custom resources (for example kube-prometheus-stack, or a
  managed Prometheus service that honors ServiceMonitors). If your Prometheus
  query endpoint requires authentication, have the credentials available.
- The three RBG Planner images pushed to a registry the cluster can pull
  from, and an image pull secret (referred to as `regcred` in the examples).
  See [Building the images](../README.md#docker-images).
- The model weights and tokenizer available on a PersistentVolumeClaim
  (referred to as `model-weights` in the examples), mounted so that the model
  directory resolves to `/models/Qwen/Qwen3.5-4B` inside the engine
  containers. Adjust the paths if your layout differs.

## Step 1: Create the Prometheus credentials Secret

Skip this step if your Prometheus query endpoint needs no authentication.

The planner reads the query endpoint with the credentials from a Secret in
the AutoScaler namespace. Create it with keys `username` and `password`:

```bash
kubectl create secret generic prometheus-auth \
  --from-literal=username=<username> \
  --from-literal=password=<password>
```

**Note:** Token-based authentication is also supported; see
`metrics.authentication.bearer` in the [CRD reference](../README.md#autoscaler-spec).

## Step 2: Deploy the RoleBasedGroup

Review [`rbg-sglang-pd.yaml`](examples/rbg-sglang-pd.yaml) and replace the
`<registry>` placeholders with your engine image location. Three settings in
this manifest matter specifically for the autoscaler:

- **`--enable-metrics`** on the prefill and decode engines. SGLang serves
  `/metrics` on the main server port, so the container port is named
  `metrics` and points at 8000. Without this flag there is nothing to
  scrape.
- **`scalingAdapter.enable: true`** on the prefill and decode roles. This
  makes the RBG controller create one `RoleBasedGroupScalingAdapter` per
  role, which is the path the planner uses to apply scaling decisions.
- The value of **`--model-path`** (`/models/Qwen/Qwen3.5-4B`). The engines
  export this exact string as the `model_name` Prometheus label, and the
  AutoScaler's `modelName` must match it character for character.

Apply it and wait for the group to become ready:

```bash
kubectl apply -f docs/examples/rbg-sglang-pd.yaml
kubectl get rbg qwen-pd -w
```

## Step 3: Expose engine metrics to Prometheus

The RBG controller creates a headless Service per role, but those Services
expose **no ports**, so Prometheus Operator cannot use them as scrape
targets. To make the engine metrics discoverable, deploy the manifest in
[`metrics-scrape.yaml`](examples/metrics-scrape.yaml), which does two things:

1. Creates a headless Service per engine role with a named `metrics` port
   selecting the engine pods.
2. Creates a `ServiceMonitor` that selects those Services and scrapes
   `/metrics` every 15 seconds.

```bash
kubectl apply -f docs/examples/metrics-scrape.yaml
```

Verify that the engines are being scraped and that the model label matches
the value of `--model-path`. Query your Prometheus for:

```promql
up{job=~".*qwen-pd.*"}
sglang:e2e_scheduling_latency{model_name="/models/Qwen/Qwen3.5-4B"}
```

The exact metric names depend on your engine version; what matters is that
some engine metrics appear, labeled with
`model_name="/models/Qwen/Qwen3.5-4B"`. If the label value differs from
`--model-path`, revisit Step 2.

## Step 4: Install the RBG Planner operator

Install the Helm chart as described in the
[Quick Start](../README.md#install-with-helm). If you did not set a
cluster-wide Prometheus endpoint in the chart values, you can set a per-CR
endpoint in Step 5 instead:

```bash
helm install rbg-planner ./charts/rbg-planner -n rbg-system --create-namespace \
  --set image.repository=<your-registry>/rbg-planner-operator \
  --set image.tag=<tag> \
  --set planner.image=<your-registry>/rbg-planner:<tag> \
  --set profiler.image=<your-registry>/rbg-profiler:<tag>
```

## Step 5: Create the AutoScaler

Review [`autoscaler-sglang-pd.yaml`](examples/autoscaler-sglang-pd.yaml),
replace the placeholders, and check two settings that are easy to get wrong:

- **`backend: sglang-pd-native`**. For PD-disaggregated SGLang, the profiling
  bench must speak the disaggregation bootstrap protocol. The plain
  `sglang-oai` backend sends ordinary OpenAI requests, which PD engines
  reject -- profiling then completes with empty data and the planner cannot
  start. For non-disaggregated SGLang, use `sglang-oai` (or `sglang-oai-chat`).
- **`modelName: "/models/Qwen/Qwen3.5-4B"`** must equal the engines'
  `--model-path` value exactly, including the leading path. The planner
  builds its Prometheus label filter as `model_name="<modelName>"`; a
  mismatch makes every query return nothing and the planner observes zero
  load.

Apply the AutoScaler:

```bash
kubectl apply -f docs/examples/autoscaler-sglang-pd.yaml
```

The operator now creates the per-CR RBAC (ServiceAccounts, Roles,
RoleBindings), runs the profiling Job, and deploys the planner. The
AutoScaler name must match the RBG name (`qwen-pd` here).

## Step 6: Verify the deployment

Watch the AutoScaler progress through its phases:

```bash
kubectl get as qwen-pd -w
```

Expected phases: `Pending` -> `Initializing` (profiling Job running) ->
`Ready`. Profiling a large model takes a while; the Job streams benchmark
results into the ConfigMap `qwen-pd-profiling`.

When the phase is `Ready`, check that the planner observes live metrics:

```bash
kubectl logs deploy/qwen-pd-planner -f
```

You should see periodic lines with non-zero `num_req`:

```
INFO  Workers: prefill=1, decode=1
INFO  Observed: num_req=18.50 isl=2048.00 osl=256.00
```

If `num_req` stays `0.00` under load, the model label does not match -- see
[Troubleshooting](#troubleshooting).

You can also confirm the per-CR RBAC is scoped as expected using kubectl
impersonation:

```bash
NS=default
AS=qwen-pd
kubectl auth can-i --as=system:serviceaccount:${NS}:${AS}-planner \
  get rolebasedgroups.workloads.x-k8s.io -n ${NS}                 # yes
kubectl auth can-i --as=system:serviceaccount:${NS}:${AS}-planner \
  patch rolebasedgroupscalingadapters.workloads.x-k8s.io/scale \
  -n ${NS}                                                          # yes
kubectl auth can-i --as=system:serviceaccount:${NS}:${AS}-planner \
  get configmaps -n ${NS}                                           # no
```

## Step 7: Generate load and watch scaling

Deploy the load generator pod, copy the script into it, and run it. The
script sends streaming chat completions through the router role, exercising
both prefill and decode:

```bash
kubectl apply -f docs/examples/loadgen.yaml
kubectl cp docs/examples/loadgen.py qwen-pd-loadgen:/tmp/loadgen.py
kubectl exec qwen-pd-loadgen -- python3 /tmp/loadgen.py
```

Tune the load with the `CONCURRENCY` and `DURATION` environment variables.
While it runs, watch the planner compute and apply new replica counts:

```bash
kubectl logs deploy/qwen-pd-planner -f
kubectl get rbg qwen-pd -w
```

**Note:** The planner skips scaling decisions while the RBG reports a
non-`Ready` condition, including when a new replica cannot be scheduled. If
scaling appears stuck after a scale-up, check
`kubectl get pods -l app=qwen-pd` for pending engine pods.

## Step 8 (optional): Verify the scale subresource scope

[`verify-scale.yaml`](examples/verify-scale.yaml) runs a one-shot pod under
the planner ServiceAccount. It patches the `scale` subresource of the
prefill and decode adapters (expected to succeed) and of the router adapter
(expected to be denied with 403):

```bash
kubectl apply -f docs/examples/verify-scale.yaml
kubectl logs verify-scale -f
```

**Caution:** The test pod patches the scale subresource to the current
replica count (1). Replace the replica values with your current counts if
they differ, or delete the pod quickly after it completes.

## Re-profiling a model

The operator skips profiling when the ConfigMap `<as-name>-profiling`
already exists. To regenerate profiling data (for example after changing
the engine version or SLA-relevant settings), delete the ConfigMap and the
planner Deployment; the operator re-runs the profiling Job and redeploys
the planner:

```bash
kubectl delete configmap qwen-pd-profiling
kubectl delete deploy qwen-pd-planner
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Profiling Job finishes quickly and the ConfigMap contains empty arrays | `backend` does not match the deployment: `sglang-oai` against PD-disaggregated engines | Set `backend: sglang-pd-native` |
| Planner logs show `Observed: num_req=0.00` while load is running | `modelName` does not match the engines' `model_name` label | Set `modelName` to the exact `--model-path` value |
| No engine metrics in Prometheus | Engines missing `--enable-metrics`, or no ServiceMonitor target | Check Step 2 and Step 3; query `up{job=~".*qwen-pd.*"}` |
| Planner fails to start with `401` / `403` against Prometheus | Wrong or missing credentials for the query endpoint | Recreate the `prometheus-auth` Secret; confirm `metrics.authentication` keys |
| Profiling or planner pods report `ImagePullBackOff` | Registry requires credentials not attached to the pod | Set `imagePullSecrets` in the AutoScaler spec |
| Planner never scales, logs `RBG ... is not ready, skipping this scaling` | A role replica cannot be scheduled (for example GPU shortage) | Check pending pods; the planner also blocks scale-down while the RBG is not ready |
| Scaling decisions appear wrong after changing SLA or engine version | Profiling data is stale | [Re-profile the model](#re-profiling-a-model) |

## What's next

- [Metrics adapter guide](metrics.md): per-slot PromQL overrides and custom
  engine presets.
- [AutoScaler CRD reference](../README.md#crd-reference): full field
  documentation.
- [Operator SOP](sop.md): CLI-driven generation and standalone profiling.
