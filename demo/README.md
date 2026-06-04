# RBG AutoScaler Demo Recording Script

## File List

| File | Description |
|------|-------------|
| `00-rbg.yaml` | RBG object (service-discovery router) |
| `01-router-rbac.yaml` | Router ServiceAccount + Role + RoleBinding |
| `02-autoscaler.yaml` | AutoScaler CR |
| `03-traffic-simulator.yaml` | Traffic simulator ConfigMap + Job (~2000 token prompts) |
| `04-grafana-portforward.sh` | Grafana port-forward script |
| `05-prometheus-grafana.yaml` | Prometheus + Grafana deployments |
| `06-grafana-dashboard.yaml` | Grafana dashboard ConfigMap |

## Prerequisites

- Kubernetes cluster with RBG controller installed
- AutoScaler CRD installed (`make install`)
- AutoScaler controller running (`make run` in a separate terminal)

## Setup (before recording)

```bash
# Create namespace
kubectl create namespace demo

# Deploy prerequisites
kubectl apply -f demo/01-router-rbac.yaml
kubectl apply -f demo/00-rbg.yaml

# Wait for RBG pods to be ready
kubectl wait --for=condition=ready pod -l app=sglang-inference -n demo --timeout=300s

# Deploy Prometheus + Grafana
kubectl apply -f demo/06-grafana-dashboard.yaml
kubectl apply -f demo/05-prometheus-grafana.yaml
```

## Terminal Layout

Open **4 terminals** side by side:

| Terminal | Purpose |
|----------|---------|
| T1 (top-left) | Pod watcher - continuously shows pod changes |
| T2 (top-right) | Planner logs - follows planner log output |
| T3 (bottom-left) | Command terminal - run kubectl commands and traffic simulator |
| T4 (bottom-right) | Grafana port-forward |

---

## Step-by-Step Script

### Scene 1: Show the RBG deployment (T3)

```bash
# Show the RBG object
kubectl get rbg -n demo

# Show pods under the RBG
kubectl get pods -n demo -l app=sglang-inference -o wide
```

### Scene 2: Start pod watcher (T1)

```bash
# This will continuously refresh every 3 seconds
watch -n 3 kubectl get pods -n demo -l app=sglang-inference -o wide
```

### Scene 3: Show the AutoScaler CR (T3)

```bash
# Display the AutoScaler YAML we are about to apply
cat demo/02-autoscaler.yaml
```

### Scene 4: Deploy the AutoScaler CR (T3)

```bash
# Apply the AutoScaler CR
kubectl apply -f demo/02-autoscaler.yaml

# Watch the AutoScaler status
kubectl get autoscaler -n demo -w
```

> Wait and narrate: The AutoScaler transitions through phases:
> - `Pending` -> validates the target RBG exists
> - `Initializing` -> creates and runs a profiling Job
> - `Ready` -> profiling complete, planner Deployment created
>
> Switch to T1 to show the profiling Job pod appear and complete,
> then the planner pod appear.

```bash
# After status shows Ready, Ctrl+C the watch
# Show the created resources
kubectl get autoscaler -n demo

# Show the profiling ConfigMap
kubectl get configmap sglang-pd-inference-profiling -n demo

# Show the planner Deployment
kubectl get deployment sglang-pd-inference-planner -n demo
```

### Scene 5: Follow planner logs (T2)

```bash
# Get the planner pod name and follow logs
kubectl logs -f -l app.kubernetes.io/name=rbg-planner -n demo
```

> Narrate: The planner runs every 60 seconds, observing traffic metrics
> from Prometheus and calculating target replica counts. Currently shows
> zero traffic and maintains minReplicas=1 for both prefill and decode.

### Scene 6: Port-forward Grafana (T4)

```bash
bash demo/04-grafana-portforward.sh
```

> Open browser at http://localhost:3000
> Navigate to: Dashboards -> RBG Planner Dashboard
> Select `rbg_name = sglang-pd-inference` from the dropdown
> Show the dashboard panels - currently flat lines at 1 worker each

### Scene 7: Start traffic simulation (T3)

```bash
# Apply the traffic simulator (ConfigMap + Job)
kubectl apply -f demo/03-traffic-simulator.yaml

# Follow the simulator output
kubectl logs -f job/traffic-simulator -n demo
```

> Narrate the phases as they happen:
>
> **Phase 1 - Baseline (3 min, concurrency=2):**
> Light traffic with long prompts (~2000 input tokens).
> Planner sees moderate load, keeps 1 prefill + 1 decode.
> Note the baseline request latency.
>
> **Phase 2 - High Load (5 min, concurrency=15):**
> Heavy traffic overwhelms single prefill/decode workers.
> Switch to T2: planner logs show increased throughput and scaling decision.
> Switch to T1: new prefill and decode pods appearing (scale-up to 2-4 replicas).
> Switch to Grafana: worker count increasing, TTFT spiking then dropping.
>
> **Phase 3 - Scaled Sustained (5 min, concurrency=15):**
> With more workers online, the router distributes load.
> TTFT should visibly decrease compared to Phase 2.
> Switch to Grafana: observe TTFT panel showing improvement.
>
> **Phase 4 - Cool-down (4 min, concurrency=1):**
> Minimal traffic. Planner scales back to minReplicas=1.
> Switch to T1: pods terminating, back to 1 prefill + 1 decode.
> Switch to Grafana: worker count dropping, TTFT returning to baseline.
>
> Wait for "Done!" message in the simulator output.

### Scene 8: Final state (T3)

```bash
# Show final pod state - back to 1 prefill + 1 decode
kubectl get pods -n demo -l app=sglang-inference -o wide

# Show AutoScaler status
kubectl get autoscaler -n demo -o yaml | grep -A 10 'status:'
```

> End of demo.

---

## Cleanup

```bash
# Delete traffic simulator
kubectl delete -f demo/03-traffic-simulator.yaml

# Delete AutoScaler (will clean up planner, RBAC, etc.)
kubectl delete -f demo/02-autoscaler.yaml

# Delete Prometheus + Grafana
kubectl delete -f demo/05-prometheus-grafana.yaml
kubectl delete -f demo/06-grafana-dashboard.yaml

# Delete RBG
kubectl delete -f demo/00-rbg.yaml

# Delete router RBAC
kubectl delete -f demo/01-router-rbac.yaml

# Delete namespace
kubectl delete namespace demo
```

## Timing

| Phase | Duration |
|-------|----------|
| Scene 1-3: Setup and show | ~2 min |
| Scene 4: Deploy AutoScaler | ~2-3 min (profiling) |
| Scene 5-6: Logs + Grafana | ~1 min |
| Scene 7: Traffic simulation | ~17 min |
| Scene 8: Wrap-up | ~1 min |
| **Total** | **~23 min** |
