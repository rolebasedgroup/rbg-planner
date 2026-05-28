"""Configuration for RBG Planner via environment variables and CLI args."""

import os


class PlannerConfig:
    """Source of truth for planner configuration defaults."""

    # Kubernetes
    rbg_name: str = os.environ.get("RBG_NAME", "")
    rbg_namespace: str = os.environ.get("RBG_NAMESPACE", "default")
    prefill_role_name: str = os.environ.get("PREFILL_ROLE_NAME", "prefill")
    decode_role_name: str = os.environ.get("DECODE_ROLE_NAME", "decode")

    # Prometheus
    prometheus_endpoint: str = os.environ.get(
        "PROMETHEUS_ENDPOINT",
        "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
    )
    metric_source: str = os.environ.get("METRIC_SOURCE", "sglang")  # sglang | vllm | patio
    model_name: str = os.environ.get("MODEL_NAME", "")

    # Planner parameters
    adjustment_interval: int = int(os.environ.get("ADJUSTMENT_INTERVAL", "180"))
    max_gpu_budget: int = int(os.environ.get("MAX_GPU_BUDGET", "8"))
    min_replicas: int = int(os.environ.get("MIN_REPLICAS", "1"))
    prefill_engine_num_gpu: int = int(os.environ.get("PREFILL_ENGINE_NUM_GPU", "1"))
    decode_engine_num_gpu: int = int(os.environ.get("DECODE_ENGINE_NUM_GPU", "1"))
    ttft_sla: float = float(os.environ.get("TTFT_SLA", "500.0"))  # milliseconds
    itl_sla: float = float(os.environ.get("ITL_SLA", "50.0"))  # milliseconds
    load_predictor: str = os.environ.get("LOAD_PREDICTOR", "arima")
    load_prediction_window_size: int = int(os.environ.get("LOAD_PREDICTION_WINDOW_SIZE", "50"))
    no_correction: bool = os.environ.get("NO_CORRECTION", "false").lower() == "true"

    # Profiling
    profile_results_dir: str = os.environ.get("PROFILE_RESULTS_DIR", "/etc/rbg-planner/profiling")

    # Planner metrics exposition
    planner_prometheus_port: int = int(os.environ.get("PLANNER_PROMETHEUS_PORT", "0"))

    # Operation mode
    no_operation: bool = os.environ.get("NO_OPERATION", "false").lower() == "true"
