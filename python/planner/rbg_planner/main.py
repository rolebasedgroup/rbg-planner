"""Entry point for the RBG Planner."""

import argparse
import asyncio
import logging
import sys

from rbg_planner.config import PlannerConfig
from rbg_planner.planner import Planner

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RBG SLA-based Planner: autoscales Prefill/Decode roles to meet TTFT/ITL targets"
    )

    # Kubernetes
    parser.add_argument("--rbg-name", type=str, default=None, help="RoleBasedGroup name")
    parser.add_argument("--rbg-namespace", type=str, default=None, help="Kubernetes namespace")
    parser.add_argument("--prefill-role-name", type=str, default=None, help="Prefill role name")
    parser.add_argument("--decode-role-name", type=str, default=None, help="Decode role name")

    # Prometheus
    parser.add_argument("--prometheus-endpoint", type=str, default=None, help="Prometheus URL")
    parser.add_argument(
        "--metric-source", type=str, default=None, choices=["sglang", "vllm", "patio"],
        help="Metric source (sglang, vllm, patio)"
    )
    parser.add_argument("--model-name", type=str, default=None, help="Model name for metric filtering")

    # Planner
    parser.add_argument("--adjustment-interval", type=int, default=None, help="Adjustment interval in seconds")
    parser.add_argument("--max-gpu-budget", type=int, default=None, help="Maximum total GPU budget")
    parser.add_argument("--min-replicas", type=int, default=None, help="Minimum replicas per role")
    parser.add_argument("--prefill-engine-num-gpu", type=int, default=None, help="GPUs per prefill engine")
    parser.add_argument("--decode-engine-num-gpu", type=int, default=None, help="GPUs per decode engine")
    parser.add_argument("--ttft-sla", type=float, default=None, help="TTFT SLA target in milliseconds")
    parser.add_argument("--itl-sla", type=float, default=None, help="ITL SLA target in milliseconds")
    parser.add_argument(
        "--load-predictor", type=str, default=None, choices=["constant", "arima", "prophet"],
        help="Load predictor type"
    )
    parser.add_argument("--load-prediction-window-size", type=int, default=None, help="Predictor window size")
    parser.add_argument("--no-correction", action="store_true", default=None, help="Disable correction factors")

    # Profiling
    parser.add_argument("--profile-results-dir", type=str, default=None, help="Path to profiling data directory")

    # Metrics exposition
    parser.add_argument("--planner-prometheus-port", type=int, default=None, help="Port for planner metrics (0 to disable)")

    # Operation mode
    parser.add_argument("--no-operation", action="store_true", default=None, help="Dry-run mode (no scaling applied)")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PlannerConfig:
    """Build PlannerConfig from CLI args (override) + env vars (default)."""
    config = PlannerConfig()

    # CLI args override env vars
    if args.rbg_name is not None:
        config.rbg_name = args.rbg_name
    if args.rbg_namespace is not None:
        config.rbg_namespace = args.rbg_namespace
    if args.prefill_role_name is not None:
        config.prefill_role_name = args.prefill_role_name
    if args.decode_role_name is not None:
        config.decode_role_name = args.decode_role_name
    if args.prometheus_endpoint is not None:
        config.prometheus_endpoint = args.prometheus_endpoint
    if args.metric_source is not None:
        config.metric_source = args.metric_source
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.adjustment_interval is not None:
        config.adjustment_interval = args.adjustment_interval
    if args.max_gpu_budget is not None:
        config.max_gpu_budget = args.max_gpu_budget
    if args.min_replicas is not None:
        config.min_replicas = args.min_replicas
    if args.prefill_engine_num_gpu is not None:
        config.prefill_engine_num_gpu = args.prefill_engine_num_gpu
    if args.decode_engine_num_gpu is not None:
        config.decode_engine_num_gpu = args.decode_engine_num_gpu
    if args.ttft_sla is not None:
        config.ttft_sla = args.ttft_sla
    if args.itl_sla is not None:
        config.itl_sla = args.itl_sla
    if args.load_predictor is not None:
        config.load_predictor = args.load_predictor
    if args.load_prediction_window_size is not None:
        config.load_prediction_window_size = args.load_prediction_window_size
    if args.no_correction is not None:
        config.no_correction = args.no_correction
    if args.profile_results_dir is not None:
        config.profile_results_dir = args.profile_results_dir
    if args.planner_prometheus_port is not None:
        config.planner_prometheus_port = args.planner_prometheus_port
    if args.no_operation is not None:
        config.no_operation = args.no_operation

    return config


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = build_config(args)

    if not config.rbg_name and not config.no_operation:
        logger.error("RBG_NAME is required (set via --rbg-name or RBG_NAME env var)")
        sys.exit(1)

    logger.info(f"RBG Planner starting for {config.rbg_name} in {config.rbg_namespace}")
    logger.info(f"Metric source: {config.metric_source}, Model: {config.model_name or '(all)'}")
    logger.info(f"SLA targets: TTFT={config.ttft_sla}ms, ITL={config.itl_sla}ms")

    planner = Planner(config)
    asyncio.run(planner.run())


if __name__ == "__main__":
    main()
