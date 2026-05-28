"""RBG Profiler - SLA profiling tool for RBG inference workloads.

Runs benchmarks against inference engines at various configurations
to generate profiling data (prefill_raw_data.json, decode_raw_data.json)
and stores results in a Kubernetes ConfigMap.
"""

import argparse
import json
import logging
import sys

from rbg_profiler.profiler import run_profiling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RBG SLA Profiler")
    parser.add_argument("--model-name", required=True, help="Model name to profile")
    parser.add_argument("--engine", default="sglang", choices=["sglang", "vllm"], help="Inference engine type")
    parser.add_argument("--ttft-sla", type=float, required=True, help="Target TTFT SLA (ms)")
    parser.add_argument("--itl-sla", type=float, required=True, help="Target ITL SLA (ms)")
    parser.add_argument("--rbg-name", required=True, help="Target RBG name")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument("--output-configmap", required=True, help="Name of ConfigMap to create with results")
    parser.add_argument("--output-dir", default="/tmp/profiling-results", help="Local output directory")
    args = parser.parse_args()

    try:
        run_profiling(args)
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
