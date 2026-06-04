"""RBG Profiler - SLA profiling tool for PD-disaggregated inference.

Runs benchmarks against inference engines at various configurations
to generate profiling data (prefill_raw_data.json, decode_raw_data.json)
and stores results in a Kubernetes ConfigMap.
"""

import argparse
import logging
import os
import sys

from rbg_profiler.profiler import run_profiling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RBG SLA Profiler")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"), help="Inference endpoint URL")
    parser.add_argument("--model-name", default=os.environ.get("MODEL", ""), help="Model name for bench_serving")
    parser.add_argument("--tokenizer", default=os.environ.get("TOKENIZER", ""), help="Tokenizer path")
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "1")), help="Number of GPUs per engine")
    parser.add_argument("--max-kv-tokens", type=int, default=int(os.environ.get("MAX_KV_TOKENS", "32768")), help="Max KV cache tokens")
    parser.add_argument("--max-context-length", type=int, default=int(os.environ.get("MAX_CONTEXT_LENGTH", "32768")), help="Max context length")
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "default"), help="Kubernetes namespace")
    parser.add_argument("--output-configmap", default=os.environ.get("CONFIGMAP_NAME", "profiling-results"), help="ConfigMap name for results")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/tmp/profiling_results"), help="Local output directory")
    parser.add_argument("--update-configmap", action="store_true", default=os.environ.get("UPDATE_CONFIGMAP", "true").lower() == "true", help="Update ConfigMap with results")
    parser.add_argument("--prefill-isl-points", default=os.environ.get("PREFILL_ISL_POINTS", "128,256,512,1024,1536,2048,3072,4096"), help="Comma-separated prefill ISL points")
    parser.add_argument("--decode-isl-points", default=os.environ.get("DECODE_ISL_POINTS", "256,512,1024,2048,3072"), help="Comma-separated decode ISL points")
    parser.add_argument("--decode-granularity", type=int, default=int(os.environ.get("DECODE_GRANULARITY", "6")), help="Number of concurrency sweep points")
    args = parser.parse_args()

    try:
        run_profiling(args)
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
