# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Profiling methodology adapted from dynamo/planner/offline for RoleBasedGroup.

"""RBG Profiler - SLA profiling tool for PD-disaggregated inference.

Runs benchmarks against inference engines at various configurations
to generate profiling data (prefill_raw_data.json, decode_raw_data.json)
and stores results in a Kubernetes ConfigMap.
"""

import argparse
import asyncio
import logging
import os
import sys

from rbg_profiler.profiler import run_profiling
from rbg_profiler.rbg_discoverer import RoleEndpoint, discover_role_endpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RBG SLA Profiler")
    parser.add_argument(
        "--rbg-name",
        default=os.environ.get("RBG_NAME", ""),
        help="RoleBasedGroup name for endpoint discovery",
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("NAMESPACE", "default"),
        help="Kubernetes namespace",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL", ""),
        help="Model name for benchmarking",
    )

    # --backend selects the bench backend (API format + request handling).
    # This is distinct from the AutoScaler's metricSource field: metricSource
    # is a planner-only field for Prometheus label filtering, while --backend
    # determines the OpenAI-compatible API path (/v1/completions vs /v1/chat/completions).
    # The default works with any inference engine that exposes an OpenAI-compatible API.
    parser.add_argument(
        "--backend",
        default=os.environ.get("BACKEND", os.environ.get("ENGINE", "sglang-oai")),
        help="Bench backend name (e.g. sglang-oai, vllm-chat). "
             "Also accepts --engine for backward compatibility.",
    )
    # Hidden alias for backward compatibility.
    parser.add_argument("--engine", dest="backend", help=argparse.SUPPRESS)

    # --input-format selects the prompt form used by bench requests.
    # "auto" keeps each backend's default (token_ids for completions
    # backends, text for chat backends); "text" sends tokenizer-decoded
    # strings (required by routers such as SMG that reject token ID
    # prompts); "token_ids" sends raw token ID arrays.
    parser.add_argument(
        "--input-format",
        choices=["auto", "text", "token_ids"],
        default=os.environ.get("INPUT_FORMAT", "auto"),
        help="Prompt input form for bench requests (default: auto)",
    )

    # Prefill engine's --disaggregation-bootstrap-port, used by the
    # sglang-pd-native backend when attaching bootstrap fields to PD requests.
    parser.add_argument(
        "--pd-bootstrap-port",
        type=int,
        default=int(os.environ.get("PD_BOOTSTRAP_PORT", "8998")),
        help="Prefill engine disaggregation bootstrap port "
             "(sglang-pd-native backend only)",
    )

    parser.add_argument(
        "--prefill-role-name",
        default=os.environ.get("PREFILL_ROLE_NAME", "prefill"),
        help="Prefill role name in RBG",
    )
    parser.add_argument(
        "--decode-role-name",
        default=os.environ.get("DECODE_ROLE_NAME", "decode"),
        help="Decode role name in RBG",
    )

    # Fallbacks used when RBG discovery cannot determine a value.
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", ""),
        help="Override inference endpoint URL (skips RBG discovery)",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("TOKENIZER", ""),
        help="Tokenizer path or HF model ID (fallback)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "1")),
        help="Number of GPUs per engine (fallback)",
    )

    parser.add_argument(
        "--max-kv-tokens",
        type=int,
        default=int(os.environ.get("MAX_KV_TOKENS", "32768")),
        help="Max KV cache tokens",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=int(os.environ.get("MAX_CONTEXT_LENGTH", "32768")),
        help="Max context length",
    )
    parser.add_argument(
        "--output-configmap",
        default=os.environ.get("CONFIGMAP_NAME", "profiling-results"),
        help="ConfigMap name for results",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "/tmp/profiling_results"),
        help="Local output directory",
    )
    parser.add_argument(
        "--update-configmap",
        action="store_true",
        default=os.environ.get("UPDATE_CONFIGMAP", "true").lower() == "true",
        help="Update ConfigMap with results",
    )
    parser.add_argument(
        "--prefill-isl-points",
        default=os.environ.get(
            "PREFILL_ISL_POINTS", "128,256,512,1024,1536,2048,3072,4096"
        ),
        help="Comma-separated prefill ISL points",
    )
    parser.add_argument(
        "--decode-isl-points",
        default=os.environ.get("DECODE_ISL_POINTS", "256,512,1024,2048,3072"),
        help="Comma-separated decode ISL points",
    )
    parser.add_argument(
        "--decode-granularity",
        type=int,
        default=int(os.environ.get("DECODE_GRANULARITY", "6")),
        help="Number of concurrency sweep points",
    )
    args = parser.parse_args()

    # Discover per-role endpoints from RBG CR, or use --base-url override.
    if args.base_url:
        logger.info(f"Using --base-url override: {args.base_url}")
        prefill_endpoint = RoleEndpoint(
            base_url=args.base_url, num_gpus=args.num_gpus, tokenizer=args.tokenizer
        )
        decode_endpoint = RoleEndpoint(
            base_url=args.base_url, num_gpus=args.num_gpus, tokenizer=args.tokenizer
        )
    elif args.rbg_name:
        prefill_endpoint = discover_role_endpoint(
            rbg_name=args.rbg_name,
            namespace=args.namespace,
            role_name=args.prefill_role_name,
            engine=args.backend,
            fallback_tokenizer=args.tokenizer,
        )
        decode_endpoint = discover_role_endpoint(
            rbg_name=args.rbg_name,
            namespace=args.namespace,
            role_name=args.decode_role_name,
            engine=args.backend,
            fallback_tokenizer=args.tokenizer,
        )
    else:
        logger.error("Either --rbg-name or --base-url must be provided")
        sys.exit(1)

    try:
        asyncio.run(run_profiling(
            args, prefill_endpoint, decode_endpoint,
            backend=args.backend, input_format=args.input_format,
        ))
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
