# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Profiling methodology adapted from dynamo/planner/offline for RoleBasedGroup.

"""SLA Profiler for PD-disaggregated inference.

Profiles prefill and decode engines through the frontend endpoint using
pluggable bench backends. Generates profiling data compatible with the
RBG planner's perf_interpolation module.

Output: prefill_raw_data.json + decode_raw_data.json -> ConfigMap update.
"""

import asyncio
import json
import logging
import os
import sys

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

from rbg_profiler.bench import create_bench_backend
from rbg_profiler.rbg_discoverer import RoleEndpoint

logger = logging.getLogger(__name__)


def _update_configmap(name: str, namespace: str, prefill_data: dict, decode_data: dict):
    """Create or update a Kubernetes ConfigMap with profiling results."""
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    cm = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        data={
            "prefill_raw_data.json": json.dumps(prefill_data, indent=2),
            "decode_raw_data.json": json.dumps(decode_data, indent=2),
        },
    )
    try:
        v1.read_namespaced_config_map(name=name, namespace=namespace)
        v1.replace_namespaced_config_map(name=name, namespace=namespace, body=cm)
        logger.info("ConfigMap %s/%s updated", namespace, name)
    except client.ApiException as e:
        if e.status == 404:
            v1.create_namespaced_config_map(namespace=namespace, body=cm)
            logger.info("ConfigMap %s/%s created", namespace, name)
        else:
            raise


async def run_profiling(
    args,
    prefill_endpoint: RoleEndpoint,
    decode_endpoint: RoleEndpoint,
    backend: str = "sglang-oai",
    input_format: str = "auto",
):
    """Run full SLA profiling pipeline and store results."""
    logger.info("SLA Profiler starting")
    logger.info("  PREFILL_BASE_URL: %s", prefill_endpoint.base_url)
    logger.info("  DECODE_BASE_URL: %s", decode_endpoint.base_url)
    logger.info("  MODEL: %s", args.model_name)
    logger.info("  BACKEND: %s", backend)
    logger.info("  INPUT_FORMAT: %s", input_format)
    logger.info("  PREFILL_GPUS: %d", prefill_endpoint.num_gpus)
    logger.info("  DECODE_GPUS: %d", decode_endpoint.num_gpus)
    logger.info("  MAX_KV_TOKENS: %d", args.max_kv_tokens)
    logger.info("  MAX_CONTEXT_LENGTH: %d", args.max_context_length)

    os.makedirs(args.output_dir, exist_ok=True)

    backend_kwargs = {}
    if backend == "sglang-pd-native":
        backend_kwargs["bootstrap_port"] = int(
            getattr(args, "pd_bootstrap_port", 8998)
        )
    bench_backend = create_bench_backend(
        backend, input_format=input_format, **backend_kwargs
    )

    prefill_isl_points = [int(x) for x in args.prefill_isl_points.split(",")]
    decode_isl_points = [int(x) for x in args.decode_isl_points.split(",")]

    # Tokenizer loading priority (spec §Tokenizer Resolution):
    #   1. endpoint.tokenizer (discoverer or operator via --tokenizer)
    #   2. args.model_name as HuggingFace ID fallback
    if not prefill_endpoint.tokenizer:
        prefill_endpoint.tokenizer = args.tokenizer or args.model_name
    if not decode_endpoint.tokenizer:
        decode_endpoint.tokenizer = args.tokenizer or args.model_name
    if not prefill_endpoint.tokenizer or not decode_endpoint.tokenizer:
        logger.error("No tokenizer source: set --tokenizer or --model-name")
        sys.exit(1)

    # Phase 1: Prefill
    logger.info("=" * 60)
    logger.info("PHASE 1: Prefill Profiling")
    logger.info("=" * 60)

    prefill_results = await bench_backend.bench_prefill(
        endpoint=prefill_endpoint,
        model=args.model_name,
        isl_points=prefill_isl_points,
        max_context_length=args.max_context_length,
    )

    prefill_data = {
        "prefill_isl": [r.isl for r in prefill_results],
        "prefill_ttft": [r.ttft_ms for r in prefill_results],
        "prefill_thpt_per_gpu": [r.thpt_per_gpu for r in prefill_results],
    }
    logger.info("Prefill profiling done: %d data points", len(prefill_results))

    # Phase 2: Decode
    logger.info("=" * 60)
    logger.info("PHASE 2: Decode Profiling")
    logger.info("=" * 60)

    decode_results = await bench_backend.bench_decode(
        endpoint=decode_endpoint,
        model=args.model_name,
        isl_points=decode_isl_points,
        max_kv_tokens=args.max_kv_tokens,
        max_context_length=args.max_context_length,
        concurrency_granularity=args.decode_granularity,
    )

    decode_data = {
        "x_kv_usage": [r.kv_usage for r in decode_results],
        "y_context_length": [r.context_length for r in decode_results],
        "z_itl": [r.itl_ms for r in decode_results],
        "z_thpt_per_gpu": [r.thpt_per_gpu for r in decode_results],
        "max_kv_tokens": args.max_kv_tokens,
    }
    logger.info("Decode profiling done: %d data points", len(decode_results))

    # Save locally
    prefill_path = os.path.join(args.output_dir, "prefill_raw_data.json")
    decode_path = os.path.join(args.output_dir, "decode_raw_data.json")
    with open(prefill_path, "w") as f:
        json.dump(prefill_data, f, indent=2)
    with open(decode_path, "w") as f:
        json.dump(decode_data, f, indent=2)
    logger.info("Results saved to %s", args.output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 60)
    logger.info("Prefill: %d points", len(prefill_data["prefill_isl"]))
    for i, isl in enumerate(prefill_data["prefill_isl"]):
        logger.info(
            "  ISL=%d: TTFT=%.1fms, thpt/gpu=%.0f tok/s",
            isl, prefill_data["prefill_ttft"][i], prefill_data["prefill_thpt_per_gpu"][i],
        )
    logger.info("Decode: %d points", len(decode_data["x_kv_usage"]))
    if decode_data["z_thpt_per_gpu"]:
        logger.info(
            "  thpt/gpu range: %.0f - %.0f tok/s",
            min(decode_data["z_thpt_per_gpu"]), max(decode_data["z_thpt_per_gpu"]),
        )
        logger.info(
            "  ITL range: %.1f - %.1f ms",
            min(decode_data["z_itl"]), max(decode_data["z_itl"]),
        )

    # Update ConfigMap
    if args.update_configmap:
        _update_configmap(args.output_configmap, args.namespace, prefill_data, decode_data)

    # Print JSON to stdout for recovery from pod logs
    print("\n===PREFILL_RAW_DATA_JSON_START===")
    print(json.dumps(prefill_data, indent=2))
    print("===PREFILL_RAW_DATA_JSON_END===")
    print("\n===DECODE_RAW_DATA_JSON_START===")
    print(json.dumps(decode_data, indent=2))
    print("===DECODE_RAW_DATA_JSON_END===")

    logger.info("Profiling complete!")
