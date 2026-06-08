# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable metrics adapter module.

Provides a unified MetricsAdapter interface with concrete adapters for
each supported inference engine (SGLang, vLLM, Dynamo, Patio).

Usage::

    from rbg_planner.metrics import create_metrics_adapter

    adapter = create_metrics_adapter("sglang", "http://prometheus:9090")
    ttft = adapter.get_avg_ttft("180s", model_name="Qwen/Qwen3-0.6B")
"""

from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.dynamo import DynamoAdapter
from rbg_planner.metrics.patio import PatioAdapter
from rbg_planner.metrics.sglang import SGLangAdapter
from rbg_planner.metrics.vllm import VLLMAdapter

ADAPTERS: dict[str, type[MetricsAdapter]] = {
    "sglang": SGLangAdapter,
    "vllm": VLLMAdapter,
    "dynamo": DynamoAdapter,
    "patio": PatioAdapter,
}


def create_metrics_adapter(source: str, prometheus_url: str) -> MetricsAdapter:
    """Create a metrics adapter for the given source.

    Args:
        source: Metric source identifier (sglang, vllm, dynamo, patio).
        prometheus_url: Prometheus server URL.

    Raises:
        ValueError: If the source is not supported.
    """
    adapter_cls = ADAPTERS.get(source)
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported metric source: {source}. "
            f"Supported: {list(ADAPTERS.keys())}"
        )
    return adapter_cls(url=prometheus_url)
