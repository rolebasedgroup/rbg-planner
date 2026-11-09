# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable metrics adapter module.

Resolves the final PromQL query per logical slot from an engine preset
(defaults.py), overridden per-slot by user-provided queries (the CRD
metrics.queries field), then substitutes the ``{modelFilter}`` and
``{interval}`` placeholders before handing the queries to the adapter.

Usage::

    from rbg_planner.metrics import create_metrics_adapter

    adapter = create_metrics_adapter("sglang", "http://prometheus:9090",
                                     model_name="Qwen/Qwen3-4B", interval="180s")
    ttft = adapter.get_avg_ttft()
"""

from typing import Optional

from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.defaults import BUILTIN_ENGINES, QUERY_SLOTS
from rbg_planner.metrics.prometheus import PrometheusAdapter

SUPPORTED_ENGINES = list(BUILTIN_ENGINES.keys())


def create_metrics_adapter(
    engine: str,
    prometheus_url: str,
    queries: Optional[dict[str, str]] = None,
    model_name: Optional[str] = None,
    interval: str = "180s",
    username: Optional[str] = None,
    password: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> MetricsAdapter:
    """Create a metrics adapter for the given engine.

    Args:
        engine: Engine identifier. Built-in: sglang, vllm, dynamo, patio.
            Any other value is a custom engine and requires ``queries``
            with all six slots defined.
        prometheus_url: Prometheus server URL.
        queries: Optional per-slot PromQL overrides (keys: ttft, itl,
            requestsTotal, requestDuration, isl, osl). Values may use the
            ``{modelFilter}`` and ``{interval}`` placeholders.
        model_name: Model name injected into ``{modelFilter}`` placeholders.
        interval: Query window (e.g. "180s") injected into ``{interval}``
            placeholders.

    Raises:
        ValueError: If the engine is unknown and no queries are given, a
            custom engine's queries are incomplete, or a query key is not
            a known slot.
    """
    unknown = [k for k in (queries or {}) if k not in QUERY_SLOTS]
    if unknown:
        raise ValueError(f"Unknown query slots: {unknown}. Valid slots: {list(QUERY_SLOTS)}")

    is_builtin = engine in BUILTIN_ENGINES
    if not is_builtin and not queries:
        raise ValueError(
            f"Unsupported metrics engine: {engine}. Provide queries to define "
            f"a custom engine, or use one of: {SUPPORTED_ENGINES}"
        )

    if is_builtin:
        preset = BUILTIN_ENGINES[engine]
        resolved: dict[str, str] = dict(preset["queries"])
        model_label = preset["modelLabel"]
    else:
        resolved = {}
        model_label = "model_name"

    for slot, query in (queries or {}).items():
        resolved[slot] = query

    if not is_builtin:
        missing = [slot for slot in QUERY_SLOTS if slot not in resolved]
        if missing:
            raise ValueError(f"Custom engine missing required queries: {missing}")

    # {modelFilter} expands to a full label matcher (braces included), e.g.
    # '{model_name="Qwen"}', or '{}' when no model name is configured.
    model_filter = f'{{{model_label}="{model_name}"}}' if model_name else "{}"
    final = {
        slot: template.replace("{modelFilter}", model_filter).replace("{interval}", interval)
        for slot, template in resolved.items()
    }

    return PrometheusAdapter(
        prometheus_url,
        final,
        username=username,
        password=password,
        bearer_token=bearer_token,
    )
