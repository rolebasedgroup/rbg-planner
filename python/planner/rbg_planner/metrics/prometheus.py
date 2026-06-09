# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring/traffic_metrics.py for RoleBasedGroup.

"""Prometheus-based metrics adapter with shared query utilities.

Concrete adapters (SGLang, vLLM, Dynamo, ...) inherit from this class
and only need to define metric names, label mappings, and override
methods where query patterns differ.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from prometheus_api_client import PrometheusConnect

from rbg_planner.metrics.base import MetricsAdapter

logger = logging.getLogger(__name__)


@dataclass
class MetricNames:
    """Metric name mapping for a specific inference engine."""

    ttft: str = ""
    itl: str = ""
    request_duration: str = ""
    requests_total: str = ""
    prompt_tokens: str = ""
    generation_tokens: str = ""
    model_label: str = "model_name"


class PrometheusAdapter(MetricsAdapter):
    """Base Prometheus adapter with shared PromQL query helpers.

    Subclasses set `self.metrics` with engine-specific metric names
    and override methods where the query pattern differs (e.g. histogram
    vs counter ratio for ISL/OSL).
    """

    metrics: MetricNames

    def __init__(self, url: str):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)

    # ── shared query helpers ─────────────────────────────────────────

    def _build_label_filter(
        self, model_name: Optional[str] = None, role: Optional[str] = None,
    ) -> str:
        filters = []
        if model_name:
            filters.append(f'{self.metrics.model_label}="{model_name}"')
        if role:
            filters.append(f'role="{role}"')
        return "{" + ",".join(filters) + "}" if filters else ""

    def _query_avg_metric(
        self,
        metric_name: str,
        interval: str,
        model_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> float:
        """Average of a histogram: increase(sum) / increase(count)."""
        lf = self._build_label_filter(model_name, role=role)
        query = (
            f"increase({metric_name}_sum{lf}[{interval}])"
            f"/increase({metric_name}_count{lf}[{interval}])"
        )
        return self._exec_avg(query, metric_name)

    def _query_increase(
        self,
        metric_name: str,
        interval: str,
        model_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> float:
        """Sum of increase of a counter metric."""
        lf = self._build_label_filter(model_name, role=role)
        query = f"increase({metric_name}{lf}[{interval}])"
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                return 0.0
            total = 0.0
            for r in result:
                val = float(r["value"][1])
                if not (val != val):  # NaN check
                    total += val
            return total
        except Exception as e:
            logger.error(f"Error querying {metric_name}: {e}")
            return 0.0

    def _query_counter_ratio(
        self,
        numerator: str,
        denominator: str,
        interval: str,
        model_name: Optional[str] = None,
    ) -> float:
        """Ratio of two counter increases: increase(num) / increase(denom)."""
        lf = self._build_label_filter(model_name)
        query = (
            f"increase({numerator}{lf}[{interval}])"
            f"/increase({denominator}{lf}[{interval}])"
        )
        return self._exec_avg(query, numerator)

    def _exec_avg(self, query: str, label: str) -> float:
        """Execute a PromQL query and return the average of all series."""
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                logger.debug(f"No data for {label}")
                return 0.0
            values = [float(r["value"][1]) for r in result if r["value"][1] != "NaN"]
            if not values:
                return 0.0
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error querying {label}: {e}")
            return 0.0

    # ── default implementations (histogram-based latency, counter-based tokens) ──

    def get_avg_ttft(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._query_avg_metric(self.metrics.ttft, interval, model_name)

    def get_avg_itl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._query_avg_metric(self.metrics.itl, interval, model_name)

    def get_avg_request_duration(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._query_avg_metric(self.metrics.request_duration, interval, model_name)

    def get_request_count(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._query_increase(self.metrics.requests_total, interval, model_name)

    def get_avg_isl(self, interval: str, model_name: Optional[str] = None) -> float:
        """Default: counter ratio prompt_tokens / requests_total."""
        return self._query_counter_ratio(
            self.metrics.prompt_tokens, self.metrics.requests_total, interval, model_name,
        )

    def get_avg_osl(self, interval: str, model_name: Optional[str] = None) -> float:
        """Default: counter ratio generation_tokens / requests_total."""
        return self._query_counter_ratio(
            self.metrics.generation_tokens, self.metrics.requests_total, interval, model_name,
        )
