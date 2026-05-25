"""Prometheus metrics querying for RBG Planner.

Supports multiple metric sources:
- sglang: SGLang native metrics
- vllm: vLLM native metrics
- patio: Unified patio metrics (future, via EngineRuntime sidecar)
"""

import logging
from typing import Optional

from prometheus_api_client import PrometheusConnect

logger = logging.getLogger(__name__)


# Metric name mappings per source
METRIC_NAMES = {
    "sglang": {
        "ttft": "sglang:time_to_first_token_seconds",
        "itl": "sglang:inter_token_latency_seconds",
        "request_duration": "sglang:e2e_request_latency_seconds",
        "requests_total": "sglang:num_requests_total",
        "prompt_tokens": "sglang:num_prompt_tokens_total",
        "generation_tokens": "sglang:num_generation_tokens_total",
    },
    "vllm": {
        "ttft": "vllm:time_to_first_token_seconds",
        "itl": "vllm:time_per_output_token_seconds",
        "request_duration": "vllm:e2e_request_latency_seconds",
        "requests_total": "vllm:num_requests_total",
        "prompt_tokens": "vllm:prompt_tokens_total",
        "generation_tokens": "vllm:generation_tokens_total",
    },
    "patio": {
        "ttft": "patio:time_to_first_token_seconds",
        "itl": "patio:inter_token_latency_seconds",
        "request_duration": "patio:request_duration_seconds",
        "requests_total": "patio:requests_total",
        "prompt_tokens": "patio:prompt_tokens_total",
        "generation_tokens": "patio:generation_tokens_total",
    },
}


class PrometheusMetricsClient:
    """Queries Prometheus for inference metrics needed by the planner."""

    def __init__(
        self,
        url: str,
        metric_source: str = "sglang",
        namespace: Optional[str] = None,
    ):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)
        self.namespace = namespace

        if metric_source not in METRIC_NAMES:
            raise ValueError(
                f"Unsupported metric_source: {metric_source}. "
                f"Supported: {list(METRIC_NAMES.keys())}"
            )
        self.metrics = METRIC_NAMES[metric_source]
        self.metric_source = metric_source

    def _build_label_filter(self, model_name: Optional[str] = None) -> str:
        """Build PromQL label filter string."""
        filters = []
        if self.namespace:
            filters.append(f'namespace="{self.namespace}"')
        if model_name:
            filters.append(f'model="{model_name.lower()}"')
        if filters:
            return "{" + ",".join(filters) + "}"
        return ""

    def _query_avg_metric(
        self, metric_name: str, interval: str, model_name: Optional[str] = None
    ) -> float:
        """Query average of a histogram metric: increase(sum)/increase(count)."""
        label_filter = self._build_label_filter(model_name)
        query = (
            f"increase({metric_name}_sum{label_filter}[{interval}])"
            f"/increase({metric_name}_count{label_filter}[{interval}])"
        )
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                logger.debug(f"No data for {metric_name}")
                return 0.0
            # Average across all matching series
            values = [float(r["value"][1]) for r in result if r["value"][1] != "NaN"]
            if not values:
                return 0.0
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error querying {metric_name}: {e}")
            return 0.0

    def _query_increase(
        self, metric_name: str, interval: str, model_name: Optional[str] = None
    ) -> float:
        """Query increase of a counter metric."""
        label_filter = self._build_label_filter(model_name)
        query = f"increase({metric_name}{label_filter}[{interval}])"
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                return 0.0
            total = 0.0
            for r in result:
                val = float(r["value"][1])
                if not (val != val):  # check NaN
                    total += val
            return total
        except Exception as e:
            logger.error(f"Error querying {metric_name}: {e}")
            return 0.0

    def get_avg_time_to_first_token(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average TTFT in seconds."""
        return self._query_avg_metric(self.metrics["ttft"], interval, model_name)

    def get_avg_inter_token_latency(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average ITL in seconds."""
        return self._query_avg_metric(self.metrics["itl"], interval, model_name)

    def get_avg_request_duration(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average request duration in seconds."""
        return self._query_avg_metric(self.metrics["request_duration"], interval, model_name)

    def get_request_count(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get total request count in the interval."""
        return self._query_increase(self.metrics["requests_total"], interval, model_name)

    def get_avg_input_sequence_tokens(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average input sequence length (tokens).

        Computed as: increase(prompt_tokens_total) / increase(requests_total)
        """
        label_filter = self._build_label_filter(model_name)
        prompt_metric = self.metrics["prompt_tokens"]
        requests_metric = self.metrics["requests_total"]
        query = (
            f"increase({prompt_metric}{label_filter}[{interval}])"
            f"/increase({requests_metric}{label_filter}[{interval}])"
        )
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                return 0.0
            values = [float(r["value"][1]) for r in result if r["value"][1] != "NaN"]
            if not values:
                return 0.0
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error querying avg input tokens: {e}")
            return 0.0

    def get_avg_output_sequence_tokens(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average output sequence length (tokens).

        Computed as: increase(generation_tokens_total) / increase(requests_total)
        """
        label_filter = self._build_label_filter(model_name)
        gen_metric = self.metrics["generation_tokens"]
        requests_metric = self.metrics["requests_total"]
        query = (
            f"increase({gen_metric}{label_filter}[{interval}])"
            f"/increase({requests_metric}{label_filter}[{interval}])"
        )
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                return 0.0
            values = [float(r["value"][1]) for r in result if r["value"][1] != "NaN"]
            if not values:
                return 0.0
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error querying avg output tokens: {e}")
            return 0.0
