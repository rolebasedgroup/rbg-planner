# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring for RoleBasedGroup.

"""Dynamo metrics adapter.

Dynamo exposes all metrics (including ISL/OSL) as histograms, unlike
SGLang/vLLM which use counters for token counts. This adapter overrides
get_avg_isl/get_avg_osl to use histogram sum/count queries.

Dynamo also uses the label ``model`` instead of ``model_name``.
"""

from typing import Optional

from rbg_planner.metrics.prometheus import MetricNames, PrometheusAdapter


class DynamoAdapter(PrometheusAdapter):

    def __init__(self, url: str):
        super().__init__(url)
        self.metrics = MetricNames(
            ttft="dynamo_frontend_time_to_first_token_seconds",
            itl="dynamo_frontend_inter_token_latency_seconds",
            request_duration="dynamo_frontend_request_duration_seconds",
            requests_total="dynamo_frontend_requests_total",
            prompt_tokens="dynamo_frontend_input_sequence_tokens",
            generation_tokens="dynamo_frontend_output_sequence_tokens",
            model_label="model",
        )

    # Dynamo exposes ISL/OSL as histograms — override counter-ratio default.

    def get_avg_isl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._query_avg_metric(self.metrics.prompt_tokens, interval, model_name)

    def get_avg_osl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._query_avg_metric(self.metrics.generation_tokens, interval, model_name)
