# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring for RoleBasedGroup.

"""Patio metrics adapter.

Patio provides a unified metrics layer via the EngineRuntime sidecar.
Metric patterns follow the same histogram/counter conventions as SGLang.
"""

from rbg_planner.metrics.prometheus import MetricNames, PrometheusAdapter


class PatioAdapter(PrometheusAdapter):

    def __init__(self, url: str):
        super().__init__(url)
        self.metrics = MetricNames(
            ttft="patio:time_to_first_token_seconds",
            itl="patio:inter_token_latency_seconds",
            request_duration="patio:request_duration_seconds",
            requests_total="patio:requests_total",
            prompt_tokens="patio:prompt_tokens_total",
            generation_tokens="patio:generation_tokens_total",
            model_label="model_name",
        )
