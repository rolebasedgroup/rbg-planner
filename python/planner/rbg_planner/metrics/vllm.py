# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring for RoleBasedGroup.

"""vLLM metrics adapter.

vLLM exposes latency as histograms and token counts as counters.
ISL/OSL are derived from counter ratios: prompt_tokens / requests.
"""

from rbg_planner.metrics.prometheus import MetricNames, PrometheusAdapter


class VLLMAdapter(PrometheusAdapter):

    def __init__(self, url: str):
        super().__init__(url)
        self.metrics = MetricNames(
            ttft="vllm:time_to_first_token_seconds",
            itl="vllm:time_per_output_token_seconds",
            request_duration="vllm:e2e_request_latency_seconds",
            requests_total="vllm:num_requests_total",
            prompt_tokens="vllm:prompt_tokens_total",
            generation_tokens="vllm:generation_tokens_total",
            model_label="model_name",
        )
