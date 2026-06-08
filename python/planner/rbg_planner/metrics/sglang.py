# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring for RoleBasedGroup.

"""SGLang metrics adapter.

SGLang exposes latency as histograms and token counts as counters.
ISL/OSL are derived from counter ratios: prompt_tokens / requests.
"""

from rbg_planner.metrics.prometheus import MetricNames, PrometheusAdapter


class SGLangAdapter(PrometheusAdapter):

    def __init__(self, url: str):
        super().__init__(url)
        self.metrics = MetricNames(
            ttft="sglang_time_to_first_token_seconds",
            itl="sglang_inter_token_latency_seconds",
            request_duration="sglang_e2e_request_latency_seconds",
            requests_total="sglang_num_requests_total",
            prompt_tokens="sglang_prompt_tokens_total",
            generation_tokens="sglang_generation_tokens_total",
            model_label="model_name",
        )
