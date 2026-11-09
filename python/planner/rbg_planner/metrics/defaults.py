# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in engine defaults: a preset of PromQL query templates per engine.

Each engine preset provides one query template per logical slot (ttft, itl,
requestsTotal, requestDuration, isl, osl). Templates may use two
placeholders, substituted at adapter construction time:

- ``{modelFilter}`` expands to a full label matcher (e.g.
  ``{model_name="Qwen/Qwen3-4B"}``) or ``{}`` when no model name is
  configured. The label name is baked into the preset.
- ``{interval}`` expands to the planner's adjustment interval (e.g. "180s").

User-provided queries from the CRD (metrics.queries) override presets per
slot and may use the same placeholders.
"""

QUERY_SLOTS = ("ttft", "itl", "requestsTotal", "requestDuration", "isl", "osl")

# Engine name -> {"modelLabel": str, "queries": dict[slot, template]}.
BUILTIN_ENGINES: dict[str, dict] = {
    "sglang": {
        "modelLabel": "model_name",
        "queries": {
            "ttft": (
                "sum(increase(sglang:time_to_first_token_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(sglang:time_to_first_token_seconds_count{modelFilter}[{interval}]))"
            ),
            "itl": (
                "sum(increase(sglang:inter_token_latency_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(sglang:inter_token_latency_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestDuration": (
                "sum(increase(sglang:e2e_request_latency_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(sglang:e2e_request_latency_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestsTotal": "sum(increase(sglang:num_requests_total{modelFilter}[{interval}]))",
            "isl": (
                "sum(increase(sglang:prompt_tokens_total{modelFilter}[{interval}]))"
                " / sum(increase(sglang:num_requests_total{modelFilter}[{interval}]))"
            ),
            "osl": (
                "sum(increase(sglang:generation_tokens_total{modelFilter}[{interval}]))"
                " / sum(increase(sglang:num_requests_total{modelFilter}[{interval}]))"
            ),
        },
    },
    "vllm": {
        "modelLabel": "model_name",
        "queries": {
            "ttft": (
                "sum(increase(vllm:time_to_first_token_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(vllm:time_to_first_token_seconds_count{modelFilter}[{interval}]))"
            ),
            "itl": (
                "sum(increase(vllm:time_per_output_token_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(vllm:time_per_output_token_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestDuration": (
                "sum(increase(vllm:e2e_request_latency_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(vllm:e2e_request_latency_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestsTotal": "sum(increase(vllm:num_requests_total{modelFilter}[{interval}]))",
            "isl": (
                "sum(increase(vllm:prompt_tokens_total{modelFilter}[{interval}]))"
                " / sum(increase(vllm:num_requests_total{modelFilter}[{interval}]))"
            ),
            "osl": (
                "sum(increase(vllm:generation_tokens_total{modelFilter}[{interval}]))"
                " / sum(increase(vllm:num_requests_total{modelFilter}[{interval}]))"
            ),
        },
    },
    "dynamo": {
        # Dynamo exposes ISL/OSL as histograms, not counter ratios.
        "modelLabel": "model",
        "queries": {
            "ttft": (
                "sum(increase(dynamo_frontend_time_to_first_token_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(dynamo_frontend_time_to_first_token_seconds_count{modelFilter}[{interval}]))"
            ),
            "itl": (
                "sum(increase(dynamo_frontend_inter_token_latency_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(dynamo_frontend_inter_token_latency_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestDuration": (
                "sum(increase(dynamo_frontend_request_duration_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(dynamo_frontend_request_duration_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestsTotal": "sum(increase(dynamo_frontend_requests_total{modelFilter}[{interval}]))",
            "isl": (
                "sum(increase(dynamo_frontend_input_sequence_tokens_sum{modelFilter}[{interval}]))"
                " / sum(increase(dynamo_frontend_input_sequence_tokens_count{modelFilter}[{interval}]))"
            ),
            "osl": (
                "sum(increase(dynamo_frontend_output_sequence_tokens_sum{modelFilter}[{interval}]))"
                " / sum(increase(dynamo_frontend_output_sequence_tokens_count{modelFilter}[{interval}]))"
            ),
        },
    },
    "patio": {
        "modelLabel": "model_name",
        "queries": {
            "ttft": (
                "sum(increase(patio:time_to_first_token_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(patio:time_to_first_token_seconds_count{modelFilter}[{interval}]))"
            ),
            "itl": (
                "sum(increase(patio:inter_token_latency_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(patio:inter_token_latency_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestDuration": (
                "sum(increase(patio:request_duration_seconds_sum{modelFilter}[{interval}]))"
                " / sum(increase(patio:request_duration_seconds_count{modelFilter}[{interval}]))"
            ),
            "requestsTotal": "sum(increase(patio:requests_total{modelFilter}[{interval}]))",
            "isl": (
                "sum(increase(patio:prompt_tokens_total{modelFilter}[{interval}]))"
                " / sum(increase(patio:requests_total{modelFilter}[{interval}]))"
            ),
            "osl": (
                "sum(increase(patio:generation_tokens_total{modelFilter}[{interval}]))"
                " / sum(increase(patio:requests_total{modelFilter}[{interval}]))"
            ),
        },
    },
}
