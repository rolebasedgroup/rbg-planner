# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in engine query presets."""

from rbg_planner.metrics.defaults import BUILTIN_ENGINES, QUERY_SLOTS


class TestBuiltinEngines:
    def test_all_slots_present_per_engine(self):
        for engine, preset in BUILTIN_ENGINES.items():
            assert set(preset["queries"].keys()) == set(QUERY_SLOTS), engine

    def test_sglang_queries_use_colon_prefixed_metrics(self):
        q = BUILTIN_ENGINES["sglang"]["queries"]
        # Real series exported by sglang --enable-metrics use a colon prefix
        # (e.g. sglang:time_to_first_token_seconds), not underscores.
        assert "sglang:time_to_first_token_seconds_sum" in q["ttft"]
        assert "sglang:inter_token_latency_seconds_sum" in q["itl"]
        assert "sglang:num_requests_total" in q["requestsTotal"]
        assert "sglang:prompt_tokens_total" in q["isl"]
        assert "sglang:generation_tokens_total" in q["osl"]

    def test_sglang_model_label(self):
        assert BUILTIN_ENGINES["sglang"]["modelLabel"] == "model_name"

    def test_vllm_queries(self):
        q = BUILTIN_ENGINES["vllm"]["queries"]
        assert "vllm:time_to_first_token_seconds_sum" in q["ttft"]
        assert "vllm:time_per_output_token_seconds_sum" in q["itl"]
        assert BUILTIN_ENGINES["vllm"]["modelLabel"] == "model_name"

    def test_dynamo_isl_osl_use_histograms(self):
        # Dynamo exposes ISL/OSL as histograms, not counter ratios.
        q = BUILTIN_ENGINES["dynamo"]["queries"]
        assert "dynamo_frontend_input_sequence_tokens_sum" in q["isl"]
        assert "dynamo_frontend_input_sequence_tokens_count" in q["isl"]
        assert "dynamo_frontend_output_sequence_tokens_sum" in q["osl"]
        assert BUILTIN_ENGINES["dynamo"]["modelLabel"] == "model"

    def test_patio_queries(self):
        q = BUILTIN_ENGINES["patio"]["queries"]
        assert "patio:time_to_first_token_seconds_sum" in q["ttft"]
        assert "patio:requests_total" in q["requestsTotal"]

    def test_placeholders_present(self):
        # Every preset template must be resolvable via {modelFilter}/{interval}.
        for engine, preset in BUILTIN_ENGINES.items():
            for slot, template in preset["queries"].items():
                assert "{modelFilter}" in template, (engine, slot)
                assert "{interval}" in template, (engine, slot)
