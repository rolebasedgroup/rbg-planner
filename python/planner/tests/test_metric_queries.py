# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for query resolution: engine presets, per-slot overrides, custom engines."""

from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.metrics import create_metrics_adapter
from rbg_planner.metrics.defaults import BUILTIN_ENGINES


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


class TestPresetResolution:
    def test_builtin_engine_resolves_all_slots(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        for slot in BUILTIN_ENGINES["sglang"]["queries"]:
            assert slot in adapter.queries
        # Placeholders are fully substituted.
        for query in adapter.queries.values():
            assert "{modelFilter}" not in query
            assert "{interval}" not in query

    def test_interval_substituted(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090", interval="60s")
        assert "[60s]" in adapter.queries["ttft"]

    def test_model_filter_substituted(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090", model_name="Qwen")
        assert 'model_name="Qwen"' in adapter.queries["ttft"]

    def test_model_filter_dynamo_label(self, mock_prom):
        adapter = create_metrics_adapter("dynamo", "http://prom:9090", model_name="Qwen")
        assert 'model="Qwen"' in adapter.queries["ttft"]

    def test_no_model_name_leaves_empty_matcher(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert "{modelFilter}" not in adapter.queries["ttft"]
        assert "model_name=" not in adapter.queries["ttft"]


class TestSlotOverride:
    def test_override_single_slot(self, mock_prom):
        adapter = create_metrics_adapter(
            "vllm",
            "http://prom:9090",
            queries={"ttft": "sum(rate(vllm:ttft_v2_sum[5m]))"},
        )
        assert adapter.queries["ttft"] == "sum(rate(vllm:ttft_v2_sum[5m]))"
        # Unspecified slots fall back to the engine preset.
        assert "vllm:time_per_output_token_seconds_sum" in adapter.queries["itl"]

    def test_override_placeholders_substituted(self, mock_prom):
        adapter = create_metrics_adapter(
            "vllm",
            "http://prom:9090",
            queries={"ttft": "avg(custom_ttft{modelFilter}[{interval}])"},
            model_name="Qwen",
            interval="180s",
        )
        assert adapter.queries["ttft"] == 'avg(custom_ttft{model_name="Qwen"}[180s])'

    def test_override_dispatch_uses_override(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        adapter = create_metrics_adapter(
            "vllm",
            "http://prom:9090",
            queries={"ttft": "avg(custom_ttft{modelFilter})"},
            model_name="Qwen",
        )
        assert adapter.get_avg_ttft() == 0.5
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == 'avg(custom_ttft{model_name="Qwen"})'

    def test_unknown_slot_raises(self, mock_prom):
        with pytest.raises(ValueError, match="Unknown query slots"):
            create_metrics_adapter("vllm", "http://prom:9090", queries={"bogus": "1"})


class TestCustomEngine:
    def test_custom_engine_complete(self, mock_prom):
        queries = {
            "ttft": "query1",
            "itl": "query2",
            "requestsTotal": "query3",
            "requestDuration": "query4",
            "isl": "query5",
            "osl": "query6",
        }
        adapter = create_metrics_adapter("acme-llm", "http://prom:9090", queries=queries)
        assert adapter.queries["ttft"] == "query1"

    def test_custom_engine_missing_slots_raises(self, mock_prom):
        queries = {"ttft": "query1"}  # missing the rest
        with pytest.raises(ValueError, match="missing required queries"):
            create_metrics_adapter("acme-llm", "http://prom:9090", queries=queries)

    def test_custom_engine_no_queries_raises(self, mock_prom):
        with pytest.raises(ValueError, match="Unsupported metrics engine"):
            create_metrics_adapter("acme-llm", "http://prom:9090")

    def test_custom_engine_model_filter_defaults_to_model_name(self, mock_prom):
        adapter = create_metrics_adapter(
            "acme-llm",
            "http://prom:9090",
            queries={
                "ttft": "avg(x{modelFilter})",
                "itl": "q",
                "requestsTotal": "q",
                "requestDuration": "q",
                "isl": "q",
                "osl": "q",
            },
            model_name="Qwen",
        )
        assert adapter.queries["ttft"] == 'avg(x{model_name="Qwen"})'
