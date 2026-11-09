# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for the configurable metrics queries feature.

These tests verify the full round-trip: the golden JSON fixture (which the Go
controller produces via json.Marshal) flows through the Python stack — env var,
PlannerConfig, factory query resolution — to produce correct PromQL queries.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.config import PlannerConfig
from rbg_planner.metrics import create_metrics_adapter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "metric_config_golden.json"


@pytest.fixture
def golden_json():
    return FIXTURE_PATH.read_text()


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


class TestGoldenJSONIntegration:
    def test_golden_json_full_stack(self, mock_prom, golden_json):
        """The golden JSON from the Go controller flows through the full
        Python stack: env var → PlannerConfig → factory → adapter → PromQL."""
        # 1. Env var → PlannerConfig
        with patch.dict(os.environ, {"METRIC_QUERIES_JSON": golden_json}):
            config = PlannerConfig()
        assert config.metric_queries_json == golden_json

        # 2. Parse JSON → factory → adapter (engine preset + per-slot overrides)
        queries = json.loads(config.metric_queries_json)
        adapter = create_metrics_adapter(
            "sglang",
            "http://prom:9090",
            queries=queries,
            model_name="Qwen",
            interval="180s",
        )

        # 3. queries.ttft override with placeholders fully substituted
        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        assert adapter.get_avg_ttft() == 0.5
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == 'sum(rate(custom_ttft_seconds{model_name="Qwen"}[180s]))'

        # 4. queries.isl override with placeholders fully substituted
        mock_prom.custom_query.reset_mock()
        mock_prom.custom_query.return_value = [{"value": [0, "512.0"]}]
        assert adapter.get_avg_isl() == 512.0
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == (
            'sum(increase(custom_prompt_tokens{model_name="Qwen"}[180s]))'
            ' / sum(increase(custom_requests{model_name="Qwen"}[180s]))'
        )

        # 5. Unspecified slots fall back to the sglang preset.
        assert "sglang:inter_token_latency_seconds_sum" in adapter.queries["itl"]

    def test_backward_compat_no_queries(self, mock_prom):
        """Empty metric_queries_json → adapter uses built-in sglang presets
        → correct PromQL query."""
        with patch.dict(os.environ, {"BACKEND": "sglang-oai"}, clear=False):
            os.environ.pop("METRIC_ENGINE", None)
            os.environ.pop("METRIC_QUERIES_JSON", None)
            config = PlannerConfig()

        assert config.metric_queries_json == ""

        # No queries → built-in sglang preset
        adapter = create_metrics_adapter(
            config.metric_engine,
            "http://prom:9090",
            model_name="Qwen",
            interval="180s",
        )

        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        adapter.get_avg_ttft()
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert "sglang:time_to_first_token_seconds_sum" in query
        assert 'model_name="Qwen"' in query  # default model label
        assert "[180s]" in query
