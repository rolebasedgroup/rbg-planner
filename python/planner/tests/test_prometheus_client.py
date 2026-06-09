"""Unit tests for the metrics adapter module."""

from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.metrics import ADAPTERS, create_metrics_adapter
from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.dynamo import DynamoAdapter
from rbg_planner.metrics.sglang import SGLangAdapter
from rbg_planner.metrics.vllm import VLLMAdapter


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


class TestFactory:
    def test_create_sglang(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert isinstance(adapter, SGLangAdapter)

    def test_create_vllm(self, mock_prom):
        adapter = create_metrics_adapter("vllm", "http://prom:9090")
        assert isinstance(adapter, VLLMAdapter)

    def test_create_dynamo(self, mock_prom):
        adapter = create_metrics_adapter("dynamo", "http://prom:9090")
        assert isinstance(adapter, DynamoAdapter)

    def test_invalid_source_raises(self, mock_prom):
        with pytest.raises(ValueError, match="Unsupported metric source"):
            create_metrics_adapter("invalid", "http://prom:9090")

    def test_all_adapters_implement_interface(self):
        for name, cls in ADAPTERS.items():
            assert issubclass(cls, MetricsAdapter), f"{name} adapter must implement MetricsAdapter"


class TestLabelFilter:
    def test_no_filters(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter._build_label_filter() == ""

    def test_model_filter_sglang(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter._build_label_filter(model_name="Qwen/Qwen3-0.6B") == '{model_name="Qwen/Qwen3-0.6B"}'

    def test_model_filter_dynamo(self, mock_prom):
        adapter = create_metrics_adapter("dynamo", "http://prom:9090")
        assert adapter._build_label_filter(model_name="Qwen/Qwen3-0.6B") == '{model="Qwen/Qwen3-0.6B"}'


class TestQueryMetrics:
    def test_avg_ttft_with_data(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "0.7"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == pytest.approx(0.6, rel=0.01)

    def test_avg_ttft_no_data(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == 0.0

    def test_avg_ttft_nan_filtered(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "NaN"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == pytest.approx(0.5)

    def test_request_count(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "100"]},
            {"value": [1234567890, "50"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_request_count("180s")
        assert result == 150.0

    def test_error_returns_zero(self, mock_prom):
        mock_prom.custom_query.side_effect = Exception("connection error")
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == 0.0

    def test_avg_isl(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "512.0"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_isl("180s")
        assert result == 512.0

    def test_avg_osl(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "128.0"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_osl("180s")
        assert result == 128.0
