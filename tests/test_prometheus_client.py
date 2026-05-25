"""Unit tests for the Prometheus metrics client."""

from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.prometheus_client import METRIC_NAMES, PrometheusMetricsClient


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.prometheus_client.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


class TestPrometheusMetricsClientInit:
    def test_sglang_source(self, mock_prom):
        client = PrometheusMetricsClient(url="http://prom:9090", metric_source="sglang")
        assert client.metric_source == "sglang"
        assert client.metrics == METRIC_NAMES["sglang"]

    def test_vllm_source(self, mock_prom):
        client = PrometheusMetricsClient(url="http://prom:9090", metric_source="vllm")
        assert client.metric_source == "vllm"
        assert client.metrics == METRIC_NAMES["vllm"]

    def test_patio_source(self, mock_prom):
        client = PrometheusMetricsClient(url="http://prom:9090", metric_source="patio")
        assert client.metric_source == "patio"

    def test_invalid_source_raises(self, mock_prom):
        with pytest.raises(ValueError, match="Unsupported metric_source"):
            PrometheusMetricsClient(url="http://prom:9090", metric_source="invalid")


class TestLabelFilter:
    def test_no_filters(self, mock_prom):
        client = PrometheusMetricsClient(url="http://prom:9090")
        assert client._build_label_filter() == ""

    def test_namespace_filter(self, mock_prom):
        client = PrometheusMetricsClient(url="http://prom:9090", namespace="prod")
        assert client._build_label_filter() == '{namespace="prod"}'

    def test_model_filter(self, mock_prom):
        client = PrometheusMetricsClient(url="http://prom:9090")
        assert client._build_label_filter(model_name="Qwen/Qwen3-0.6B") == '{model="qwen/qwen3-0.6b"}'

    def test_both_filters(self, mock_prom):
        client = PrometheusMetricsClient(url="http://prom:9090", namespace="ns1")
        result = client._build_label_filter(model_name="MyModel")
        assert 'namespace="ns1"' in result
        assert 'model="mymodel"' in result


class TestQueryMetrics:
    def test_avg_metric_with_data(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "0.7"]},
        ]
        client = PrometheusMetricsClient(url="http://prom:9090")
        result = client.get_avg_time_to_first_token("180s")
        assert result == pytest.approx(0.6, rel=0.01)

    def test_avg_metric_no_data(self, mock_prom):
        mock_prom.custom_query.return_value = []
        client = PrometheusMetricsClient(url="http://prom:9090")
        result = client.get_avg_time_to_first_token("180s")
        assert result == 0.0

    def test_avg_metric_nan_filtered(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "NaN"]},
        ]
        client = PrometheusMetricsClient(url="http://prom:9090")
        result = client.get_avg_time_to_first_token("180s")
        assert result == pytest.approx(0.5)

    def test_request_count(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "100"]},
            {"value": [1234567890, "50"]},
        ]
        client = PrometheusMetricsClient(url="http://prom:9090")
        result = client.get_request_count("180s")
        assert result == 150.0

    def test_error_returns_zero(self, mock_prom):
        mock_prom.custom_query.side_effect = Exception("connection error")
        client = PrometheusMetricsClient(url="http://prom:9090")
        result = client.get_avg_time_to_first_token("180s")
        assert result == 0.0

    def test_avg_input_tokens(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "512.0"]},
        ]
        client = PrometheusMetricsClient(url="http://prom:9090")
        result = client.get_avg_input_sequence_tokens("180s")
        assert result == 512.0

    def test_avg_output_tokens(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "128.0"]},
        ]
        client = PrometheusMetricsClient(url="http://prom:9090")
        result = client.get_avg_output_sequence_tokens("180s")
        assert result == 128.0
