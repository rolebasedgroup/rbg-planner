"""Unit tests for the metrics adapter module."""

from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.metrics import SUPPORTED_ENGINES, create_metrics_adapter
from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.defaults import QUERY_SLOTS
from rbg_planner.metrics.prometheus import PrometheusAdapter


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


def _all_slots_query():
    return {slot: f"query_{slot}" for slot in QUERY_SLOTS}


class TestFactory:
    def test_create_sglang(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert isinstance(adapter, PrometheusAdapter)
        assert "sglang:time_to_first_token_seconds_sum" in adapter.queries["ttft"]

    def test_create_vllm(self, mock_prom):
        adapter = create_metrics_adapter("vllm", "http://prom:9090")
        assert isinstance(adapter, PrometheusAdapter)
        assert "vllm:time_to_first_token_seconds_sum" in adapter.queries["ttft"]

    def test_invalid_engine_raises(self, mock_prom):
        with pytest.raises(ValueError, match="Unsupported metrics engine"):
            create_metrics_adapter("invalid", "http://prom:9090")

    def test_all_builtins_create_metrics_adapter(self, mock_prom):
        mock_prom.custom_query.return_value = []
        for name in SUPPORTED_ENGINES:
            adapter = create_metrics_adapter(name, "http://prom:9090")
            assert isinstance(adapter, MetricsAdapter)


class TestQueryExecution:
    def test_avg_ttft_with_data(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "0.7"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft()
        assert result == pytest.approx(0.6, rel=0.01)

    def test_avg_ttft_no_data(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter.get_avg_ttft() == 0.0

    def test_avg_ttft_nan_filtered(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "NaN"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter.get_avg_ttft() == pytest.approx(0.5)

    def test_request_count(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [1234567890, "150.0"]}]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter.get_request_count() == 150.0

    def test_error_returns_zero(self, mock_prom):
        mock_prom.custom_query.side_effect = Exception("connection error")
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter.get_avg_ttft() == 0.0

    def test_avg_isl(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [1234567890, "512.0"]}]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter.get_avg_isl() == 512.0

    def test_avg_osl(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [1234567890, "128.0"]}]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter.get_avg_osl() == 128.0

    def test_each_slot_executes_its_query(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [0, "1.0"]}]
        adapter = create_metrics_adapter(
            "sglang", "http://prom:9090", queries={slot: f"q_{slot}" for slot in QUERY_SLOTS}
        )
        adapter.get_avg_request_duration()
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == "q_requestDuration"


class TestAdapterConstruction:
    def test_missing_slot_raises(self, mock_prom):
        queries = _all_slots_query()
        del queries["osl"]
        with pytest.raises(ValueError, match="Missing queries for slots"):
            PrometheusAdapter("http://prom:9090", queries)


class TestAuth:
    def test_auth_from_env(self, mock_prom, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_USERNAME", "user")
        monkeypatch.setenv("PROMETHEUS_PASSWORD", "pass$token")
        with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
            PrometheusAdapter("http://prom:9090", _all_slots_query())
            assert mock_cls.call_args.kwargs["auth"] == ("user", "pass$token")

    def test_no_auth_when_env_unset(self, mock_prom, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_USERNAME", raising=False)
        monkeypatch.delenv("PROMETHEUS_PASSWORD", raising=False)
        with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
            PrometheusAdapter("http://prom:9090", _all_slots_query())
            assert mock_cls.call_args.kwargs["auth"] is None

    def test_bearer_token_from_env(self, mock_prom, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_TOKEN", "tok123")
        monkeypatch.delenv("PROMETHEUS_USERNAME", raising=False)
        monkeypatch.delenv("PROMETHEUS_PASSWORD", raising=False)
        with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
            PrometheusAdapter("http://prom:9090", _all_slots_query())
            assert mock_cls.call_args.kwargs["headers"] == {"Authorization": "Bearer tok123"}
            assert mock_cls.call_args.kwargs["auth"] is None

    def test_bearer_token_wins_over_basic(self, mock_prom, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_TOKEN", "tok123")
        monkeypatch.setenv("PROMETHEUS_USERNAME", "user")
        monkeypatch.setenv("PROMETHEUS_PASSWORD", "pass")
        with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
            PrometheusAdapter("http://prom:9090", _all_slots_query())
            assert mock_cls.call_args.kwargs["headers"] == {"Authorization": "Bearer tok123"}
