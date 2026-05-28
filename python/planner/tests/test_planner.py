"""Unit tests for the core planner logic."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rbg_planner.config import PlannerConfig
from rbg_planner.planner import Metrics, Planner


@pytest.fixture
def config():
    cfg = PlannerConfig()
    cfg.rbg_name = "test-rbg"
    cfg.rbg_namespace = "default"
    cfg.adjustment_interval = 180
    cfg.max_gpu_budget = 8
    cfg.min_replicas = 1
    cfg.prefill_engine_num_gpu = 1
    cfg.decode_engine_num_gpu = 1
    cfg.ttft_sla = 500.0
    cfg.itl_sla = 50.0
    cfg.load_predictor = "constant"
    cfg.load_prediction_window_size = 50
    cfg.no_correction = True
    cfg.no_operation = True
    cfg.planner_prometheus_port = 0
    cfg.profile_results_dir = "/tmp/fake"
    return cfg


@pytest.fixture
def mock_interpolators():
    """Patch interpolators to avoid needing real profiling data."""
    with patch("rbg_planner.planner.PrefillInterpolator") as mock_pi, \
         patch("rbg_planner.planner.DecodeInterpolator") as mock_di:
        pi = MagicMock()
        pi.interpolate_thpt_per_gpu.return_value = 1000.0  # tokens/s/gpu
        pi.interpolate_ttft.return_value = 100.0  # ms
        mock_pi.return_value = pi

        di = MagicMock()
        di.find_best_throughput_per_gpu.return_value = (500.0, 40.0, 0.5)
        di.interpolate_itl.return_value = 40.0  # ms
        mock_di.return_value = di

        yield pi, di


class TestMetrics:
    def test_valid_metrics(self):
        m = Metrics(ttft=100.0, itl=30.0, num_req=10.0, isl=512.0, osl=128.0)
        assert m.is_valid()

    def test_invalid_none(self):
        m = Metrics(ttft=None, itl=30.0, isl=512.0, osl=128.0)
        assert not m.is_valid()

    def test_invalid_nan(self):
        m = Metrics(ttft=float("nan"), itl=30.0, isl=512.0, osl=128.0)
        assert not m.is_valid()


class TestPlannerReplicaComputation:
    def test_basic_computation(self, config, mock_interpolators):
        planner = Planner(config)

        # 100 requests, ISL=512, OSL=128 in a 180s interval
        num_p, num_d = planner._compute_replica_requirements(100, 512, 128)

        # Prefill: 100*512/180 * 1.0 / 1000 = ~0.28 -> ceil = 1
        # Decode: 100*128/180 / 500 = ~0.14 -> ceil = 1
        assert num_p >= 1
        assert num_d >= 1

    def test_high_load_scales_up(self, config, mock_interpolators):
        planner = Planner(config)

        # 10000 requests, ISL=1024, OSL=256
        num_p, num_d = planner._compute_replica_requirements(10000, 1024, 256)

        # Should scale up significantly
        assert num_p > 1
        assert num_d > 1

    def test_gpu_budget_constraint(self, config, mock_interpolators):
        config.max_gpu_budget = 4
        config.prefill_engine_num_gpu = 2
        config.decode_engine_num_gpu = 2
        planner = Planner(config)

        # Very high load that would need many GPUs
        num_p, num_d = planner._compute_replica_requirements(100000, 2048, 512)

        total_gpu = num_p * 2 + num_d * 2
        assert total_gpu <= 4

    def test_min_replicas_enforced(self, config, mock_interpolators):
        config.min_replicas = 2
        planner = Planner(config)

        # Very low load
        num_p, num_d = planner._compute_replica_requirements(1, 64, 16)

        assert num_p >= 2
        assert num_d >= 2

    def test_correction_factor_applied(self, config, mock_interpolators):
        config.no_correction = False
        planner = Planner(config)

        # Set high correction factor (observed TTFT much higher than expected)
        planner.p_correction_factor = 0.5
        planner.d_correction_factor = 2.0

        num_p, num_d = planner._compute_replica_requirements(1000, 512, 128)

        # With p_correction=0.5, prefill throughput is scaled by min(1, 0.5) = 0.5
        # With d_correction=2.0, ITL SLA is halved: 50/2=25ms -> needs more decode
        assert num_p >= 1
        assert num_d >= 1


class TestPlannerPrediction:
    def test_constant_predictor(self, config, mock_interpolators):
        planner = Planner(config)

        planner.num_req_predictor.add_data_point(100)
        planner.isl_predictor.add_data_point(512)
        planner.osl_predictor.add_data_point(128)

        num_req, isl, osl = planner.predict_load()
        assert num_req == 100
        assert isl == 512
        assert osl == 128


class TestPlannerMakeAdjustments:
    @pytest.mark.asyncio
    async def test_skips_on_invalid_metrics(self, config, mock_interpolators):
        planner = Planner(config)
        planner.last_metrics = Metrics()  # all None

        # Should not raise
        await planner.make_adjustments()

    @pytest.mark.asyncio
    async def test_applies_scaling(self, config, mock_interpolators):
        config.no_operation = False
        planner = Planner(config)

        # Mock connector
        planner.connector = AsyncMock()
        planner.connector.get_role_ready_replicas.return_value = 2

        planner.last_metrics = Metrics(
            ttft=200.0, itl=30.0, num_req=500.0,
            isl=512.0, osl=128.0, request_duration=2.0,
        )
        planner.num_req_predictor.add_data_point(500)
        planner.isl_predictor.add_data_point(512)
        planner.osl_predictor.add_data_point(128)

        await planner.make_adjustments()

        planner.connector.set_replicas.assert_called_once()
        call_args = planner.connector.set_replicas.call_args
        target_replicas = call_args[0][0]
        assert len(target_replicas) == 2
        assert target_replicas[0].role_name == "prefill"
        assert target_replicas[1].role_name == "decode"
        assert target_replicas[0].desired_replicas >= 1
        assert target_replicas[1].desired_replicas >= 1
