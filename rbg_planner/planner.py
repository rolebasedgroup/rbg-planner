"""Core SLA-based planner for RBG Prefill/Decode autoscaling.

Ported from dynamo/planner/utils/planner_core.py.
Removes all Dynamo runtime dependencies (etcd, NATS, DistributedRuntime).
Scales RBG roles via RBGConnector based on predicted load and SLA targets.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from prometheus_client import Gauge, start_http_server

from rbg_planner.config import PlannerConfig
from rbg_planner.planner_connector import PlannerConnector, TargetReplica
from rbg_planner.prometheus_client import PrometheusMetricsClient
from rbg_planner.rbg_connector import RBGConnector
from rbg_planner.utils.load_predictor import LOAD_PREDICTORS
from rbg_planner.utils.perf_interpolation import DecodeInterpolator, PrefillInterpolator

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    ttft: Optional[float] = None
    itl: Optional[float] = None
    num_req: Optional[float] = None
    isl: Optional[float] = None
    osl: Optional[float] = None
    request_duration: Optional[float] = None

    def is_valid(self) -> bool:
        """Check if all metrics are valid (not None and not NaN)."""
        return (
            self.ttft is not None
            and self.itl is not None
            and self.isl is not None
            and self.osl is not None
            and not math.isnan(self.ttft)
            and not math.isnan(self.itl)
            and not math.isnan(self.isl)
            and not math.isnan(self.osl)
        )


class PlannerPrometheusMetrics:
    """Planner's own exported Prometheus metrics for observability."""

    def __init__(self, prefix: str = "rbg_planner"):
        self.num_p_workers = Gauge(
            f"{prefix}_num_prefill_workers", "Number of prefill workers"
        )
        self.num_d_workers = Gauge(
            f"{prefix}_num_decode_workers", "Number of decode workers"
        )
        self.observed_ttft = Gauge(
            f"{prefix}_observed_ttft_ms", "Observed time to first token (ms)"
        )
        self.observed_itl = Gauge(
            f"{prefix}_observed_itl_ms", "Observed inter-token latency (ms)"
        )
        self.observed_request_rate = Gauge(
            f"{prefix}_observed_request_rate", "Observed request rate (req/s)"
        )
        self.observed_isl = Gauge(
            f"{prefix}_observed_isl", "Observed input sequence length"
        )
        self.observed_osl = Gauge(
            f"{prefix}_observed_osl", "Observed output sequence length"
        )
        self.observed_request_duration = Gauge(
            f"{prefix}_observed_request_duration_seconds",
            "Observed request duration (s)",
        )
        self.p_correction_factor = Gauge(
            f"{prefix}_p_correction_factor", "Prefill correction factor"
        )
        self.d_correction_factor = Gauge(
            f"{prefix}_d_correction_factor", "Decode correction factor"
        )
        self.predicted_request_rate = Gauge(
            f"{prefix}_predicted_request_rate", "Predicted request rate (req/s)"
        )
        self.predicted_isl = Gauge(
            f"{prefix}_predicted_isl", "Predicted input sequence length"
        )
        self.predicted_osl = Gauge(
            f"{prefix}_predicted_osl", "Predicted output sequence length"
        )
        self.predicted_num_p = Gauge(
            f"{prefix}_predicted_num_prefill", "Predicted number of prefill replicas"
        )
        self.predicted_num_d = Gauge(
            f"{prefix}_predicted_num_decode", "Predicted number of decode replicas"
        )
        self.gpu_hours = Gauge(
            f"{prefix}_gpu_hours_total", "Cumulative GPU hours used"
        )


class Planner:
    """SLA-based autoscaler for RBG Prefill/Decode roles.

    The planner loop:
    1. Observes metrics from Prometheus (TTFT, ITL, request count, ISL, OSL)
    2. Predicts next load using configured predictor (ARIMA/Constant/Prophet)
    3. Computes required prefill/decode replicas based on profiling interpolation
    4. Applies scaling decisions via RBGConnector
    """

    def __init__(self, config: PlannerConfig):
        self.config = config

        # Connector for scaling RBG roles
        if not config.no_operation:
            self.connector: PlannerConnector = RBGConnector(
                rbg_name=config.rbg_name,
                rbg_namespace=config.rbg_namespace,
                prefill_role_name=config.prefill_role_name,
                decode_role_name=config.decode_role_name,
            )

        # Prometheus client for querying inference metrics
        self.metrics_client = PrometheusMetricsClient(
            url=config.prometheus_endpoint,
            metric_source=config.metric_source,
        )

        # Load predictors
        predictor_cls = LOAD_PREDICTORS[config.load_predictor]
        self.num_req_predictor = predictor_cls(
            window_size=config.load_prediction_window_size,
        )
        self.isl_predictor = predictor_cls(
            window_size=config.load_prediction_window_size,
        )
        self.osl_predictor = predictor_cls(
            window_size=config.load_prediction_window_size,
        )

        # Performance interpolators from profiling data
        self.prefill_interpolator = PrefillInterpolator(config.profile_results_dir)
        self.decode_interpolator = DecodeInterpolator(config.profile_results_dir)

        # State
        self.last_metrics = Metrics()
        self.last_adjustment_time = time.time()
        self.p_correction_factor = 1.0
        self.d_correction_factor = 1.0
        self.cumulative_gpu_hours = 0.0

        # Planner's own metrics exposition
        self.prom_metrics: Optional[PlannerPrometheusMetrics] = None
        if config.planner_prometheus_port > 0:
            try:
                start_http_server(config.planner_prometheus_port)
                self.prom_metrics = PlannerPrometheusMetrics()
                logger.info(
                    f"Started planner metrics server on port {config.planner_prometheus_port}"
                )
            except Exception as e:
                logger.error(f"Failed to start planner metrics server: {e}")

    def _get_num_workers(self, role_name: str) -> int:
        """Get the number of ready replicas for a role from RBG status."""
        if self.config.no_operation:
            return 1
        return self.connector.get_role_ready_replicas(role_name)

    async def observe_metrics(self):
        """Query Prometheus for current inference metrics."""
        num_p = self._get_num_workers(self.config.prefill_role_name)
        num_d = self._get_num_workers(self.config.decode_role_name)
        logger.info(f"Workers: prefill={num_p}, decode={num_d}")

        interval = f"{self.config.adjustment_interval}s"
        model = self.config.model_name or None

        # Prometheus returns seconds, convert to milliseconds for TTFT/ITL
        self.last_metrics.ttft = (
            self.metrics_client.get_avg_time_to_first_token(interval, model) * 1000
        )
        self.last_metrics.itl = (
            self.metrics_client.get_avg_inter_token_latency(interval, model) * 1000
        )
        self.last_metrics.num_req = self.metrics_client.get_request_count(interval, model)
        self.last_metrics.request_duration = (
            self.metrics_client.get_avg_request_duration(interval, model)
        )
        self.last_metrics.isl = (
            self.metrics_client.get_avg_input_sequence_tokens(interval, model)
        )
        self.last_metrics.osl = (
            self.metrics_client.get_avg_output_sequence_tokens(interval, model)
        )

        logger.info(
            f"Observed: num_req={self.last_metrics.num_req:.2f} "
            f"isl={self.last_metrics.isl:.2f} osl={self.last_metrics.osl:.2f}"
        )
        logger.info(
            f"Observed: ttft={self.last_metrics.ttft:.2f}ms itl={self.last_metrics.itl:.2f}ms"
        )

        # Feed predictors
        self.num_req_predictor.add_data_point(self.last_metrics.num_req)
        self.isl_predictor.add_data_point(self.last_metrics.isl)
        self.osl_predictor.add_data_point(self.last_metrics.osl)

        # Export metrics
        if self.prom_metrics:
            self.prom_metrics.num_p_workers.set(num_p)
            self.prom_metrics.num_d_workers.set(num_d)
            self.prom_metrics.observed_ttft.set(self.last_metrics.ttft)
            self.prom_metrics.observed_itl.set(self.last_metrics.itl)
            self.prom_metrics.observed_request_rate.set(
                self.last_metrics.num_req / self.config.adjustment_interval
            )
            self.prom_metrics.observed_isl.set(self.last_metrics.isl)
            self.prom_metrics.observed_osl.set(self.last_metrics.osl)
            self.prom_metrics.observed_request_duration.set(
                self.last_metrics.request_duration or 0
            )

            # Track GPU hours
            interval_gpu_hours = (
                (
                    num_p * self.config.prefill_engine_num_gpu
                    + num_d * self.config.decode_engine_num_gpu
                )
                * self.config.adjustment_interval
                / 3600
            )
            self.cumulative_gpu_hours += interval_gpu_hours
            self.prom_metrics.gpu_hours.set(self.cumulative_gpu_hours)

    def predict_load(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Predict the next interval's load using configured predictor."""
        try:
            next_num_req = self.num_req_predictor.predict_next()
            next_isl = self.isl_predictor.predict_next()
            next_osl = self.osl_predictor.predict_next()
            logger.info(
                f"Predicted: num_req={next_num_req:.2f} isl={next_isl:.2f} osl={next_osl:.2f}"
            )
            return next_num_req, next_isl, next_osl
        except Exception as e:
            logger.error(f"Failed to predict load: {e}")
            return None, None, None

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> tuple[int, int]:
        """Compute the number of prefill and decode replicas needed.

        Prefill: based on predicted token throughput vs profiled throughput/gpu.
        Decode: based on ITL SLA target, corrected by observation vs prediction ratio.
        """
        # Prefill: compute required replicas based on token throughput
        pred_prefill_throughput = (
            next_num_req
            * next_isl
            / self.config.adjustment_interval
            * min(1, self.p_correction_factor)
        )
        prefill_engine_cap = (
            self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl)
            * self.config.prefill_engine_num_gpu
        )
        next_num_p = math.ceil(pred_prefill_throughput / prefill_engine_cap)

        logger.info(
            f"Prefill: {pred_prefill_throughput:.2f}(tok/s) / "
            f"{prefill_engine_cap:.2f}(engine_cap) = {next_num_p}(replicas)"
        )

        # Decode: find throughput/gpu that achieves ITL <= corrected SLA
        if self.d_correction_factor <= 0:
            logger.warning(
                f"d_correction_factor={self.d_correction_factor}, using 1.0"
            )
            corrected_itl = self.config.itl_sla
        else:
            corrected_itl = self.config.itl_sla / self.d_correction_factor

        pred_decode_thpt_per_gpu, _, _ = (
            self.decode_interpolator.find_best_throughput_per_gpu(
                itl=corrected_itl, context_length=next_isl + next_osl / 2
            )
        )
        pred_decode_throughput = (
            next_num_req * next_osl / self.config.adjustment_interval
        )
        decode_engine_cap = pred_decode_thpt_per_gpu * self.config.decode_engine_num_gpu
        next_num_d = math.ceil(pred_decode_throughput / decode_engine_cap)

        logger.info(
            f"Decode: {pred_decode_throughput:.2f}(tok/s) / "
            f"{decode_engine_cap:.2f}(engine_cap) = {next_num_d}(replicas)"
        )

        # Enforce minimums
        next_num_p = max(next_num_p, self.config.min_replicas)
        next_num_d = max(next_num_d, self.config.min_replicas)

        # Enforce GPU budget
        total_gpu = (
            next_num_p * self.config.prefill_engine_num_gpu
            + next_num_d * self.config.decode_engine_num_gpu
        )
        if total_gpu > self.config.max_gpu_budget:
            scale = self.config.max_gpu_budget / total_gpu
            next_num_p = max(self.config.min_replicas, round(next_num_p * scale))
            next_num_d = max(
                self.config.min_replicas,
                round(
                    (
                        self.config.max_gpu_budget
                        - next_num_p * self.config.prefill_engine_num_gpu
                    )
                    / self.config.decode_engine_num_gpu
                ),
            )
            logger.warning(
                f"GPU budget exceeded ({total_gpu} > {self.config.max_gpu_budget}), "
                f"scaled to prefill={next_num_p}, decode={next_num_d}"
            )

        logger.info(f"Target replicas: prefill={next_num_p}, decode={next_num_d}")
        return next_num_p, next_num_d

    def _update_correction_factors(self):
        """Update correction factors based on observed vs expected metrics."""
        num_d = self._get_num_workers(self.config.decode_role_name)
        if num_d == 0:
            logger.warning("No decode workers, skipping correction factor update")
            return

        # TTFT correction: actual / expected (captures queuing delay)
        expect_ttft = self.prefill_interpolator.interpolate_ttft(self.last_metrics.isl)
        if expect_ttft > 0:
            self.p_correction_factor = self.last_metrics.ttft / expect_ttft

        # ITL correction: actual / expected
        concurrency = (
            self.last_metrics.num_req
            / num_d
            * self.last_metrics.request_duration
            / self.config.adjustment_interval
        )
        context_length = self.last_metrics.isl + self.last_metrics.osl / 2
        expect_itl = self.decode_interpolator.interpolate_itl(
            concurrency=concurrency, context_length=context_length
        )
        if expect_itl > 0:
            self.d_correction_factor = self.last_metrics.itl / expect_itl

        logger.info(
            f"Correction factors: TTFT={self.p_correction_factor:.3f}, "
            f"ITL={self.d_correction_factor:.3f}"
        )

        if self.prom_metrics:
            self.prom_metrics.p_correction_factor.set(self.p_correction_factor)
            self.prom_metrics.d_correction_factor.set(self.d_correction_factor)

    async def make_adjustments(self):
        """Compute and apply scaling decisions based on observed and predicted metrics."""
        if not self.last_metrics.is_valid():
            logger.info(
                "Metrics contain None/NaN (no active requests), skipping adjustment"
            )
            return

        # Update correction factors unless disabled
        if not self.config.no_correction:
            try:
                self._update_correction_factors()
            except Exception as e:
                logger.error(f"Failed to update correction factors: {e}")
                return

        # Predict next interval's load
        next_num_req, next_isl, next_osl = self.predict_load()
        if next_num_req is None or next_isl is None or next_osl is None:
            return

        # Export predicted metrics
        if self.prom_metrics:
            self.prom_metrics.predicted_request_rate.set(
                next_num_req / self.config.adjustment_interval
            )
            self.prom_metrics.predicted_isl.set(next_isl)
            self.prom_metrics.predicted_osl.set(next_osl)

        # Compute required replicas
        try:
            next_num_p, next_num_d = self._compute_replica_requirements(
                next_num_req, next_isl, next_osl
            )
        except Exception as e:
            logger.error(f"Failed to compute replica requirements: {e}")
            return

        if self.prom_metrics:
            self.prom_metrics.predicted_num_p.set(next_num_p)
            self.prom_metrics.predicted_num_d.set(next_num_d)

        # Apply scaling
        if not self.config.no_operation:
            target_replicas = [
                TargetReplica(
                    role_name=self.config.prefill_role_name,
                    desired_replicas=next_num_p,
                ),
                TargetReplica(
                    role_name=self.config.decode_role_name,
                    desired_replicas=next_num_d,
                ),
            ]
            await self.connector.set_replicas(target_replicas, blocking=False)

    async def run(self):
        """Main planner loop."""
        if not self.config.no_operation:
            logger.info("Validating RBG deployment...")
            await self.connector.validate_deployment()
            logger.info("RBG deployment validated successfully")

            logger.info("Waiting for RBG to be ready...")
            await self.connector.wait_for_ready()
            logger.info("RBG is ready")

        self.last_adjustment_time = time.time()
        logger.info(
            f"Planner started: interval={self.config.adjustment_interval}s, "
            f"TTFT SLA={self.config.ttft_sla}ms, ITL SLA={self.config.itl_sla}ms"
        )

        while True:
            current_time = time.time()
            if current_time - self.last_adjustment_time >= self.config.adjustment_interval:
                self.last_adjustment_time = time.time()
                logger.info("--- New adjustment interval ---")

                await self.observe_metrics()
                await self.make_adjustments()

            await asyncio.sleep(self.config.adjustment_interval / 10)
