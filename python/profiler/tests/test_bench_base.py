# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bench backend data structures and ABC."""

import pytest

from rbg_profiler.bench.base import (
    BaseBenchBackend,
    BenchBackend,
    BenchMetrics,
    ContextLengthExceededError,
    DecodeProfileData,
    PrefillProfileData,
    RequestFuncOutput,
    UnifiedProfileData,
)
from rbg_profiler.rbg_discoverer import RoleEndpoint


def test_bench_metrics_dataclass():
    m = BenchMetrics(
        mean_ttft_ms=100.0,
        mean_itl_ms=20.0,
        mean_tpot_ms=20.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=1000.0,
    )
    assert m.mean_ttft_ms == 100.0
    assert m.output_throughput == 500.0


def test_prefill_profile_data():
    d = PrefillProfileData(isl=1024, ttft_ms=50.0, thpt_per_gpu=20480.0)
    assert d.isl == 1024
    assert d.thpt_per_gpu == 20480.0


def test_decode_profile_data():
    d = DecodeProfileData(
        isl=512, concurrency=4, kv_usage=0.5,
        context_length=762, itl_ms=15.0, thpt_per_gpu=3000.0,
    )
    assert d.concurrency == 4
    assert d.kv_usage == 0.5


def test_unified_profile_data():
    d = UnifiedProfileData(
        isl=512, concurrency=2, ttft_ms=50.0, itl_ms=15.0, thpt_per_gpu=3000.0,
    )
    assert d.ttft_ms == 50.0


def test_bench_backend_is_abstract():
    """BenchBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BenchBackend()


def test_request_func_output_defaults():
    out = RequestFuncOutput()
    assert out.success is False
    assert out.ttft == 0.0
    assert out.itl == []
    assert out.output_len == 0
    assert out.status_code == 0


def test_calculate_metrics_all_success():
    from rbg_profiler.bench.base import calculate_metrics

    outputs = [
        RequestFuncOutput(success=True, ttft=0.1, itl=[0.02, 0.02], latency=0.15, output_len=3),
        RequestFuncOutput(success=True, ttft=0.12, itl=[0.025, 0.025], latency=0.17, output_len=3),
    ]
    result = calculate_metrics(outputs, dur_s=1.0)
    assert result["completed"] == 2
    assert result["num_errors"] == 0
    assert abs(result["mean_ttft_ms"] - 110.0) < 1.0  # (0.1+0.12)/2 * 1000
    assert result["output_throughput"] == 6.0  # 6 tokens / 1.0s


def test_calculate_metrics_all_fail():
    from rbg_profiler.bench.base import calculate_metrics

    outputs = [RequestFuncOutput(success=False), RequestFuncOutput(success=False)]
    result = calculate_metrics(outputs, dur_s=1.0)
    assert "error" in result
    assert result["num_errors"] == 2


def test_base_bench_backend_is_abstract():
    """BaseBenchBackend cannot be instantiated without overriding abstract methods."""
    with pytest.raises(TypeError):
        BaseBenchBackend()


def test_context_length_exceeded_error():
    e = ContextLengthExceededError()
    assert isinstance(e, Exception)


def test_calculate_metrics_all_fail_with_status_code():
    from rbg_profiler.bench.base import calculate_metrics

    outputs = [
        RequestFuncOutput(success=False, error="Bad Request", status_code=400),
        RequestFuncOutput(success=False, error="Bad Request", status_code=400),
    ]
    result = calculate_metrics(outputs, dur_s=1.0)
    assert "error" in result
    assert result["num_errors"] == 2
    assert result["first_status_code"] == 400
    assert "Bad Request" in result["first_error"]


def test_calculate_metrics_all_fail_rate_limited():
    from rbg_profiler.bench.base import calculate_metrics

    outputs = [
        RequestFuncOutput(success=False, error="Too Many Requests", status_code=429),
    ]
    result = calculate_metrics(outputs, dur_s=1.0)
    assert result["first_status_code"] == 429


def test_calculate_metrics_all_fail_no_status():
    from rbg_profiler.bench.base import calculate_metrics

    outputs = [RequestFuncOutput(success=False), RequestFuncOutput(success=False)]
    result = calculate_metrics(outputs, dur_s=1.0)
    assert "error" in result
    assert result["first_error"] == ""
    assert result["first_status_code"] == 0


class _StubBackend(BenchBackend):
    """Test backend that simulates HTTP error responses when ISL+OSL > limit."""

    def __init__(self, max_model_len: int = 1024, fail_status: int = 400):
        self.max_model_len = max_model_len
        self.fail_status = fail_status
        self.calls: list[tuple[int, int]] = []

    async def run_benchmark(
        self, endpoint, model, num_prompts, isl, osl, request_rate=float("inf"),
    ) -> BenchMetrics:
        self.calls.append((isl, osl))
        if isl + osl > self.max_model_len:
            if self.fail_status == 400:
                raise ContextLengthExceededError()
            if self.fail_status == 429:
                raise RuntimeError("Benchmark failed: rate limited (HTTP 429)")
            raise RuntimeError("Benchmark failed: all requests failed")
        return BenchMetrics(
            mean_ttft_ms=50.0,
            mean_itl_ms=15.0,
            mean_tpot_ms=15.0,
            request_throughput=10.0,
            output_throughput=500.0,
            mean_e2e_latency_ms=100.0,
        )


@pytest.fixture
def stub_endpoint():
    return RoleEndpoint(base_url="http://test:8000", num_gpus=1, tokenizer="t")


@pytest.mark.asyncio
async def test_bench_prefill_stops_on_400(stub_endpoint):
    backend = _StubBackend(max_model_len=1024, fail_status=400)
    results = await backend.bench_prefill(
        endpoint=stub_endpoint,
        model="test-model",
        isl_points=[128, 256, 512, 1024, 2048],
        osl=5,
    )
    assert [r.isl for r in results] == [128, 256, 512]
    # 3 successful + 1 that raised (ISL=1024), 2048 never tried
    assert len(backend.calls) == 4


@pytest.mark.asyncio
async def test_bench_prefill_continues_on_429(stub_endpoint):
    backend = _StubBackend(max_model_len=1024, fail_status=429)
    results = await backend.bench_prefill(
        endpoint=stub_endpoint,
        model="test-model",
        isl_points=[128, 256, 512, 1024, 2048],
        osl=5,
    )
    # 429 is transient — sweep continues, all ISLs attempted
    assert len(backend.calls) == 5
    # Only ISLs within limit have results
    assert [r.isl for r in results] == [128, 256, 512]


@pytest.mark.asyncio
async def test_bench_decode_stops_on_400(stub_endpoint):
    backend = _StubBackend(max_model_len=1024, fail_status=400)
    results = await backend.bench_decode(
        endpoint=stub_endpoint,
        model="test-model",
        isl_points=[256, 512, 1024, 2048],
        max_kv_tokens=32768,
        osl=500,
    )
    isls = {r.isl for r in results}
    assert 256 in isls
    assert 512 in isls
    assert 1024 not in isls
    assert 2048 not in isls
