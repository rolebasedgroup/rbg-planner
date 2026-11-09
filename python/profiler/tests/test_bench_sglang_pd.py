# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sglang-pd-native bench backend."""

from unittest.mock import AsyncMock

import pytest

from rbg_profiler.bench import create_bench_backend
from rbg_profiler.bench.base import BenchMetrics, RequestFuncInput
from rbg_profiler.bench.sglang_pd import (
    DEFAULT_BOOTSTRAP_PORT,
    FAKE_BOOTSTRAP_HOST,
    SGLangPDNativeBenchBackend,
)
from rbg_profiler.rbg_discoverer import RoleEndpoint


@pytest.fixture
def endpoint():
    return RoleEndpoint(base_url="http://test:8000", num_gpus=1, tokenizer="t")


def _metrics():
    return BenchMetrics(
        mean_ttft_ms=50.0,
        mean_itl_ms=15.0,
        mean_tpot_ms=15.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    )


def test_registry_contains_pd_native():
    backend = create_bench_backend("sglang-pd-native")
    assert isinstance(backend, SGLangPDNativeBenchBackend)
    assert backend.api_path == "/v1/completions"
    # Engine port, not the router port 30000.
    assert backend.default_port == 8000


def test_pd_native_bootstrap_port_configurable():
    assert SGLangPDNativeBenchBackend().bootstrap_port == DEFAULT_BOOTSTRAP_PORT
    assert SGLangPDNativeBenchBackend(bootstrap_port=9001).bootstrap_port == 9001


def test_pd_native_prefill_payload_has_minted_bootstrap_fields():
    backend = SGLangPDNativeBenchBackend()
    payload = backend._build_request_payload(prompt=[1, 2, 3], model="m", osl=1)
    assert payload["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
    assert payload["bootstrap_port"] == DEFAULT_BOOTSTRAP_PORT
    assert isinstance(payload["bootstrap_room"], int)
    assert 0 <= payload["bootstrap_room"] <= 2**63 - 1
    # Standard completions fields are preserved.
    assert payload["prompt"] == [1, 2, 3]
    assert payload["max_tokens"] == 1
    assert payload["stream"] is True


def test_pd_native_prefill_mints_unique_rooms():
    backend = SGLangPDNativeBenchBackend()
    rooms = {
        backend._build_request_payload(prompt=[1], model="m", osl=1)["bootstrap_room"]
        for _ in range(20)
    }
    assert len(rooms) > 1


def test_pd_native_decode_payload_uses_fake_prefill_shortcut():
    backend = SGLangPDNativeBenchBackend()
    backend._phase = "decode"
    payload = backend._build_request_payload(prompt=[1, 2, 3], model="m", osl=500)
    assert payload["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
    assert payload["bootstrap_room"] == 0
    assert payload["bootstrap_port"] == DEFAULT_BOOTSTRAP_PORT


def test_pd_native_custom_bootstrap_port_in_payload():
    backend = SGLangPDNativeBenchBackend(bootstrap_port=9100)
    payload = backend._build_request_payload(prompt=[1], model="m", osl=1)
    assert payload["bootstrap_port"] == 9100


@pytest.mark.asyncio
async def test_pd_native_bench_prefill_forces_max_tokens_one(endpoint):
    """The prefill sweep must benchmark with osl=1 regardless of the caller's osl."""
    backend = SGLangPDNativeBenchBackend()
    backend.run_benchmark = AsyncMock(return_value=_metrics())

    await backend.bench_prefill(
        endpoint, model="m", isl_points=[128, 256], num_prompts=2, osl=5,
    )

    assert backend.run_benchmark.await_count == 2
    for call in backend.run_benchmark.await_args_list:
        # run_benchmark(endpoint, model, num_prompts, isl, osl, request_rate)
        assert call.args[4] == 1
    assert backend._phase == "prefill"


@pytest.mark.asyncio
async def test_pd_native_bench_decode_sets_decode_phase(endpoint):
    backend = SGLangPDNativeBenchBackend()
    backend.run_benchmark = AsyncMock(return_value=_metrics())

    await backend.bench_decode(
        endpoint, model="m", isl_points=[256], max_kv_tokens=32768, osl=100,
    )

    assert backend.run_benchmark.await_count >= 1
    assert backend._phase == "decode"
    # The payload built during the decode sweep carries the fake-prefill fields.
    payload = backend._build_request_payload(prompt=[1], model="m", osl=100)
    assert payload["bootstrap_room"] == 0


@pytest.mark.asyncio
async def test_pd_native_bench_prefill_payload_carries_bootstrap_fields(endpoint):
    """End-to-end: payloads built inside the prefill sweep have bootstrap fields."""
    captured = []

    class _CaptureBackend(SGLangPDNativeBenchBackend):
        async def run_benchmark(self, endpoint, model, num_prompts, isl, osl,
                                request_rate=float("inf")):
            captured.append(
                self._build_request_payload(prompt=[1] * isl, model=model, osl=osl)
            )
            return _metrics()

    backend = _CaptureBackend()
    await backend.bench_prefill(endpoint, model="m", isl_points=[128], num_prompts=1)

    assert len(captured) == 1
    assert captured[0]["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
    assert captured[0]["bootstrap_room"] != 0
    assert captured[0]["max_tokens"] == 1


def test_pd_native_send_request_passes_prompt_text():
    """The completions payload selects prompt_text in text mode like sglang-oai."""
    backend = SGLangPDNativeBenchBackend(input_format="text")
    payload = backend._build_request_payload(
        prompt=[1, 2, 3], prompt_text="hello", model="m", osl=1,
    )
    assert payload["prompt"] == "hello"
    assert payload["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
