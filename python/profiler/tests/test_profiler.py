# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiler integration with bench backends."""

import asyncio
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rbg_profiler.rbg_discoverer import RoleEndpoint
from rbg_profiler.bench.base import BenchMetrics, PrefillProfileData, DecodeProfileData


@pytest.fixture
def mock_endpoint():
    return RoleEndpoint(
        base_url="http://test:8000",
        num_gpus=1,
        tokenizer="test-tokenizer",
    )


@pytest.fixture
def mock_args():
    return Namespace(
        model_name="test-model",
        max_kv_tokens=32768,
        max_context_length=32768,
        output_dir="/tmp/test_profiling",
        prefill_isl_points="128,256",
        decode_isl_points="256",
        decode_granularity=3,
        output_configmap="test-cm",
        namespace="default",
        update_configmap=False,
    )


@pytest.fixture
def mock_metrics():
    return BenchMetrics(
        mean_ttft_ms=50.0,
        mean_itl_ms=15.0,
        mean_tpot_ms=15.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    )


@pytest.mark.asyncio
async def test_profile_prefill_uses_backend(mock_endpoint, mock_args, mock_metrics):
    """profile_prefill should call backend.bench_prefill, not subprocess."""
    from rbg_profiler.profiler import run_profiling

    mock_backend = MagicMock()
    mock_backend.bench_prefill = AsyncMock(return_value=[
        PrefillProfileData(isl=128, ttft_ms=50.0, thpt_per_gpu=2560.0),
        PrefillProfileData(isl=256, ttft_ms=80.0, thpt_per_gpu=3200.0),
    ])
    mock_backend.bench_decode = AsyncMock(return_value=[
        DecodeProfileData(
            isl=256, concurrency=1, kv_usage=0.5,
            context_length=506, itl_ms=15.0, thpt_per_gpu=500.0,
        ),
    ])

    with patch("rbg_profiler.profiler.create_bench_backend", return_value=mock_backend):
        await run_profiling(
            mock_args, mock_endpoint, mock_endpoint, backend="sglang-oai"
        )

    mock_backend.bench_prefill.assert_called_once()
    mock_backend.bench_decode.assert_called_once()


@pytest.mark.asyncio
async def test_run_profiling_passes_input_format(mock_endpoint, mock_args, mock_metrics):
    """run_profiling forwards input_format to create_bench_backend."""
    from rbg_profiler.profiler import run_profiling

    mock_backend = MagicMock()
    mock_backend.bench_prefill = AsyncMock(return_value=[])
    mock_backend.bench_decode = AsyncMock(return_value=[])

    with patch(
        "rbg_profiler.profiler.create_bench_backend", return_value=mock_backend
    ) as mock_create:
        await run_profiling(
            mock_args, mock_endpoint, mock_endpoint,
            backend="sglang-oai", input_format="text",
        )

    mock_create.assert_called_once_with(
        "sglang-oai", input_format="text"
    )


@pytest.mark.asyncio
async def test_run_profiling_passes_bootstrap_port_for_pd_native(mock_endpoint, mock_args):
    """sglang-pd-native receives --pd-bootstrap-port via constructor kwargs."""
    from rbg_profiler.profiler import run_profiling

    mock_backend = MagicMock()
    mock_backend.bench_prefill = AsyncMock(return_value=[])
    mock_backend.bench_decode = AsyncMock(return_value=[])
    mock_args.pd_bootstrap_port = 9100

    with patch(
        "rbg_profiler.profiler.create_bench_backend", return_value=mock_backend
    ) as mock_create:
        await run_profiling(
            mock_args, mock_endpoint, mock_endpoint,
            backend="sglang-pd-native", input_format="auto",
        )

    mock_create.assert_called_once_with(
        "sglang-pd-native", input_format="auto", bootstrap_port=9100
    )


@pytest.mark.asyncio
async def test_run_profiling_no_backend_kwargs_for_non_pd(mock_endpoint, mock_args):
    """Non-PD backends do not receive PD-specific constructor kwargs."""
    from rbg_profiler.profiler import run_profiling

    mock_backend = MagicMock()
    mock_backend.bench_prefill = AsyncMock(return_value=[])
    mock_backend.bench_decode = AsyncMock(return_value=[])
    mock_args.pd_bootstrap_port = 9100

    with patch(
        "rbg_profiler.profiler.create_bench_backend", return_value=mock_backend
    ) as mock_create:
        await run_profiling(
            mock_args, mock_endpoint, mock_endpoint,
            backend="sglang-oai", input_format="auto",
        )

    mock_create.assert_called_once_with(
        "sglang-oai", input_format="auto"
    )
