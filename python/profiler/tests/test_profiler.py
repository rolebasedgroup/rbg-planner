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
    mock_backend.bench_prefill = AsyncMock(
        return_value=[
            PrefillProfileData(isl=128, ttft_ms=50.0, thpt_per_gpu=2560.0),
            PrefillProfileData(isl=256, ttft_ms=80.0, thpt_per_gpu=3200.0),
        ]
    )
    mock_backend.bench_decode = AsyncMock(
        return_value=[
            DecodeProfileData(
                isl=256,
                concurrency=1,
                kv_usage=0.5,
                context_length=506,
                itl_ms=15.0,
                thpt_per_gpu=500.0,
            ),
        ]
    )

    with patch("rbg_profiler.profiler.create_bench_backend", return_value=mock_backend):
        await run_profiling(
            mock_args, mock_endpoint, mock_endpoint, backend="sglang-oai"
        )

    mock_backend.bench_prefill.assert_called_once()
    mock_backend.bench_decode.assert_called_once()


@pytest.mark.asyncio
async def test_run_profiling_passes_input_format(
    mock_endpoint, mock_args, mock_metrics
):
    """run_profiling forwards input_format to create_bench_backend."""
    from rbg_profiler.profiler import run_profiling

    mock_backend = MagicMock()
    mock_backend.bench_prefill = AsyncMock(return_value=[])
    mock_backend.bench_decode = AsyncMock(return_value=[])

    with patch(
        "rbg_profiler.profiler.create_bench_backend", return_value=mock_backend
    ) as mock_create:
        await run_profiling(
            mock_args,
            mock_endpoint,
            mock_endpoint,
            backend="sglang-oai",
            input_format="text",
        )

    mock_create.assert_called_once_with("sglang-oai", input_format="text")


@pytest.mark.asyncio
async def test_run_profiling_passes_bootstrap_port_for_pd_native(
    mock_endpoint, mock_args
):
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
            mock_args,
            mock_endpoint,
            mock_endpoint,
            backend="sglang-pd-native",
            input_format="auto",
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
            mock_args,
            mock_endpoint,
            mock_endpoint,
            backend="sglang-oai",
            input_format="auto",
        )

    mock_create.assert_called_once_with("sglang-oai", input_format="auto")


def test_parse_result_labels():
    """parse_result_labels converts k=v CLI args to a label dict."""
    from rbg_profiler.profiler import parse_result_labels

    assert parse_result_labels(["a=b", "c=d"]) == {"a": "b", "c": "d"}
    assert parse_result_labels(None) is None
    assert parse_result_labels([]) is None

    with pytest.raises(ValueError):
        parse_result_labels(["not-a-pair"])


def _make_api_exception(status):
    from kubernetes import client as k8s_client

    return k8s_client.ApiException(status=status)


def test_update_configmap_create_stamps_labels():
    """A created profiling ConfigMap carries the managed labels."""
    from rbg_profiler import profiler

    v1 = MagicMock()
    v1.read_namespaced_config_map.side_effect = _make_api_exception(404)
    with (
        patch("rbg_profiler.profiler.config"),
        patch("rbg_profiler.profiler.client.CoreV1Api", return_value=v1),
    ):
        profiler._update_configmap(
            "test-cm",
            "default",
            {},
            {},
            labels={"rbg-planner.io/managed": "true"},
        )

    v1.create_namespaced_config_map.assert_called_once()
    body = v1.create_namespaced_config_map.call_args.kwargs["body"]
    assert body.metadata.labels == {"rbg-planner.io/managed": "true"}


def test_update_configmap_update_merges_labels():
    """Replacing an existing ConfigMap preserves and merges its labels."""
    from rbg_profiler import profiler

    existing = MagicMock()
    existing.metadata.labels = {"team": "ai"}
    v1 = MagicMock()
    v1.read_namespaced_config_map.return_value = existing
    with (
        patch("rbg_profiler.profiler.config"),
        patch("rbg_profiler.profiler.client.CoreV1Api", return_value=v1),
    ):
        profiler._update_configmap(
            "test-cm",
            "default",
            {},
            {},
            labels={"rbg-planner.io/managed": "true"},
        )

    v1.create_namespaced_config_map.assert_not_called()
    v1.replace_namespaced_config_map.assert_called_once()
    body = v1.replace_namespaced_config_map.call_args.kwargs["body"]
    assert body.metadata.labels == {
        "team": "ai",
        "rbg-planner.io/managed": "true",
    }


def test_update_configmap_no_labels_omits_labels():
    """Without managed labels the ConfigMap metadata stays label-free."""
    from rbg_profiler import profiler

    v1 = MagicMock()
    v1.read_namespaced_config_map.side_effect = _make_api_exception(404)
    with (
        patch("rbg_profiler.profiler.config"),
        patch("rbg_profiler.profiler.client.CoreV1Api", return_value=v1),
    ):
        profiler._update_configmap("test-cm", "default", {}, {})

    body = v1.create_namespaced_config_map.call_args.kwargs["body"]
    assert body.metadata.labels is None


@pytest.mark.asyncio
async def test_run_profiling_passes_result_labels(mock_endpoint, mock_args):
    """run_profiling forwards parsed result labels to _update_configmap."""
    from rbg_profiler.profiler import run_profiling

    mock_args.update_configmap = True
    mock_args.result_labels = ["rbg-planner.io/managed=true"]

    mock_backend = MagicMock()
    mock_backend.bench_prefill = AsyncMock(return_value=[])
    mock_backend.bench_decode = AsyncMock(return_value=[])

    with (
        patch("rbg_profiler.profiler.create_bench_backend", return_value=mock_backend),
        patch("rbg_profiler.profiler._update_configmap") as mock_update,
    ):
        await run_profiling(mock_args, mock_endpoint, mock_endpoint)

    mock_update.assert_called_once_with(
        "test-cm",
        "default",
        mock_update.call_args.args[2],
        mock_update.call_args.args[3],
        labels={"rbg-planner.io/managed": "true"},
    )
