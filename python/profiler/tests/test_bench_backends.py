# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for concrete bench backends and the registry."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from rbg_profiler.bench import BENCH_BACKENDS, create_bench_backend
from rbg_profiler.bench.base import RequestFuncInput
from rbg_profiler.bench.sglang import SGLangBenchBackend, SGLangChatBenchBackend
from rbg_profiler.bench.vllm import VLLMBenchBackend, VLLMChatBenchBackend


def test_registry_contains_all_backends():
    assert "sglang-oai" in BENCH_BACKENDS
    assert "sglang-oai-chat" in BENCH_BACKENDS
    assert "vllm" in BENCH_BACKENDS
    assert "vllm-chat" in BENCH_BACKENDS


def test_create_bench_backend_sglang():
    backend = create_bench_backend("sglang-oai")
    assert isinstance(backend, SGLangBenchBackend)
    assert backend.api_path == "/v1/completions"
    assert backend.default_port == 30000


def test_create_bench_backend_sglang_chat():
    backend = create_bench_backend("sglang-oai-chat")
    assert isinstance(backend, SGLangChatBenchBackend)
    assert backend.api_path == "/v1/chat/completions"


def test_create_bench_backend_vllm():
    backend = create_bench_backend("vllm")
    assert isinstance(backend, VLLMBenchBackend)
    assert backend.api_path == "/v1/completions"
    assert backend.default_port == 8000


def test_create_bench_backend_vllm_chat():
    backend = create_bench_backend("vllm-chat")
    assert isinstance(backend, VLLMChatBenchBackend)
    assert backend.api_path == "/v1/chat/completions"


def test_create_bench_backend_unknown():
    with pytest.raises(ValueError, match="Unknown bench backend"):
        create_bench_backend("nonexistent")


def test_sglang_build_request_payload():
    backend = SGLangBenchBackend()
    payload = backend._build_request_payload(
        prompt=[1, 2, 3], model="test-model", osl=10,
    )
    assert payload["model"] == "test-model"
    assert payload["prompt"] == [1, 2, 3]
    assert payload["max_tokens"] == 10
    assert payload["stream"] is True
    assert payload["temperature"] == 0.0
    assert payload["ignore_eos"] is True


def test_sglang_chat_build_request_payload():
    backend = SGLangChatBenchBackend()
    payload = backend._build_request_payload(
        prompt=[1, 2, 3], model="test-model", osl=10,
    )
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": [1, 2, 3]}]
    assert payload["max_completion_tokens"] == 10
    assert payload["stream"] is True


class _MockContent:
    """Async iterable over canned SSE bytes for mocking response.content."""

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


def _make_mock_session(sse_lines: list[bytes]):
    """Create a mock aiohttp session that yields canned SSE lines."""
    response = MagicMock()
    response.status = 200
    response.content = _MockContent(sse_lines)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.post = MagicMock(return_value=response)
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_sglang_send_request_parses_sse():
    """_send_request correctly parses SSE stream and extracts TTFT/ITL."""
    from rbg_profiler.bench.sglang import SGLangBenchBackend
    from rbg_profiler.bench.base import RequestFuncInput

    sse_lines = [
        b'data: {"choices": [{"text": "Hello"}]}\n\n',
        b'data: {"choices": [{"text": " world"}]}\n\n',
        b'data: {"usage": {"completion_tokens": 2}}\n\n',
        b'data: [DONE]\n\n',
    ]
    session = _make_mock_session(sse_lines)

    backend = SGLangBenchBackend()
    request = RequestFuncInput(
        prompt=[1, 2, 3], prompt_len=3, output_len=10,
        model="test-model", api_url="http://test:8000/v1/completions",
        is_chat=False,
    )
    output = await backend._send_request(session, request)

    assert output.success is True
    assert output.ttft > 0
    assert len(output.itl) == 1  # 2 content chunks: first is TTFT, second is ITL
    assert output.output_len == 2
    assert output.latency > 0
