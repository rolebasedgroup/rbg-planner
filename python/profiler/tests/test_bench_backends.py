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
    assert "sglang-pd-native" in BENCH_BACKENDS
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


# --- Input format (prompt input form) tests ---


def test_input_format_auto_resolves_to_preferred_form():
    """auto resolves to the backend's preferred_input_form."""
    assert SGLangBenchBackend().input_format == "token_ids"
    assert SGLangChatBenchBackend().input_format == "text"
    assert VLLMBenchBackend().input_format == "token_ids"
    assert VLLMChatBenchBackend().input_format == "text"


def test_input_format_explicit_overrides_preferred_form():
    assert SGLangBenchBackend(input_format="text").input_format == "text"
    assert SGLangChatBenchBackend(input_format="token_ids").input_format == "token_ids"


def test_input_format_invalid_raises():
    with pytest.raises(ValueError, match="Invalid input_format"):
        SGLangBenchBackend(input_format="bogus")


def test_create_bench_backend_passes_input_format():
    backend = create_bench_backend("sglang-oai", input_format="text")
    assert isinstance(backend, SGLangBenchBackend)
    assert backend.input_format == "text"


def test_sglang_payload_uses_token_ids_by_default():
    backend = SGLangBenchBackend()
    payload = backend._build_request_payload(
        prompt=[1, 2, 3], prompt_text="hello world", model="m", osl=10,
    )
    assert payload["prompt"] == [1, 2, 3]


def test_sglang_payload_uses_text_when_input_format_text():
    backend = SGLangBenchBackend(input_format="text")
    payload = backend._build_request_payload(
        prompt=[1, 2, 3], prompt_text="hello world", model="m", osl=10,
    )
    assert payload["prompt"] == "hello world"


def test_sglang_payload_text_mode_falls_back_to_ids_without_text():
    """Direct payload calls without prompt_text keep the token IDs."""
    backend = SGLangBenchBackend(input_format="text")
    payload = backend._build_request_payload(prompt=[1, 2, 3], model="m", osl=10)
    assert payload["prompt"] == [1, 2, 3]


def test_sglang_chat_payload_consumes_prompt_text():
    backend = SGLangChatBenchBackend()  # auto -> text
    payload = backend._build_request_payload(
        prompt=[1, 2, 3], prompt_text="hello world", model="m", osl=10,
    )
    assert payload["messages"] == [{"role": "user", "content": "hello world"}]


def test_make_request_input_decodes_prompt_when_text():
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded prompt"

    backend = SGLangBenchBackend(input_format="text")
    req = backend._make_request_input(
        prompt=[1, 2, 3], tokenizer=tokenizer, model="m",
        api_url="http://t:8000/v1/completions", is_chat=False, output_len=10,
    )
    tokenizer.decode.assert_called_once_with([1, 2, 3])
    assert req.prompt == [1, 2, 3]
    assert req.prompt_text == "decoded prompt"
    assert req.prompt_len == 3


def test_make_request_input_keeps_prompt_ids_when_token_ids():
    tokenizer = MagicMock()

    backend = SGLangBenchBackend()  # auto -> token_ids
    req = backend._make_request_input(
        prompt=[1, 2, 3], tokenizer=tokenizer, model="m",
        api_url="http://t:8000/v1/completions", is_chat=False, output_len=10,
    )
    tokenizer.decode.assert_not_called()
    assert req.prompt == [1, 2, 3]
    assert req.prompt_text == ""


@pytest.mark.asyncio
async def test_run_benchmark_builds_text_requests():
    """run_benchmark decodes prompts when the resolved input format is text."""
    from rbg_profiler.bench.base import (
        BaseBenchBackend,
        RequestFuncInput,
        RequestFuncOutput,
    )
    from rbg_profiler.rbg_discoverer import RoleEndpoint

    tokenizer = MagicMock()
    tokenizer.vocab_size = 100
    tokenizer.decode.side_effect = lambda ids: f"text-{len(ids)}"

    captured_payloads = []

    class _TextCaptureBackend(SGLangBenchBackend):
        async def _send_request(self, session, request):
            captured_payloads.append(
                self._build_request_payload(
                    prompt=request.prompt,
                    prompt_text=request.prompt_text,
                    model=request.model,
                    osl=request.output_len,
                )
            )
            return RequestFuncOutput(
                success=True, ttft=0.05, itl=[0.02], latency=0.1, output_len=3,
            )

    backend = _TextCaptureBackend(input_format="text")
    # Inject the mock tokenizer so run_benchmark does not load from HF.
    _TextCaptureBackend._tokenizer_cache["mock-tok"] = tokenizer

    endpoint = RoleEndpoint(
        base_url="http://test:8000", num_gpus=1, tokenizer="mock-tok",
    )
    await backend.run_benchmark(endpoint, "m", num_prompts=2, isl=4, osl=3)

    assert len(captured_payloads) == 3  # warmup + 2 requests
    for payload in captured_payloads:
        assert payload["prompt"] == "text-4"
        assert payload["max_tokens"] == 3
