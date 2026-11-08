# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM bench backends for OpenAI-compatible API profiling.

vLLM exposes the same OpenAI-compatible API as SGLang, so these
backends reuse SGLang's request/parsing logic with vLLM's default port.
"""

from rbg_profiler.bench.sglang import SGLangBenchBackend, SGLangChatBenchBackend


class VLLMBenchBackend(SGLangBenchBackend):
    """Bench backend for vLLM via OpenAI completions API."""

    api_path = "/v1/completions"
    default_port = 8000


class VLLMChatBenchBackend(SGLangChatBenchBackend):
    """Bench backend for vLLM via OpenAI chat completions API."""

    api_path = "/v1/chat/completions"
    default_port = 8000
