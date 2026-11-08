# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable bench backend module.

Provides a unified BenchBackend interface with concrete backends for
each supported inference engine (SGLang, vLLM).

Usage::

    from rbg_profiler.bench import create_bench_backend

    backend = create_bench_backend("sglang-oai")
    metrics = await backend.run_benchmark(endpoint, model, 5, 1024, 5)
"""

from rbg_profiler.bench.base import BaseBenchBackend, BenchBackend
from rbg_profiler.bench.sglang import SGLangBenchBackend, SGLangChatBenchBackend
from rbg_profiler.bench.vllm import VLLMBenchBackend, VLLMChatBenchBackend

BENCH_BACKENDS: dict[str, type[BenchBackend]] = {
    "sglang-oai": SGLangBenchBackend,
    "sglang-oai-chat": SGLangChatBenchBackend,
    "vllm": VLLMBenchBackend,
    "vllm-chat": VLLMChatBenchBackend,
}


def create_bench_backend(name: str) -> BenchBackend:
    """Create a bench backend for the given name.

    Args:
        name: Backend identifier (e.g. "sglang-oai", "vllm-chat").

    Raises:
        ValueError: If the name is not a registered backend.
    """
    backend_cls = BENCH_BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown bench backend: {name}. "
            f"Available: {list(BENCH_BACKENDS.keys())}"
        )
    return backend_cls()
