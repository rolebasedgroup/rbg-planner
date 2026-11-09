# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bench backend abstraction for pluggable inference profiling.

Defines the BenchBackend interface and standard data structures.
Concrete backends inherit BaseBenchBackend (in this module) or
BenchBackend directly for fully custom bench logic.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator

import aiohttp
import numpy as np

from rbg_profiler.rbg_discoverer import RoleEndpoint

logger = logging.getLogger(__name__)


class ContextLengthExceededError(Exception):
    """Raised when the server returns HTTP 400 (e.g. ISL+OSL exceeds max-model-len).

    A 400 is a permanent rejection: all larger ISLs will also fail, so the
    caller should stop the sweep immediately.
    """


@dataclass
class BenchMetrics:
    """Standard metrics from a single benchmark run."""

    mean_ttft_ms: float
    mean_itl_ms: float
    mean_tpot_ms: float
    request_throughput: float
    output_throughput: float
    mean_e2e_latency_ms: float


@dataclass
class PrefillProfileData:
    """Profiling data point for prefill performance."""

    isl: int
    ttft_ms: float
    thpt_per_gpu: float


@dataclass
class DecodeProfileData:
    """Profiling data point for decode performance."""

    isl: int
    concurrency: int
    kv_usage: float
    context_length: int
    itl_ms: float
    thpt_per_gpu: float


@dataclass
class UnifiedProfileData:
    """Profiling data point for unified (non-PD) deployment."""

    isl: int
    concurrency: int
    ttft_ms: float
    itl_ms: float
    thpt_per_gpu: float


class BenchBackend(ABC):
    """Pluggable bench backend for profiling inference services.

    Subclasses must implement run_benchmark(). The high-level methods
    (bench_prefill, bench_decode, bench_unified) have default
    implementations that sweep parameter grids and call run_benchmark.
    Override them for fully custom profiling flows.
    """

    @abstractmethod
    async def run_benchmark(
        self,
        endpoint: RoleEndpoint,
        model: str,
        num_prompts: int,
        isl: int,
        osl: int,
        request_rate: float = float("inf"),
    ) -> BenchMetrics:
        """Send num_prompts concurrent requests and return standard metrics."""
        raise NotImplementedError

    async def bench_prefill(
        self,
        endpoint: RoleEndpoint,
        model: str,
        isl_points: list[int],
        num_prompts: int = 5,
        osl: int = 5,
        max_context_length: int = 32768,
    ) -> list[PrefillProfileData]:
        """Profile prefill across ISL points.

        Default: sweep ISL, low concurrency, minimal OSL to isolate prefill time.
        """
        results: list[PrefillProfileData] = []
        for isl in isl_points:
            if isl >= max_context_length - 512:
                logger.info("Skipping ISL=%d (exceeds max_context_length)", isl)
                continue
            try:
                metrics = await self.run_benchmark(
                    endpoint, model, num_prompts, isl, osl, float("inf")
                )
            except ContextLengthExceededError:
                logger.info(
                    "ISL=%d rejected with HTTP 400, stopping prefill sweep", isl,
                )
                break
            except Exception:
                logger.exception("ISL=%d: benchmark failed, skipping", isl)
                continue
            if not (metrics.mean_ttft_ms > 0) or endpoint.num_gpus <= 0:
                logger.warning(
                    "ISL=%d: invalid metrics (ttft=%s, gpus=%d), skipping",
                    isl, metrics.mean_ttft_ms, endpoint.num_gpus,
                )
                continue
            thpt = isl / metrics.mean_ttft_ms * 1000 / endpoint.num_gpus
            results.append(
                PrefillProfileData(
                    isl=isl, ttft_ms=metrics.mean_ttft_ms, thpt_per_gpu=round(thpt, 2)
                )
            )
            logger.info(
                "ISL=%d: TTFT=%.2fms, thpt/gpu=%.1f tok/s",
                isl, metrics.mean_ttft_ms, thpt,
            )
        return results

    async def bench_decode(
        self,
        endpoint: RoleEndpoint,
        model: str,
        isl_points: list[int],
        max_kv_tokens: int,
        osl: int = 500,
        max_context_length: int = 32768,
        concurrency_granularity: int = 6,
    ) -> list[DecodeProfileData]:
        """Profile decode across ISL x concurrency grid.

        Default: sweep ISL, for each ISL sweep concurrency from 1 to max.
        """
        results: list[DecodeProfileData] = []
        for isl in isl_points:
            if isl + osl >= max_context_length:
                logger.info("Skipping ISL=%d (isl+osl exceeds context length)", isl)
                continue
            context_length = isl + osl // 2
            max_concurrency = min(max_kv_tokens // (isl + osl), 512)
            if max_concurrency < 1:
                logger.warning("ISL=%d: max_concurrency=0, skipping", isl)
                continue
            if max_concurrency <= concurrency_granularity:
                concurrency_points = list(range(1, max_concurrency + 1))
            else:
                step = (max_concurrency - 1) / (concurrency_granularity - 1)
                concurrency_points = sorted(
                    set(max(1, int(1 + i * step)) for i in range(concurrency_granularity))
                )
            logger.info(
                "ISL=%d: context_len=%d, max_conc=%d, sweep=%s",
                isl, context_length, max_concurrency, concurrency_points,
            )
            stop_sweep = False
            for conc in concurrency_points:
                try:
                    metrics = await self.run_benchmark(
                        endpoint, model, conc, isl, osl, float("inf")
                    )
                except ContextLengthExceededError:
                    logger.info(
                        "ISL=%d rejected with HTTP 400, stopping decode sweep", isl,
                    )
                    stop_sweep = True
                    break
                except Exception:
                    logger.exception("ISL=%d conc=%d: benchmark failed, skipping", isl, conc)
                    continue
                if endpoint.num_gpus <= 0:
                    logger.warning(
                        "ISL=%d conc=%d: invalid num_gpus, skipping", isl, conc,
                    )
                    continue
                kv_usage = (isl + osl / 2) * conc / max_kv_tokens
                thpt_per_gpu = metrics.output_throughput / endpoint.num_gpus
                results.append(
                    DecodeProfileData(
                        isl=isl,
                        concurrency=conc,
                        kv_usage=round(kv_usage, 6),
                        context_length=context_length,
                        itl_ms=round(metrics.mean_itl_ms, 3),
                        thpt_per_gpu=round(thpt_per_gpu, 2),
                    )
                )
                logger.info(
                    "  conc=%d: kv_usage=%.4f, ITL=%.2fms, thpt/gpu=%.1f tok/s",
                    conc, kv_usage, metrics.mean_itl_ms, thpt_per_gpu,
                )
            if stop_sweep:
                break
        return results

    async def bench_unified(
        self,
        endpoint: RoleEndpoint,
        model: str,
        isl_points: list[int],
        concurrency_grid: list[int],
        osl: int = 500,
    ) -> list[UnifiedProfileData]:
        """Profile unified prefill+decode.

        Default: sweep ISL x concurrency, track both TTFT and ITL.
        """
        results: list[UnifiedProfileData] = []
        for isl in isl_points:
            stop_sweep = False
            for conc in concurrency_grid:
                try:
                    metrics = await self.run_benchmark(
                        endpoint, model, conc, isl, osl, float("inf")
                    )
                except ContextLengthExceededError:
                    logger.info(
                        "ISL=%d rejected with HTTP 400, stopping unified sweep", isl,
                    )
                    stop_sweep = True
                    break
                if endpoint.num_gpus <= 0:
                    logger.warning(
                        "ISL=%d conc=%d: invalid num_gpus, skipping", isl, conc,
                    )
                    continue
                thpt_per_gpu = metrics.output_throughput / endpoint.num_gpus
                results.append(
                    UnifiedProfileData(
                        isl=isl,
                        concurrency=conc,
                        ttft_ms=round(metrics.mean_ttft_ms, 2),
                        itl_ms=round(metrics.mean_itl_ms, 3),
                        thpt_per_gpu=round(thpt_per_gpu, 2),
                    )
                )
            if stop_sweep:
                break
        return results


@dataclass
class RequestFuncInput:
    prompt: list[int]
    prompt_len: int
    output_len: int
    model: str
    api_url: str
    is_chat: bool
    # Decoded prompt text. Populated when the resolved input format is "text";
    # empty otherwise. ``prompt`` always holds the token IDs.
    prompt_text: str = ""


@dataclass
class RequestFuncOutput:
    success: bool = False
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    latency: float = 0.0
    output_len: int = 0
    error: str = ""
    status_code: int = 0


def generate_random_input_ids(tokenizer, input_len: int) -> list[int]:
    """Generate random token IDs as prompt."""
    vocab_size = tokenizer.vocab_size
    return np.random.randint(0, vocab_size, size=input_len).tolist()


async def get_request(
    input_requests: list[RequestFuncInput],
    request_rate: float,
) -> AsyncGenerator[RequestFuncInput, None]:
    """Yield requests at the specified rate."""
    if request_rate == float("inf"):
        for req in input_requests:
            yield req
    else:
        for req in input_requests:
            yield req
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)


def calculate_metrics(outputs: list[RequestFuncOutput], dur_s: float) -> dict:
    """Calculate benchmark metrics from request outputs."""
    completed = 0
    ttfts = []
    itls = []
    tpots = []
    e2e_latencies = []
    output_lens = []

    for out in outputs:
        if not out.success:
            continue
        completed += 1
        output_len = out.output_len
        output_lens.append(output_len)
        ttfts.append(out.ttft)
        itls.extend(out.itl)
        e2e_latencies.append(out.latency)
        if output_len > 1:
            tpots.append((out.latency - out.ttft) / (output_len - 1))

    if completed == 0:
        first_error = next((o.error for o in outputs if o.error), "")
        first_status = next((o.status_code for o in outputs if o.status_code), 0)
        return {
            "error": "all requests failed",
            "num_errors": len(outputs),
            "first_error": first_error,
            "first_status_code": first_status,
        }

    mean_ttft_ms = np.mean(ttfts) * 1000
    mean_itl_ms = np.mean(itls) * 1000 if itls else 0.0
    mean_tpot_ms = np.mean(tpots) * 1000 if tpots else 0.0
    output_throughput = sum(output_lens) / dur_s if dur_s > 0 else 0
    request_throughput = completed / dur_s if dur_s > 0 else 0

    return {
        "duration": dur_s,
        "completed": completed,
        "total_output_tokens": sum(output_lens),
        "request_throughput": request_throughput,
        "output_throughput": output_throughput,
        "mean_ttft_ms": float(mean_ttft_ms),
        "mean_itl_ms": float(mean_itl_ms),
        "mean_tpot_ms": float(mean_tpot_ms),
        "mean_e2e_latency_ms": float(np.mean(e2e_latencies) * 1000),
        "num_errors": len(outputs) - completed,
    }


def _create_bench_client_session() -> aiohttp.ClientSession:
    """Create an aiohttp session with generous timeout for benchmarking."""
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    return aiohttp.ClientSession(
        timeout=timeout,
        read_bufsize=10 * 1024 ** 2,
    )


def _remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


class BaseBenchBackend(BenchBackend):
    """Shared benchmark implementation using HTTP streaming requests.

    Subclasses must override:
      - api_path: str (e.g. "/v1/completions")
      - default_port: int
      - _build_request_payload(): construct the request JSON payload
      - _send_request(): send a single streaming request and collect timing

    Input format:
      ``input_format`` controls the prompt form sent to the server:
      "token_ids" sends the raw token ID array, "text" sends the
      tokenizer-decoded string (required by routers that reject token ID
      prompts, e.g. SMG). "auto" (default) resolves to the backend's
      ``preferred_input_form``.
    """

    api_path: str = "/v1/completions"
    default_port: int = 30000

    #: Prompt form used when input_format is "auto".
    preferred_input_form: str = "token_ids"

    _tokenizer_cache: dict[str, object] = {}

    VALID_INPUT_FORMATS = ("auto", "text", "token_ids")

    def __init__(self, input_format: str = "auto"):
        if input_format not in self.VALID_INPUT_FORMATS:
            raise ValueError(
                f"Invalid input_format: {input_format!r}. "
                f"Valid values: {list(self.VALID_INPUT_FORMATS)}"
            )
        if input_format == "auto":
            input_format = self.preferred_input_form
        self.input_format = input_format

    @abstractmethod
    def _build_request_payload(
        self, prompt: list[int], model: str, osl: int, **kwargs,
    ) -> dict:
        """Build engine-specific request payload."""
        raise NotImplementedError

    @abstractmethod
    async def _send_request(
        self, session: aiohttp.ClientSession, request: RequestFuncInput,
    ) -> RequestFuncOutput:
        """Send a single streaming request and collect timing data."""
        raise NotImplementedError

    def _get_tokenizer(self, tokenizer_path: str):
        """Load and cache a tokenizer by path or HF model ID."""
        if tokenizer_path not in self._tokenizer_cache:
            from transformers import AutoTokenizer
            self._tokenizer_cache[tokenizer_path] = AutoTokenizer.from_pretrained(
                tokenizer_path
            )
        return self._tokenizer_cache[tokenizer_path]

    def _make_request_input(
        self,
        prompt: list[int],
        tokenizer,
        model: str,
        api_url: str,
        is_chat: bool,
        output_len: int,
    ) -> RequestFuncInput:
        """Build a RequestFuncInput, decoding prompt text when needed."""
        prompt_text = ""
        if self.input_format == "text":
            prompt_text = tokenizer.decode(prompt)
        return RequestFuncInput(
            prompt=prompt,
            prompt_text=prompt_text,
            prompt_len=len(prompt),
            output_len=output_len,
            model=model,
            api_url=api_url,
            is_chat=is_chat,
        )

    async def run_benchmark(
        self,
        endpoint: RoleEndpoint,
        model: str,
        num_prompts: int,
        isl: int,
        osl: int,
        request_rate: float = float("inf"),
    ) -> BenchMetrics:
        """Send num_prompts concurrent requests and return standard metrics."""
        api_url = f"{endpoint.base_url.rstrip('/')}{self.api_path}"
        is_chat = "chat" in self.api_path

        tokenizer = self._get_tokenizer(endpoint.tokenizer)

        input_requests = []
        for _ in range(num_prompts):
            prompt = generate_random_input_ids(tokenizer, isl)
            input_requests.append(self._make_request_input(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                api_url=api_url,
                is_chat=is_chat,
                output_len=osl,
            ))

        session = _create_bench_client_session()
        try:
            # Warmup with one request.
            warmup_req = self._make_request_input(
                prompt=generate_random_input_ids(tokenizer, min(isl, 128)),
                tokenizer=tokenizer,
                model=model,
                api_url=api_url,
                is_chat=is_chat,
                output_len=min(osl, 32),
            )
            warmup_output = await self._send_request(session, warmup_req)
            if not warmup_output.success:
                logger.warning("Warmup request failed: %s", warmup_output.error[:200])

            # Run benchmark.
            benchmark_start = time.perf_counter()
            tasks = []
            async for req in get_request(input_requests, request_rate):
                tasks.append(asyncio.create_task(self._send_request(session, req)))

            outputs = await asyncio.gather(*tasks)
            benchmark_duration = time.perf_counter() - benchmark_start
        finally:
            await session.close()

        result = calculate_metrics(outputs, benchmark_duration)
        if "error" in result:
            status = result.get("first_status_code", 0)
            if status == 400:
                raise ContextLengthExceededError()
            if status == 429:
                raise RuntimeError(
                    f"Benchmark failed: rate limited (HTTP 429)"
                )
            raise RuntimeError(f"Benchmark failed: {result['error']}")

        return BenchMetrics(
            mean_ttft_ms=result["mean_ttft_ms"],
            mean_itl_ms=result["mean_itl_ms"],
            mean_tpot_ms=result["mean_tpot_ms"],
            request_throughput=result["request_throughput"],
            output_throughput=result["output_throughput"],
            mean_e2e_latency_ms=result["mean_e2e_latency_ms"],
        )
