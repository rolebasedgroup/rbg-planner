# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/benchmark/serving.py

"""Self-contained benchmark serving script for OpenAI-compatible endpoints.

Ports the core logic from sglang v0.5.16's benchmark/serving.py without the
full sglang runtime dependency tree. Supports OpenAI-compatible completions
and chat/completions endpoints, SSE streaming for TTFT/ITL measurement, and
outputs the same JSON result fields the profiler already parses.

Only supports the random-ids dataset (random token IDs as input).
"""

import warnings

warnings.warn(
    "rbg_profiler.bench_serving is deprecated. "
    "Use rbg_profiler.bench.create_bench_backend() instead.",
    DeprecationWarning,
    stacklevel=2,
)

# TODO: This module should delegate its request functions to
# rbg_profiler.bench.sglang instead of duplicating the logic.
# The duplication is retained for now to avoid breaking the CLI
# entry point during the transition period.

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import AsyncGenerator

import aiohttp
import numpy as np
from transformers import AutoTokenizer

_BACKEND_API_PATHS = {
    "sglang": "/generate",
    "sglang-native": "/generate",
    "sglang-oai": "/v1/completions",
    "sglang-oai-chat": "/v1/chat/completions",
    "vllm": "/v1/completions",
    "vllm-chat": "/v1/chat/completions",
    "lmdeploy": "/v1/completions",
    "lmdeploy-chat": "/v1/chat/completions",
}

DEFAULT_PORTS = {
    # Keep in sync with ENGINE_DEFAULT_PORTS in rbg_discoverer.py.
    "sglang-pd-native": 8000,
    "sglang": 30000,
    "sglang-native": 30000,
    "sglang-oai": 30000,
    "sglang-oai-chat": 30000,
    "vllm": 8000,
    "vllm-chat": 8000,
    "lmdeploy": 23333,
    "lmdeploy-chat": 23333,
}


@dataclass
class RequestFuncInput:
    prompt: list[int]
    prompt_len: int
    output_len: int
    model: str
    api_url: str
    is_chat: bool


@dataclass
class RequestFuncOutput:
    success: bool = False
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    latency: float = 0.0
    output_len: int = 0
    error: str = ""


def _create_bench_client_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    return aiohttp.ClientSession(
        timeout=timeout,
        read_bufsize=10 * 1024 ** 2,
    )


def _remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


async def async_request_openai_completions(
    session: aiohttp.ClientSession,
    request: RequestFuncInput,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    prompt = request.prompt

    payload = {
        "model": request.model,
        "prompt": prompt,
        "best_of": 1,
        "max_tokens": request.output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "ignore_eos": True,
    }

    st = time.perf_counter()
    most_recent_timestamp = st

    try:
        async with session.post(request.api_url, json=payload) as response:
            if response.status != 200:
                output.error = f"{response.reason or ''}: {(await response.text())[:500]}"
                return output

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                data = _remove_prefix(line, "data: ")
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    text_content = choices[0].get("text")
                    if text_content:
                        timestamp = time.perf_counter()
                        if output.ttft == 0.0:
                            output.ttft = timestamp - st
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)
                        most_recent_timestamp = timestamp

                usage = chunk.get("usage")
                if usage:
                    output.output_len = usage.get("completion_tokens", 0)

    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
        return output

    output.latency = time.perf_counter() - st
    if output.output_len == 0:
        output.output_len = len(output.itl) + (1 if output.ttft > 0 else 0)
    output.success = True
    return output


async def async_request_openai_chat_completions(
    session: aiohttp.ClientSession,
    request: RequestFuncInput,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    messages = [{"role": "user", "content": request.prompt}]

    payload = {
        "model": request.model,
        "messages": messages,
        "max_completion_tokens": request.output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "ignore_eos": True,
    }

    st = time.perf_counter()
    most_recent_timestamp = st

    try:
        async with session.post(request.api_url, json=payload) as response:
            if response.status != 200:
                output.error = f"{response.reason or ''}: {(await response.text())[:500]}"
                return output

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                data = _remove_prefix(line, "data: ")
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                text_content = delta.get("content")
                if text_content:
                    timestamp = time.perf_counter()
                    if output.ttft == 0.0:
                        output.ttft = timestamp - st
                    else:
                        output.itl.append(timestamp - most_recent_timestamp)
                    most_recent_timestamp = timestamp

                usage = chunk.get("usage")
                if usage:
                    output.output_len = usage.get("completion_tokens", 0)

    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
        return output

    output.latency = time.perf_counter() - st
    if output.output_len == 0:
        output.output_len = len(output.itl) + (1 if output.ttft > 0 else 0)
    output.success = True
    return output


ASYNC_REQUEST_FUNCS = {
    "sglang-oai": async_request_openai_completions,
    "sglang-oai-chat": async_request_openai_chat_completions,
    "vllm": async_request_openai_completions,
    "vllm-chat": async_request_openai_chat_completions,
    "lmdeploy": async_request_openai_completions,
    "lmdeploy-chat": async_request_openai_chat_completions,
}


def generate_random_input_ids(tokenizer, input_len: int) -> list[int]:
    vocab_size = tokenizer.vocab_size
    return np.random.randint(0, vocab_size, size=input_len).tolist()


async def get_request(
    input_requests: list[RequestFuncInput],
    request_rate: float,
) -> AsyncGenerator[RequestFuncInput, None]:
    if request_rate == float("inf"):
        for req in input_requests:
            yield req
    else:
        for req in input_requests:
            yield req
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)


def calculate_metrics(outputs: list[RequestFuncOutput], dur_s: float) -> dict:
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
        return {"error": "all requests failed", "num_errors": len(outputs)}

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


async def benchmark(
    base_url: str,
    model: str,
    tokenizer,
    backend: str,
    num_prompts: int,
    request_rate: float,
    input_len: int,
    output_len: int,
) -> dict:
    api_path = _BACKEND_API_PATHS.get(backend, "/v1/completions")
    api_url = f"{base_url.rstrip('/')}{api_path}"
    is_chat = "chat" in backend
    request_func = ASYNC_REQUEST_FUNCS.get(backend, async_request_openai_completions)

    input_requests = []
    for _ in range(num_prompts):
        prompt = generate_random_input_ids(tokenizer, input_len)
        input_requests.append(RequestFuncInput(
            prompt=prompt,
            prompt_len=input_len,
            output_len=output_len,
            model=model,
            api_url=api_url,
            is_chat=is_chat,
        ))

    session = _create_bench_client_session()

    try:
        # Warmup with one request.
        warmup_req = RequestFuncInput(
            prompt=generate_random_input_ids(tokenizer, min(input_len, 128)),
            prompt_len=min(input_len, 128),
            output_len=min(output_len, 32),
            model=model,
            api_url=api_url,
            is_chat=is_chat,
        )
        warmup_output = await request_func(session, warmup_req)
        if not warmup_output.success:
            print(f"WARNING: Warmup request failed: {warmup_output.error[:200]}")

        # Run benchmark.
        benchmark_start = time.perf_counter()
        tasks = []
        async for req in get_request(input_requests, request_rate):
            tasks.append(asyncio.create_task(request_func(session, req)))

        outputs = await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - benchmark_start
    finally:
        await session.close()

    return calculate_metrics(outputs, benchmark_duration)


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible serving")
    parser.add_argument("--backend", default="sglang-oai",
                        choices=list(_BACKEND_API_PATHS.keys()),
                        help="Backend type")
    parser.add_argument("--base-url", required=True, help="Inference endpoint URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or HF model ID")
    parser.add_argument("--dataset-name", default="random-ids", help="Dataset name (only random-ids supported)")
    parser.add_argument("--random-input-len", type=int, default=1024, help="Random input length in tokens")
    parser.add_argument("--random-output-len", type=int, default=1024, help="Random output length in tokens")
    parser.add_argument("--random-range-ratio", type=float, default=1.0, help="Random range ratio (kept for compatibility)")
    parser.add_argument("--num-prompts", type=int, required=True, help="Number of prompts to send")
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Request rate (requests/sec, inf=all at once)")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar")
    parser.add_argument("--disable-stream", action="store_true", help="Disable streaming (TTFT=total latency)")
    parser.add_argument("--disable-ignore-eos", action="store_true", help="Don't ignore EOS")
    parser.add_argument("--output-file", default=None, help="Output file path for JSON results")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    result = asyncio.run(benchmark(
        base_url=args.base_url,
        model=args.model,
        tokenizer=tokenizer,
        backend=args.backend,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
    ))

    output = json.dumps(result)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
