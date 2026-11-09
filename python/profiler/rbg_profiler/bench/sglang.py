# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang bench backends for OpenAI-compatible API profiling."""

import json
import sys
import time
import traceback

import aiohttp

from rbg_profiler.bench.base import (
    BaseBenchBackend,
    RequestFuncInput,
    RequestFuncOutput,
    _remove_prefix,
)


class SGLangBenchBackend(BaseBenchBackend):
    """Bench backend for SGLang via OpenAI completions API."""

    api_path = "/v1/completions"
    default_port = 30000

    def _build_request_payload(self, prompt, model, osl, prompt_text="", **kwargs):
        # Send the decoded text when the resolved input format is "text"
        # (e.g. behind routers that only accept string prompts); otherwise
        # send the raw token ID array.
        if self.input_format == "text" and prompt_text:
            prompt_value = prompt_text
        else:
            prompt_value = prompt
        return {
            "model": model,
            "prompt": prompt_value,
            "best_of": 1,
            "max_tokens": osl,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
            "ignore_eos": True,
        }

    async def _send_request(
        self, session: aiohttp.ClientSession, request: RequestFuncInput,
    ) -> RequestFuncOutput:
        output = RequestFuncOutput()
        payload = self._build_request_payload(
            prompt=request.prompt,
            prompt_text=request.prompt_text,
            model=request.model,
            osl=request.output_len,
        )
        st = time.perf_counter()
        most_recent_timestamp = st

        try:
            async with session.post(request.api_url, json=payload) as response:
                if response.status != 200:
                    output.status_code = response.status
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
        output.success = output.ttft > 0
        return output


class SGLangChatBenchBackend(BaseBenchBackend):
    """Bench backend for SGLang via OpenAI chat completions API."""

    api_path = "/v1/chat/completions"
    default_port = 30000
    # Chat APIs take string content, so prefer decoded text prompts.
    preferred_input_form = "text"

    def _build_request_payload(self, prompt, model, osl, prompt_text="", **kwargs):
        # Chat payloads are strings by construction; consume prompt_text when
        # available and fall back to the raw prompt value otherwise.
        content = prompt_text if (self.input_format == "text" and prompt_text) else prompt
        return {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": osl,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
            "ignore_eos": True,
        }

    async def _send_request(
        self, session: aiohttp.ClientSession, request: RequestFuncInput,
    ) -> RequestFuncOutput:
        output = RequestFuncOutput()
        payload = self._build_request_payload(
            prompt=request.prompt,
            prompt_text=request.prompt_text,
            model=request.model,
            osl=request.output_len,
        )
        st = time.perf_counter()
        most_recent_timestamp = st

        try:
            async with session.post(request.api_url, json=payload) as response:
                if response.status != 200:
                    output.status_code = response.status
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
        output.success = output.ttft > 0
        return output
