# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang PD-native bench backend for direct prefill/decode engine profiling.

Unlike the ``sglang-oai`` backend, which profiles through a frontend/router
endpoint, this backend connects directly to the PD-disaggregated SGLang
engines and measures each phase in isolation:

- Prefill phase (TTFT vs ISL): requests go straight to the prefill engine's
  ``/v1/completions``. SGLang engines running in PD mode reject requests
  without bootstrap fields ("Disaggregated request received without bootstrap
  room id"), so every payload carries a client-minted bootstrap room plus
  bootstrap host/port fields. The room is minted client-side
  (``random.randint(0, 2**63 - 1)``) without calling the bootstrap server.
- Decode phase (ITL vs concurrency): requests go straight to the decode
  engine using the "fake prefill" shortcut from SGLang's official
  ``bench_serving.py --fake-prefill``: ``bootstrap_host`` is set to SGLang's
  ``FAKE_BOOTSTRAP_HOST`` sentinel ("2.2.2.2") and ``bootstrap_room`` to 0.
  The decode engine then skips waiting for prefill KV and starts decoding
  immediately, yielding a pure ITL sequence.

Both phases set ``bootstrap_host`` to the fake sentinel. SGLang selects its
fake (no-op) KV sender/receiver pair purely on
``bootstrap_host == FAKE_BOOTSTRAP_HOST`` — the room value does not trigger
the fake path — and a real transfer backend would leave the request pending
until the KV transfer completes, which never happens without a decode peer.
On the prefill side the fake sender completes the request as soon as prefill
compute finishes, which is exactly the TTFT-vs-ISL signal this phase wants;
on the decode side the fake receiver skips the KV wait, keeping ITL pure.

Protocol references (sgl-project/sglang, verified v0.4.9 .. main):
- ``srt/disaggregation/mode.py``: ``FAKE_BOOTSTRAP_HOST = "2.2.2.2"``
- ``srt/disaggregation/utils.py``: ``_is_fake_transfer(req)`` checks
  ``req.bootstrap_host == FAKE_BOOTSTRAP_HOST``
- ``srt/disaggregation/decode.py`` / ``prefill.py``: fake KV receiver/sender
  are selected when the bootstrap host equals the sentinel
- ``srt/managers/scheduler.py`` (v0.4.9): disagg requests abort when
  ``bootstrap_room`` is None — hence the room is always sent, even for fake
  transfers
- ``benchmark/serving.py --fake-prefill``: sets
  ``bootstrap_host=FAKE_BOOTSTRAP_HOST, bootstrap_room=0``
"""

import random

from rbg_profiler.bench.sglang import SGLangBenchBackend

# Sentinel bootstrap host that selects SGLang's fake (no-op) KV transfer path.
# Must stay in sync with sglang.srt.disaggregation.mode.FAKE_BOOTSTRAP_HOST.
FAKE_BOOTSTRAP_HOST = "2.2.2.2"

# Default for the prefill engine's --disaggregation-bootstrap-port.
DEFAULT_BOOTSTRAP_PORT = 8998


class SGLangPDNativeBenchBackend(SGLangBenchBackend):
    """Bench backend for PD-disaggregated SGLang engines, hit directly.

    Requests are sent engine-to-engine without the router: bench_prefill
    targets the prefill engine and bench_decode targets the decode engine.
    Both phases attach SGLang disaggregation bootstrap fields to the payload
    because PD-mode engines reject requests without a bootstrap room.
    """

    api_path = "/v1/completions"
    # PD engine ports (not the router port 30000); the discoverer prefers
    # each role's --port argument over this default.
    default_port = 8000

    def __init__(self, input_format: str = "auto", bootstrap_port: int = DEFAULT_BOOTSTRAP_PORT):
        super().__init__(input_format=input_format)
        self.bootstrap_port = bootstrap_port
        # Phase flag distinguishing prefill vs decode runs. The base class
        # sweeps call run_benchmark() without phase context, so the phase is
        # set by the overridden bench_prefill/bench_decode entry points.
        self._phase = "prefill"

    async def bench_prefill(
        self,
        endpoint,
        model: str,
        isl_points: list[int],
        num_prompts: int = 5,
        osl: int = 1,
        max_context_length: int = 32768,
    ) -> list:
        """Profile prefill across ISL points against the prefill engine.

        Overrides the base sweep to force max_tokens=1: with no decode peer
        the measurement of interest is the time from request to response,
        i.e. pure prefill compute.
        """
        self._phase = "prefill"
        return await super().bench_prefill(
            endpoint,
            model,
            isl_points,
            num_prompts=num_prompts,
            osl=1,
            max_context_length=max_context_length,
        )

    async def bench_decode(
        self,
        endpoint,
        model: str,
        isl_points: list[int],
        max_kv_tokens: int,
        osl: int = 500,
        max_context_length: int = 32768,
        concurrency_granularity: int = 6,
    ) -> list:
        """Profile decode across the ISL x concurrency grid on the decode engine.

        Uses the fake-prefill shortcut so the decode engine does not wait for
        prefill KV; only the ITL sequence and throughput are consumed.
        """
        self._phase = "decode"
        return await super().bench_decode(
            endpoint,
            model,
            isl_points,
            max_kv_tokens,
            osl=osl,
            max_context_length=max_context_length,
            concurrency_granularity=concurrency_granularity,
        )

    def _build_request_payload(self, prompt, model, osl, prompt_text="", **kwargs):
        payload = super()._build_request_payload(
            prompt=prompt, prompt_text=prompt_text, model=model, osl=osl,
        )
        payload.update(self._bootstrap_fields())
        return payload

    def _bootstrap_fields(self) -> dict:
        """Attach SGLang PD disaggregation bootstrap fields to a payload.

        Prefill phase: mint a fresh room per request (client-side, no
        bootstrap-server round trip). The fake sender ignores the room and
        completes the request once prefill compute finishes, so the minted
        room does not affect the TTFT measurement.

        Decode phase: use the official fake-prefill sentinel values
        (room 0) so the decode engine skips waiting for transferred KV.
        """
        if self._phase == "decode":
            bootstrap_room = 0
        else:
            bootstrap_room = random.randint(0, 2**63 - 1)
        return {
            "bootstrap_host": FAKE_BOOTSTRAP_HOST,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }
