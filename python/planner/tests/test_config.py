# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for planner config metrics engine derivation."""

import os
from unittest.mock import patch

from rbg_planner.config import PlannerConfig


def test_metric_engine_from_backend_sglang():
    with patch.dict(os.environ, {"BACKEND": "sglang-oai"}, clear=False):
        os.environ.pop("METRIC_ENGINE", None)
        config = PlannerConfig()
        assert config.metric_engine == "sglang"


def test_metric_engine_from_backend_vllm():
    with patch.dict(os.environ, {"BACKEND": "vllm-chat"}, clear=False):
        os.environ.pop("METRIC_ENGINE", None)
        config = PlannerConfig()
        assert config.metric_engine == "vllm"


def test_metric_engine_explicit_overrides_backend():
    with patch.dict(os.environ, {"BACKEND": "vllm", "METRIC_ENGINE": "sglang"}):
        config = PlannerConfig()
        assert config.metric_engine == "sglang"


def test_backend_env_var_stored():
    with patch.dict(os.environ, {"BACKEND": "sglang-oai"}):
        config = PlannerConfig()
        assert config.backend == "sglang-oai"


def test_metric_queries_json_from_env():
    with patch.dict(os.environ, {"METRIC_QUERIES_JSON": '{"ttft":"custom"}'}):
        config = PlannerConfig()
        assert config.metric_queries_json == '{"ttft":"custom"}'


def test_metric_queries_json_default_empty():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("METRIC_QUERIES_JSON", None)
        config = PlannerConfig()
        assert config.metric_queries_json == ""
