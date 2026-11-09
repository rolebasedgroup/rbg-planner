# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for planner config metric source derivation."""

import os
from unittest.mock import patch

from rbg_planner.config import PlannerConfig


def test_metric_source_from_backend_sglang():
    with patch.dict(os.environ, {"BACKEND": "sglang-oai", "METRIC_SOURCE": ""}, clear=False):
        os.environ.pop("METRIC_SOURCE", None)
        config = PlannerConfig()
        assert config.metric_source == "sglang"


def test_metric_source_from_backend_vllm():
    with patch.dict(os.environ, {"BACKEND": "vllm-chat"}, clear=False):
        os.environ.pop("METRIC_SOURCE", None)
        config = PlannerConfig()
        assert config.metric_source == "vllm"


def test_metric_source_explicit_overrides_backend():
    with patch.dict(os.environ, {"BACKEND": "vllm", "METRIC_SOURCE": "sglang"}):
        config = PlannerConfig()
        assert config.metric_source == "sglang"


def test_backend_env_var_stored():
    with patch.dict(os.environ, {"BACKEND": "sglang-oai"}):
        config = PlannerConfig()
        assert config.backend == "sglang-oai"
