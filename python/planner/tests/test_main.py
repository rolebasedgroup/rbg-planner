# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI argument parsing and config building."""

from unittest.mock import patch

from rbg_planner.main import build_config, parse_args


class TestParseArgs:
    def test_metric_engine_any_string(self):
        with patch("sys.argv", ["planner", "--metric-engine", "acme-llm"]):
            args = parse_args()
        assert args.metric_engine == "acme-llm"

    def test_metric_queries_json_arg(self):
        json_str = '{"ttft":"custom"}'
        with patch("sys.argv", ["planner", "--metric-queries-json", json_str]):
            args = parse_args()
        assert args.metric_queries_json == json_str


class TestBuildConfig:
    def test_both_overrides_applied(self):
        json_str = '{"ttft":"custom"}'
        with patch(
            "sys.argv",
            [
                "planner",
                "--metric-engine",
                "acme-llm",
                "--metric-queries-json",
                json_str,
            ],
        ):
            args = parse_args()
        config = build_config(args)
        assert config.metric_engine == "acme-llm"
        assert config.metric_queries_json == json_str
