# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring for RoleBasedGroup.

"""Abstract metrics adapter interface.

The planner depends only on this interface. Concrete adapters
(SGLang, vLLM, Dynamo, ...) implement source-specific query logic.
"""

from abc import ABC, abstractmethod
from typing import Optional


class MetricsAdapter(ABC):
    """Unified metrics query interface for the planner."""

    @abstractmethod
    def get_avg_ttft(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average Time to First Token in seconds."""

    @abstractmethod
    def get_avg_itl(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average Inter-Token Latency in seconds."""

    @abstractmethod
    def get_request_count(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get total request count in the interval."""

    @abstractmethod
    def get_avg_request_duration(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average request duration in seconds."""

    @abstractmethod
    def get_avg_isl(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average input sequence length (tokens)."""

    @abstractmethod
    def get_avg_osl(self, interval: str, model_name: Optional[str] = None) -> float:
        """Get average output sequence length (tokens)."""
