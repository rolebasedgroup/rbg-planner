# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract metrics adapter interface.

The planner depends only on this interface. The Prometheus adapter executes
one pre-resolved query per logical metric; interval and model filtering are
already baked into the query strings.
"""

from abc import ABC, abstractmethod


class MetricsAdapter(ABC):
    """Unified metrics query interface for the planner."""

    @abstractmethod
    def get_avg_ttft(self) -> float:
        """Get average Time to First Token in seconds."""

    @abstractmethod
    def get_avg_itl(self) -> float:
        """Get average Inter-Token Latency in seconds."""

    @abstractmethod
    def get_request_count(self) -> float:
        """Get total request count in the adjustment interval."""

    @abstractmethod
    def get_avg_request_duration(self) -> float:
        """Get average request duration in seconds."""

    @abstractmethod
    def get_avg_isl(self) -> float:
        """Get average input sequence length (tokens)."""

    @abstractmethod
    def get_avg_osl(self) -> float:
        """Get average output sequence length (tokens)."""
