# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring/traffic_metrics.py for RoleBasedGroup.

"""Query-first Prometheus metrics adapter.

A single PrometheusAdapter executes one fully-resolved PromQL query per
logical slot (ttft, itl, requestsTotal, requestDuration, isl, osl). Query
resolution — engine presets, CRD overrides, placeholder substitution —
happens in the factory (create_metrics_adapter); the adapter only runs the
final query strings and averages the resulting series.
"""

import logging
import os
from typing import Optional

from prometheus_api_client import PrometheusConnect

from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.defaults import QUERY_SLOTS

logger = logging.getLogger(__name__)


class PrometheusAdapter(MetricsAdapter):
    """Prometheus adapter executing pre-resolved queries per slot."""

    def __init__(
        self,
        url: str,
        queries: dict[str, str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ):
        missing = [slot for slot in QUERY_SLOTS if slot not in queries]
        if missing:
            raise ValueError(f"Missing queries for slots: {missing}")
        # Credentials from args (tests) or env (operator-injected planner
        # env, e.g. bearer token for the ARMS Prometheus query API).
        # Bearer token wins over Basic auth.
        bearer_token = bearer_token if bearer_token is not None else os.environ.get("PROMETHEUS_TOKEN")
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else None
        username = username if username is not None else os.environ.get("PROMETHEUS_USERNAME")
        password = password if password is not None else os.environ.get("PROMETHEUS_PASSWORD")
        auth = (username, password) if username and password else None
        self.prom = PrometheusConnect(url=url, auth=auth, headers=headers, disable_ssl=True)
        self.queries = queries

    def _query_slot(self, slot: str) -> float:
        """Execute the slot's query and average all returned series."""
        query = self.queries[slot]
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                logger.debug(f"No data for {slot}")
                return 0.0
            values = [float(r["value"][1]) for r in result if r["value"][1] != "NaN"]
            if not values:
                return 0.0
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error querying {slot}: {e}")
            return 0.0

    # ── MetricsAdapter interface ─────────────────────────────────────

    def get_avg_ttft(self) -> float:
        return self._query_slot("ttft")

    def get_avg_itl(self) -> float:
        return self._query_slot("itl")

    def get_request_count(self) -> float:
        return self._query_slot("requestsTotal")

    def get_avg_request_duration(self) -> float:
        return self._query_slot("requestDuration")

    def get_avg_isl(self) -> float:
        return self._query_slot("isl")

    def get_avg_osl(self) -> float:
        return self._query_slot("osl")
