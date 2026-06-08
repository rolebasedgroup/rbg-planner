# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/kubernetes_connector.py + kube.py for RoleBasedGroup.

"""RBG Kubernetes connector for applying scaling decisions.

Adapted from dynamo/planner/kubernetes_connector.py + kube.py.
Scales RBG roles via RBGSA (RoleBasedGroupScalingAdapter) scale subresource,
with fallback to direct RBG spec.roles[].replicas patching.
"""

import asyncio
import logging
from typing import Optional

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

from rbg_planner.planner_connector import PlannerConnector, TargetReplica
from rbg_planner.utils.exceptions import (
    DeploymentNotReadyError,
    EmptyTargetReplicasError,
    RBGNotFoundError,
    RoleNotFoundError,
)

logger = logging.getLogger(__name__)

# RBG CRD coordinates
RBG_GROUP = "workloads.x-k8s.io"
RBG_VERSION = "v1alpha2"
RBG_PLURAL = "rolebasedgroups"
RBGSA_PLURAL = "rolebasedgroupscalingadapters"


def _get_current_namespace() -> str:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "default"


class RBGConnector(PlannerConnector):
    """Connector that scales RBG roles via RBGSA or direct RBG patch."""

    def __init__(
        self,
        rbg_name: str,
        rbg_namespace: Optional[str] = None,
        prefill_role_name: str = "prefill",
        decode_role_name: str = "decode",
    ):
        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config()

        self.custom_api = client.CustomObjectsApi()
        self.rbg_name = rbg_name
        self.namespace = rbg_namespace or _get_current_namespace()
        self.prefill_role_name = prefill_role_name
        self.decode_role_name = decode_role_name

        self._cached_rbg: Optional[dict] = None

    def _get_rbg(self) -> dict:
        """Fetch the RoleBasedGroup object."""
        try:
            rbg = self.custom_api.get_namespaced_custom_object(
                group=RBG_GROUP,
                version=RBG_VERSION,
                namespace=self.namespace,
                plural=RBG_PLURAL,
                name=self.rbg_name,
            )
            self._cached_rbg = rbg
            return rbg
        except client.ApiException as e:
            if e.status == 404:
                raise RBGNotFoundError(self.rbg_name, self.namespace)
            raise

    def _get_role_from_rbg(self, rbg: dict, role_name: str) -> dict:
        """Find a role spec in the RBG."""
        roles = rbg.get("spec", {}).get("roles", [])
        for role in roles:
            if role.get("name") == role_name:
                return role
        raise RoleNotFoundError(role_name, self.rbg_name)

    def _get_role_status(self, rbg: dict, role_name: str) -> Optional[dict]:
        """Get role status from RBG status."""
        statuses = rbg.get("status", {}).get("roleStatuses", [])
        for status in statuses:
            if status.get("name") == role_name:
                return status
        return None

    def _try_scale_via_rbgsa(self, role_name: str, replicas: int) -> bool:
        """Try to scale via RBGSA scale subresource. Returns True if successful."""
        # RBGSA naming convention: <rbg-name>-<role-name>
        adapter_name = f"{self.rbg_name}-{role_name}"
        try:
            self.custom_api.patch_namespaced_custom_object_scale(
                group=RBG_GROUP,
                version=RBG_VERSION,
                namespace=self.namespace,
                plural=RBGSA_PLURAL,
                name=adapter_name,
                body={"spec": {"replicas": replicas}},
            )
            logger.info(f"Scaled RBGSA {adapter_name} to {replicas} replicas")
            return True
        except client.ApiException as e:
            if e.status == 404:
                logger.debug(f"RBGSA {adapter_name} not found, will fall back to RBG patch")
                return False
            raise

    def _patch_rbg_role_replicas(self, role_name: str, replicas: int):
        """Fallback: directly patch RBG spec.roles[].replicas."""
        rbg = self._get_rbg()
        roles = rbg.get("spec", {}).get("roles", [])

        role_index = None
        for i, role in enumerate(roles):
            if role.get("name") == role_name:
                role_index = i
                break

        if role_index is None:
            raise RoleNotFoundError(role_name, self.rbg_name)

        # Use JSON patch to update the specific role's replicas
        patch = [
            {
                "op": "replace",
                "path": f"/spec/roles/{role_index}/replicas",
                "value": replicas,
            }
        ]
        self.custom_api.patch_namespaced_custom_object(
            group=RBG_GROUP,
            version=RBG_VERSION,
            namespace=self.namespace,
            plural=RBG_PLURAL,
            name=self.rbg_name,
            body=patch,
            _content_type="application/json-patch+json",
        )
        logger.info(
            f"Patched RBG {self.rbg_name} role {role_name} to {replicas} replicas"
        )

    def _scale_role(self, role_name: str, replicas: int):
        """Scale a role: try RBGSA first, fall back to RBG patch."""
        if not self._try_scale_via_rbgsa(role_name, replicas):
            self._patch_rbg_role_replicas(role_name, replicas)

    async def set_replicas(self, target_replicas: list[TargetReplica], blocking: bool = True):
        if not target_replicas:
            raise EmptyTargetReplicasError()

        rbg = self._get_rbg()
        if not self.is_ready():
            logger.warning(
                f"RBG {self.rbg_name} is not ready, skipping this scaling"
            )
            return

        for target in target_replicas:
            # Check current replicas to avoid unnecessary patches
            role = self._get_role_from_rbg(rbg, target.role_name)
            current_replicas = role.get("replicas", 0)
            if current_replicas != target.desired_replicas:
                logger.info(
                    f"Scaling role {target.role_name}: {current_replicas} -> {target.desired_replicas}"
                )
                self._scale_role(target.role_name, target.desired_replicas)
            else:
                logger.info(
                    f"Role {target.role_name} already at {target.desired_replicas} replicas"
                )

        if blocking:
            await self.wait_for_ready()

    async def validate_deployment(self):
        rbg = self._get_rbg()
        # Verify prefill and decode roles exist
        self._get_role_from_rbg(rbg, self.prefill_role_name)
        self._get_role_from_rbg(rbg, self.decode_role_name)
        logger.info(f"Validated RBG {self.rbg_name}: found roles {self.prefill_role_name}, {self.decode_role_name}")

    async def wait_for_ready(self, max_attempts: int = 180, delay_seconds: int = 10):
        for attempt in range(max_attempts):
            await asyncio.sleep(delay_seconds)
            rbg = self._get_rbg()
            conditions = rbg.get("status", {}).get("conditions", [])
            ready = next((c for c in conditions if c.get("type") == "Ready"), None)
            if ready and ready.get("status") == "True":
                return
            logger.info(
                f"[{attempt + 1}/{max_attempts}] Waiting for RBG {self.rbg_name} to be ready"
            )
        raise DeploymentNotReadyError(self.rbg_name)

    def get_role_ready_replicas(self, role_name: str) -> int:
        rbg = self._get_rbg()
        status = self._get_role_status(rbg, role_name)
        if status is None:
            return 0
        return status.get("readyReplicas", 0)

    def is_ready(self) -> bool:
        rbg = self._cached_rbg or self._get_rbg()
        conditions = rbg.get("status", {}).get("conditions", [])
        ready = next((c for c in conditions if c.get("type") == "Ready"), None)
        return ready is not None and ready.get("status") == "True"
