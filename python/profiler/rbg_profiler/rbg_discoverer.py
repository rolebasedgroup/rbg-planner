"""Discover inference endpoints from a RoleBasedGroup Custom Resource.

Reads the RBG CR via the Kubernetes API and assembles per-role base-URLs
by determining the role's pattern type (Standalone, LeaderWorker, or
CustomComponents), reading the headless service name, and inferring the
container port from the pod template's container args.

DNS naming follows RBG conventions:
  Standalone:        {rbg}-{role}-0.{svc}.{ns}.svc.cluster.local
  LeaderWorker:      {rbg}-{role}-0-0.{svc}.{ns}.svc.cluster.local  (leader pod)
  CustomComponents:  {rbg}-{role}-0-{component}-0.{svc}.{ns}.svc.cluster.local
"""

import logging
import re
from dataclasses import dataclass

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

logger = logging.getLogger(__name__)

RBG_GROUP = "workloads.x-k8s.io"
RBG_VERSION = "v1alpha2"
RBG_PLURAL = "rolebasedgroups"

# Default ports per engine type. Must stay in sync with DEFAULT_PORTS
# in bench_serving.py. Matching uses startswith in dict order, so more
# specific prefixes must come first ("sglang-pd-native" before "sglang").
ENGINE_DEFAULT_PORTS = {
    # PD-native engines listen on the engine port (e.g. 8000/8001), not the
    # router port 30000; each role's --port argument takes precedence.
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
class RoleEndpoint:
    base_url: str
    num_gpus: int
    tokenizer: str


def _load_k8s_client():
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi()


def _find_role(rbg: dict, role_name: str) -> dict | None:
    roles = rbg.get("spec", {}).get("roles", [])
    for role in roles:
        if role.get("name") == role_name:
            return role
    return None


def _detect_pattern_type(role: dict) -> str:
    if "standalonePattern" in role:
        return "standalone"
    if "leaderWorkerPattern" in role:
        return "leaderWorker"
    if "customComponentsPattern" in role:
        return "customComponents"
    return "standalone"


def _get_template(role: dict, pattern_type: str) -> tuple[dict, str]:
    """Return (template, component_name). component_name is non-empty only
    for customComponents; empty string for standalone/leaderWorker."""
    if pattern_type == "standalone":
        return role.get("standalonePattern", {}).get("template", {}), ""
    if pattern_type == "leaderWorker":
        return role.get("leaderWorkerPattern", {}).get("leaderTemplate", {}), ""
    if pattern_type == "customComponents":
        components = role.get("customComponentsPattern", {}).get("components", [])
        for comp in components:
            if comp.get("name") == "engine" or comp.get("name") == "server":
                return comp.get("template", {}), comp.get("name", "")
        if components:
            comp_name = components[0].get("name", "")
            return components[0].get("template", {}), comp_name
    return {}, ""


def _get_service_name(role: dict, pattern_type: str, rbg_name: str, role_name: str) -> str:
    if pattern_type == "standalone":
        svc = role.get("standalonePattern", {}).get("serviceName", "")
    elif pattern_type == "leaderWorker":
        svc = role.get("leaderWorkerPattern", {}).get("serviceName", "")
    elif pattern_type == "customComponents":
        svc = role.get("customComponentsPattern", {}).get("serviceName", "")
    else:
        svc = ""

    if not svc:
        svc = f"s-{rbg_name}-{role_name}"
        logger.warning(
            "No serviceName found in RBG role %s, falling back to convention: %s",
            role_name, svc,
        )
    return svc


def _parse_port_from_args(args: list[str], engine: str) -> int:
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                pass
        if arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass

    for engine_prefix, port in ENGINE_DEFAULT_PORTS.items():
        if engine.startswith(engine_prefix):
            return port

    return 8000


def _parse_gpu_count(containers: list[dict]) -> int:
    for container in containers:
        requests = (
            container.get("resources", {})
            .get("requests", {})
        )
        for key, value in requests.items():
            if key.endswith("/gpu"):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass
    return 1


def _build_dns_name(
    rbg_name: str,
    role_name: str,
    pattern_type: str,
    service_name: str,
    namespace: str,
    component_name: str = "",
) -> str:
    instance = f"{rbg_name}-{role_name}-0"

    if pattern_type == "leaderWorker":
        pod_name = f"{instance}-0"
    elif pattern_type == "customComponents":
        comp = component_name or "engine"
        pod_name = f"{instance}-{comp}-0"
    else:
        pod_name = instance

    return f"{pod_name}.{service_name}.{namespace}.svc.cluster.local"


def discover_role_endpoint(
    rbg_name: str,
    namespace: str,
    role_name: str,
    engine: str,
    fallback_tokenizer: str = "",
) -> RoleEndpoint:
    api = _load_k8s_client()
    rbg = api.get_namespaced_custom_object(
        group=RBG_GROUP,
        version=RBG_VERSION,
        namespace=namespace,
        plural=RBG_PLURAL,
        name=rbg_name,
    )

    role = _find_role(rbg, role_name)
    if role is None:
        raise ValueError(f"Role '{role_name}' not found in RBG '{rbg_name}'")

    pattern_type = _detect_pattern_type(role)
    template, component_name = _get_template(role, pattern_type)
    containers = template.get("spec", {}).get("containers", [])

    service_name = _get_service_name(role, pattern_type, rbg_name, role_name)

    all_args = []
    for container in containers:
        all_args.extend(container.get("args", []))
        all_args.extend(container.get("command", []))

    port = _parse_port_from_args(all_args, engine)
    tokenizer = fallback_tokenizer
    num_gpus = _parse_gpu_count(containers)

    dns_name = _build_dns_name(rbg_name, role_name, pattern_type, service_name, namespace, component_name)
    base_url = f"http://{dns_name}:{port}"

    logger.info(
        "Discovered endpoint for role '%s': %s (pattern=%s, gpus=%d, tokenizer=%s)",
        role_name, base_url, pattern_type, num_gpus, tokenizer or "(none)",
    )

    return RoleEndpoint(base_url=base_url, num_gpus=num_gpus, tokenizer=tokenizer)
