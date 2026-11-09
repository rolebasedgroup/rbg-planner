"""Tests for rbg_discoverer — verifies tokenizer is not inferred from RBG container args."""

from unittest.mock import patch, MagicMock

from rbg_profiler.rbg_discoverer import (
    _build_dns_name,
    _detect_pattern_type,
    _get_template,
    _parse_port_from_args,
    _parse_gpu_count,
    discover_role_endpoint,
)


def _mock_rbg_with_tokenizer_args():
    """Return a mock RBG whose prefill container has --tokenizer in args."""
    return {
        "spec": {
            "roles": [
                {
                    "name": "prefill",
                    "standalonePattern": {
                        "serviceName": "test-svc",
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "args": [
                                            "--tokenizer", "/inferred/path",
                                            "--port", "30000",
                                        ],
                                        "resources": {
                                            "requests": {"nvidia.com/gpu": "2"}
                                        },
                                    }
                                ]
                            }
                        },
                    },
                }
            ]
        }
    }


@patch("rbg_profiler.rbg_discoverer._load_k8s_client")
def test_tokenizer_not_inferred_from_args(mock_load_client):
    """Tokenizer must come from fallback_tokenizer, not from container args.

    The container args contain '--tokenizer /inferred/path', but the endpoint
    must use the explicit fallback_tokenizer value instead.
    """
    mock_api = MagicMock()
    mock_api.get_namespaced_custom_object.return_value = _mock_rbg_with_tokenizer_args()
    mock_load_client.return_value = mock_api

    endpoint = discover_role_endpoint(
        rbg_name="test-rbg",
        namespace="default",
        role_name="prefill",
        engine="sglang",
        fallback_tokenizer="/explicit/tokenizer",
    )

    assert endpoint.tokenizer == "/explicit/tokenizer", (
        f"expected '/explicit/tokenizer', got '{endpoint.tokenizer}'"
    )


@patch("rbg_profiler.rbg_discoverer._load_k8s_client")
def test_tokenizer_empty_when_no_fallback(mock_load_client):
    """When fallback_tokenizer is empty and args contain --tokenizer,
    the endpoint tokenizer must still be empty (no inference)."""
    mock_api = MagicMock()
    mock_api.get_namespaced_custom_object.return_value = _mock_rbg_with_tokenizer_args()
    mock_load_client.return_value = mock_api

    endpoint = discover_role_endpoint(
        rbg_name="test-rbg",
        namespace="default",
        role_name="prefill",
        engine="sglang",
        fallback_tokenizer="",
    )

    assert endpoint.tokenizer == "", (
        f"expected empty tokenizer, got '{endpoint.tokenizer}'"
    )


def test_build_dns_name_custom_components_with_engine():
    dns = _build_dns_name(
        "my-rbg", "prefill", "customComponents", "my-svc", "default",
        component_name="engine",
    )
    assert dns == "my-rbg-prefill-0-engine-0.my-svc.default.svc.cluster.local"


def test_build_dns_name_custom_components_with_server():
    dns = _build_dns_name(
        "my-rbg", "prefill", "customComponents", "my-svc", "default",
        component_name="server",
    )
    assert dns == "my-rbg-prefill-0-server-0.my-svc.default.svc.cluster.local"


def test_build_dns_name_standalone():
    dns = _build_dns_name(
        "my-rbg", "prefill", "standalone", "my-svc", "default",
        component_name="",
    )
    assert dns == "my-rbg-prefill-0.my-svc.default.svc.cluster.local"


def test_build_dns_name_leader_worker():
    dns = _build_dns_name(
        "my-rbg", "prefill", "leaderWorker", "my-svc", "default",
        component_name="",
    )
    assert dns == "my-rbg-prefill-0-0.my-svc.default.svc.cluster.local"


def test_parse_port_from_args_explicit_flag():
    assert _parse_port_from_args(["--port", "8080", "--model", "x"], "sglang") == 8080


def test_parse_port_from_args_equals_form():
    assert _parse_port_from_args(["--port=9000"], "sglang") == 9000


def test_parse_port_from_args_default_by_engine():
    assert _parse_port_from_args([], "sglang") == 30000
    assert _parse_port_from_args([], "vllm") == 8000
    assert _parse_port_from_args([], "lmdeploy") == 23333


def test_parse_port_from_args_pd_native_default_is_engine_port():
    """sglang-pd-native defaults to the engine port, not the router port."""
    assert _parse_port_from_args([], "sglang-pd-native") == 8000


def test_parse_port_from_args_pd_native_prefers_explicit_port():
    assert (
        _parse_port_from_args(["--port", "8001"], "sglang-pd-native") == 8001
    )
    assert _parse_port_from_args(["--port=8000"], "sglang-pd-native") == 8000


def test_parse_port_from_args_fallback():
    assert _parse_port_from_args([], "unknown-engine") == 8000


def test_parse_gpu_count_nvidia():
    containers = [{"resources": {"requests": {"nvidia.com/gpu": "2"}}}]
    assert _parse_gpu_count(containers) == 2


def test_parse_gpu_count_amd():
    containers = [{"resources": {"requests": {"amd.com/gpu": "4"}}}]
    assert _parse_gpu_count(containers) == 4


def test_parse_gpu_count_no_gpu():
    containers = [{"resources": {"requests": {"cpu": "1"}}}]
    assert _parse_gpu_count(containers) == 1


def test_detect_pattern_type_standalone():
    role = {"standalonePattern": {}}
    assert _detect_pattern_type(role) == "standalone"


def test_detect_pattern_type_leader_worker():
    role = {"leaderWorkerPattern": {}}
    assert _detect_pattern_type(role) == "leaderWorker"


def test_detect_pattern_type_custom_components():
    role = {"customComponentsPattern": {}}
    assert _detect_pattern_type(role) == "customComponents"


def test_detect_pattern_type_default():
    role = {}
    assert _detect_pattern_type(role) == "standalone"


def test_get_template_custom_components_engine():
    role = {
        "customComponentsPattern": {
            "components": [
                {"name": "init", "template": {}},
                {"name": "engine", "template": {"spec": {"containers": [{"args": ["--port", "30000"]}]}}, "name": "engine"},
            ]
        }
    }
    template, comp_name = _get_template(role, "customComponents")
    assert comp_name == "engine"
    assert "containers" in template.get("spec", {})


def test_get_template_custom_components_server():
    role = {
        "customComponentsPattern": {
            "components": [
                {"name": "server", "template": {"spec": {"containers": []}}},
            ]
        }
    }
    template, comp_name = _get_template(role, "customComponents")
    assert comp_name == "server"


def test_get_template_custom_components_fallback_first():
    role = {
        "customComponentsPattern": {
            "components": [
                {"name": "worker", "template": {"spec": {"containers": []}}},
            ]
        }
    }
    template, comp_name = _get_template(role, "customComponents")
    assert comp_name == "worker"


def test_get_template_standalone():
    role = {"standalonePattern": {"template": {"spec": {"containers": []}}}}
    template, comp_name = _get_template(role, "standalone")
    assert comp_name == ""
    assert "containers" in template.get("spec", {})
