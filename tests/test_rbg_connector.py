"""Unit tests for the RBG connector."""

from unittest.mock import MagicMock, patch

import pytest
from kubernetes.config.config_exception import ConfigException

from rbg_planner.rbg_connector import RBGConnector
from rbg_planner.utils.exceptions import RBGNotFoundError, RoleNotFoundError


@pytest.fixture
def mock_k8s():
    """Patch kubernetes config loading and CustomObjectsApi."""
    from kubernetes import client as real_client

    with patch("rbg_planner.rbg_connector.config") as mock_config, \
         patch("rbg_planner.rbg_connector.client") as mock_client:
        mock_config.load_incluster_config.side_effect = ConfigException("not in cluster")
        mock_api = MagicMock()
        mock_client.CustomObjectsApi.return_value = mock_api
        # Use real ApiException so isinstance checks work
        mock_client.ApiException = real_client.ApiException
        yield mock_api, mock_client


@pytest.fixture
def sample_rbg():
    return {
        "metadata": {"name": "test-rbg", "namespace": "default"},
        "spec": {
            "roles": [
                {"name": "prefill", "replicas": 2},
                {"name": "decode", "replicas": 3},
            ]
        },
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True"}
            ],
            "roleStatuses": [
                {"name": "prefill", "readyReplicas": 2, "replicas": 2},
                {"name": "decode", "readyReplicas": 3, "replicas": 3},
            ],
        },
    }


class TestRBGConnectorInit:
    def test_init_with_defaults(self, mock_k8s):
        connector = RBGConnector(rbg_name="my-rbg", rbg_namespace="test-ns")
        assert connector.rbg_name == "my-rbg"
        assert connector.namespace == "test-ns"
        assert connector.prefill_role_name == "prefill"
        assert connector.decode_role_name == "decode"


class TestGetRBG:
    def test_get_rbg_success(self, mock_k8s, sample_rbg):
        mock_api, _ = mock_k8s
        mock_api.get_namespaced_custom_object.return_value = sample_rbg

        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        rbg = connector._get_rbg()

        assert rbg["metadata"]["name"] == "test-rbg"
        mock_api.get_namespaced_custom_object.assert_called_once_with(
            group="workloads.x-k8s.io",
            version="v1alpha2",
            namespace="default",
            plural="rolebasedgroups",
            name="test-rbg",
        )

    def test_get_rbg_not_found(self, mock_k8s):
        mock_api, mock_client = mock_k8s
        exc = mock_client.ApiException(status=404)
        mock_api.get_namespaced_custom_object.side_effect = exc

        connector = RBGConnector(rbg_name="missing-rbg", rbg_namespace="default")
        with pytest.raises(RBGNotFoundError):
            connector._get_rbg()


class TestGetRoleFromRBG:
    def test_find_existing_role(self, mock_k8s, sample_rbg):
        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        role = connector._get_role_from_rbg(sample_rbg, "prefill")
        assert role["name"] == "prefill"
        assert role["replicas"] == 2

    def test_role_not_found(self, mock_k8s, sample_rbg):
        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        with pytest.raises(RoleNotFoundError):
            connector._get_role_from_rbg(sample_rbg, "nonexistent")


class TestScaleViaRBGSA:
    def test_scale_success(self, mock_k8s):
        mock_api, _ = mock_k8s
        mock_api.patch_namespaced_custom_object_scale.return_value = {}

        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        result = connector._try_scale_via_rbgsa("prefill", 4)

        assert result is True
        mock_api.patch_namespaced_custom_object_scale.assert_called_once_with(
            group="workloads.x-k8s.io",
            version="v1alpha2",
            namespace="default",
            plural="rolebasedgroupscalingadapters",
            name="test-rbg-prefill",
            body={"spec": {"replicas": 4}},
        )

    def test_scale_rbgsa_not_found_falls_back(self, mock_k8s):
        mock_api, mock_client = mock_k8s
        exc = mock_client.ApiException(status=404)
        mock_api.patch_namespaced_custom_object_scale.side_effect = exc

        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        result = connector._try_scale_via_rbgsa("prefill", 4)

        assert result is False


class TestPatchRBGRoleReplicas:
    def test_patch_success(self, mock_k8s, sample_rbg):
        mock_api, _ = mock_k8s
        mock_api.get_namespaced_custom_object.return_value = sample_rbg
        mock_api.patch_namespaced_custom_object.return_value = {}

        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        connector._patch_rbg_role_replicas("decode", 5)

        mock_api.patch_namespaced_custom_object.assert_called_once()
        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["body"] == [
            {"op": "replace", "path": "/spec/roles/1/replicas", "value": 5}
        ]
        assert call_kwargs["_content_type"] == "application/json-patch+json"


class TestIsReady:
    def test_ready_true(self, mock_k8s, sample_rbg):
        mock_api, _ = mock_k8s
        mock_api.get_namespaced_custom_object.return_value = sample_rbg

        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        connector._cached_rbg = sample_rbg
        assert connector.is_ready() is True

    def test_ready_false(self, mock_k8s):
        mock_api, _ = mock_k8s
        rbg_not_ready = {
            "status": {
                "conditions": [
                    {"type": "Ready", "status": "False"}
                ]
            }
        }
        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        connector._cached_rbg = rbg_not_ready
        assert connector.is_ready() is False


class TestGetRoleReadyReplicas:
    def test_get_ready_replicas(self, mock_k8s, sample_rbg):
        mock_api, _ = mock_k8s
        mock_api.get_namespaced_custom_object.return_value = sample_rbg

        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        assert connector.get_role_ready_replicas("prefill") == 2
        assert connector.get_role_ready_replicas("decode") == 3

    def test_missing_role_returns_zero(self, mock_k8s, sample_rbg):
        mock_api, _ = mock_k8s
        mock_api.get_namespaced_custom_object.return_value = sample_rbg

        connector = RBGConnector(rbg_name="test-rbg", rbg_namespace="default")
        assert connector.get_role_ready_replicas("nonexistent") == 0
