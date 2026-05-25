"""Custom exceptions for RBG Planner."""


class PlannerError(Exception):
    """Base exception for planner errors."""
    pass


class RBGNotFoundError(PlannerError):
    """Raised when the target RoleBasedGroup is not found."""

    def __init__(self, rbg_name: str, namespace: str):
        super().__init__(
            f"RoleBasedGroup '{rbg_name}' not found in namespace '{namespace}'"
        )


class RoleNotFoundError(PlannerError):
    """Raised when a role is not found in the RBG."""

    def __init__(self, role_name: str, rbg_name: str):
        super().__init__(
            f"Role '{role_name}' not found in RoleBasedGroup '{rbg_name}'"
        )


class DeploymentNotReadyError(PlannerError):
    """Raised when the deployment is not ready."""

    def __init__(self, rbg_name: str):
        super().__init__(f"RoleBasedGroup '{rbg_name}' is not ready")


class EmptyTargetReplicasError(PlannerError):
    """Raised when target replicas list is empty."""

    def __init__(self):
        super().__init__("Target replicas list cannot be empty")


class ProfilingDataNotFoundError(PlannerError):
    """Raised when profiling data files are not found."""

    def __init__(self, path: str):
        super().__init__(
            f"Profiling data not found at '{path}'. "
            "Please generate profiling data using inference-ext-cli generate-profiling "
            "and mount it as a ConfigMap."
        )
