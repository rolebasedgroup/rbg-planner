"""Abstract connector interface for the planner."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TargetReplica:
    role_name: str
    desired_replicas: int


class PlannerConnector(ABC):
    """Abstract interface for applying scaling decisions."""

    @abstractmethod
    async def set_replicas(self, target_replicas: list[TargetReplica], blocking: bool = True):
        """Set the replicas for multiple roles at once."""
        pass

    @abstractmethod
    async def validate_deployment(self):
        """Verify the deployment exists and has the expected roles."""
        pass

    @abstractmethod
    async def wait_for_ready(self):
        """Wait for the deployment to be ready."""
        pass

    @abstractmethod
    def get_role_ready_replicas(self, role_name: str) -> int:
        """Get the number of ready replicas for a role."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the deployment is ready."""
        pass
