"""Universal command-level contract implemented by every robot adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.commands.models import UniversalCommand
from app.robots.models import Robot


class RobotInterface(ABC):
    """Robot-agnostic boundary used by the gateway.

    Individual operations such as MOVE, ROTATE, HOME, STOP, GET_STATUS,
    GRIP, and RELEASE remain UniversalCommand tasks. Whether a robot exposes
    each operation is determined by the registry capability declaration.
    """

    simulated: bool = True

    def __init__(self, robot: Robot) -> None:
        self.robot = robot

    @abstractmethod
    def validate(self, command: UniversalCommand) -> bool:
        """Return whether this implementation can safely accept the command."""

    @abstractmethod
    def prepare(self, command: UniversalCommand) -> list[dict[str, Any]]:
        """Translate validated tasks without executing them."""

    @abstractmethod
    def execute(self, command: UniversalCommand) -> list[str]:
        """Execute or simulate the command, raising on failure."""

    @abstractmethod
    def get_status(self) -> str:
        """Return implementation status; it may include pose information."""
