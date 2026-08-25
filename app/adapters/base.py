"""Generic adapter contract. No transport or vendor concepts belong here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.commands.models import UniversalCommand
from app.robots.models import Robot


class RobotAdapter(ABC):
    def __init__(self, robot: Robot) -> None:
        self.robot = robot

    @abstractmethod
    def validate(self, command: UniversalCommand) -> bool:
        """Return whether this adapter can accept the validated command."""

    @abstractmethod
    def prepare(self, command: UniversalCommand) -> list[dict[str, Any]]:
        """Translate the command into generic, non-hardware preparation records."""

    @abstractmethod
    def execute(self, command: UniversalCommand) -> list[str]:
        """Execute through this adapter (simulation only in Phase 2)."""

    @abstractmethod
    def get_status(self) -> str:
        """Return adapter-level status."""
