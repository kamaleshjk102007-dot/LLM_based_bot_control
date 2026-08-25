"""Pure in-memory simulated adapter for Phase 2."""

from __future__ import annotations

from typing import Any

from app.adapters.base import RobotAdapter
from app.commands.models import UniversalCommand
from app.commands.validator import CommandValidationError, validate_command


class MockRobotAdapter(RobotAdapter):
    """Records simulated work and never opens a physical connection."""

    def __init__(self, robot) -> None:
        super().__init__(robot)
        self.prepared: list[dict[str, Any]] = []
        self.executed: list[str] = []

    def validate(self, command: UniversalCommand) -> bool:
        try:
            validate_command(command)
        except CommandValidationError:
            return False
        return True

    def prepare(self, command: UniversalCommand) -> list[dict[str, Any]]:
        self.prepared = [
            {"sequence": index, "action": task.action.value}
            for index, task in enumerate(command.tasks, start=1)
        ]
        return list(self.prepared)

    def execute(self, command: UniversalCommand) -> list[str]:
        self.executed = [
            f"[MOCK] {task.action.value} accepted" for task in command.tasks
        ]
        return list(self.executed)

    def get_status(self) -> str:
        return "SIMULATED"
