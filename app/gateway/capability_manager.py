"""All-or-nothing robot capability validation."""

from __future__ import annotations

from app.commands.models import Action, UniversalCommand
from app.robots.capabilities import missing_actions, required_actions
from app.robots.models import Robot


class CapabilityError(ValueError):
    def __init__(self, missing: frozenset[Action]) -> None:
        self.missing = missing
        actions = ", ".join(sorted(action.value for action in missing))
        super().__init__(f"Robot does not support: {actions}")


class CapabilityManager:
    def has_capability(self, robot: Robot, action: Action | str) -> bool:
        try:
            normalized = Action(action)
        except ValueError:
            return False
        return normalized in robot.capabilities

    def get_supported_actions(self, robot: Robot) -> frozenset[Action]:
        return robot.capabilities

    def validate_command_against_capabilities(
        self, robot: Robot, command: UniversalCommand
    ) -> None:
        missing = missing_actions(
            robot.capabilities, required_actions(command.tasks)
        )
        if missing:
            raise CapabilityError(missing)
