"""Logical safety checks only; no physical-world safety implementation."""

from __future__ import annotations

from app.commands.models import Action, UniversalCommand
from app.gateway.capability_manager import CapabilityError, CapabilityManager
from app.robots.models import Robot, RobotStatus


class SafetyError(ValueError):
    pass


class SafetyValidator:
    SUPPORTED_VERSIONS = frozenset({"1.0"})

    def __init__(self, capabilities: CapabilityManager) -> None:
        self.capabilities = capabilities

    def validate(self, command: UniversalCommand, robot: Robot) -> None:
        if command is None:
            raise SafetyError("Command is required.")
        if command.version not in self.SUPPORTED_VERSIONS:
            raise SafetyError(f"Unsupported command version: {command.version}")
        if not command.tasks:
            raise SafetyError("Command must contain at least one task.")
        if robot.status is not RobotStatus.ONLINE:
            raise SafetyError(f"Robot {robot.robot_id} is not online.")
        try:
            self.capabilities.validate_command_against_capabilities(robot, command)
        except CapabilityError as exc:
            raise SafetyError(str(exc)) from exc

        actions = [task.action for task in command.tasks]
        if Action.STOP in actions and len(actions) > 1:
            raise SafetyError("STOP cannot be combined with other tasks.")
