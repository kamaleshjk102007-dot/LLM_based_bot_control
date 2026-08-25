"""Deterministic robot selection. Gemini is never involved."""

from __future__ import annotations

from app.commands.models import UniversalCommand
from app.gateway.capability_manager import CapabilityError, CapabilityManager
from app.gateway.robot_registry import RobotNotFoundError, RobotRegistry
from app.robots.models import Robot, RobotStatus


class RobotSelectionError(ValueError):
    pass


class NoRobotError(RobotSelectionError):
    pass


class RobotUnavailableError(RobotSelectionError):
    pass


class UnsupportedRobotError(RobotSelectionError):
    pass


class RobotSelector:
    def __init__(
        self, registry: RobotRegistry, capabilities: CapabilityManager
    ) -> None:
        self.registry = registry
        self.capabilities = capabilities

    def select(self, command: UniversalCommand) -> Robot:
        if command.robot_id is not None:
            try:
                robot = self.registry.get(command.robot_id)
            except RobotNotFoundError as exc:
                raise RobotUnavailableError(str(exc)) from exc
            if robot.status is not RobotStatus.ONLINE:
                raise RobotUnavailableError(
                    f"Robot {robot.robot_id} is not online: {robot.status.value}"
                )
            try:
                self.capabilities.validate_command_against_capabilities(
                    robot, command
                )
            except CapabilityError as exc:
                raise UnsupportedRobotError(str(exc)) from exc
            return robot

        compatible: list[Robot] = []
        for robot in self.registry.list_online():
            try:
                self.capabilities.validate_command_against_capabilities(
                    robot, command
                )
            except CapabilityError:
                continue
            compatible.append(robot)

        if not compatible:
            actions = ", ".join(
                sorted({task.action.value for task in command.tasks})
            )
            raise NoRobotError(
                f"No online robot supports all required actions: {actions}"
            )
        return min(compatible, key=lambda robot: (robot.priority, robot.robot_id))
