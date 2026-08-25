"""In-memory robot registry for Phase 2."""

from __future__ import annotations

from pydantic import ValidationError

from app.commands.models import Action
from app.robots.models import Robot, RobotStatus


class RobotRegistryError(ValueError):
    pass


class DuplicateRobotError(RobotRegistryError):
    pass


class RobotNotFoundError(RobotRegistryError):
    pass


class RobotRegistry:
    def __init__(self) -> None:
        self._robots: dict[str, Robot] = {}

    def register(self, robot: Robot | dict) -> Robot:
        try:
            payload = robot.model_dump() if isinstance(robot, Robot) else robot
            validated = Robot.model_validate(payload)
        except (ValidationError, TypeError, ValueError) as exc:
            raise RobotRegistryError(f"Invalid robot: {exc}") from exc
        if validated.robot_id in self._robots:
            raise DuplicateRobotError(
                f"Robot ID already registered: {validated.robot_id}"
            )
        self._robots[validated.robot_id] = validated
        return validated

    def unregister(self, robot_id: str) -> Robot:
        try:
            return self._robots.pop(robot_id)
        except KeyError as exc:
            raise RobotNotFoundError(f"Robot not found: {robot_id}") from exc

    def get(self, robot_id: str) -> Robot:
        try:
            return self._robots[robot_id]
        except KeyError as exc:
            raise RobotNotFoundError(f"Robot not found: {robot_id}") from exc

    def list_all(self) -> list[Robot]:
        return [self._robots[key] for key in sorted(self._robots)]

    def list_online(self) -> list[Robot]:
        return [
            robot for robot in self.list_all() if robot.status is RobotStatus.ONLINE
        ]

    def find_by_capability(self, action: Action | str) -> list[Robot]:
        try:
            capability = Action(action)
        except ValueError as exc:
            raise RobotRegistryError(f"Unsupported action: {action}") from exc
        return [
            robot
            for robot in self.list_all()
            if capability in robot.capabilities
        ]

    def update_status(self, robot_id: str, status: RobotStatus | str) -> Robot:
        robot = self.get(robot_id)
        try:
            normalized_status = RobotStatus(status)
        except ValueError as exc:
            raise RobotRegistryError(f"Invalid robot status: {status}") from exc
        updated = robot.model_copy(update={"status": normalized_status})
        self._robots[robot_id] = updated
        return updated
