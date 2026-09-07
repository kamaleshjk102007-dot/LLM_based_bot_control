"""Configuration-backed, hardware-independent robot registry."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.commands.models import Action
from app.robots.models import Robot, RobotStatus

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "robots.json"


class RobotRegistryError(ValueError):
    """Base error for controlled registry failures."""


class RobotConfigurationError(RobotRegistryError):
    """Raised when the robot configuration cannot be loaded or validated."""


class DuplicateRobotError(RobotRegistryError):
    """Raised when a robot ID is registered more than once."""


class RobotNotFoundError(RobotRegistryError):
    """Raised by strict operations when a robot ID is unknown."""


class RobotRegistry:
    """Store robot metadata; never communicate with or move hardware."""

    def __init__(self) -> None:
        self._robots: dict[str, Robot] = {}

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "RobotRegistry":
        """Load a registry from JSON, failing explicitly on bad configuration."""
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        try:
            raw = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RobotConfigurationError(
                f"Could not read robot configuration {config_path}: {exc}"
            ) from exc

        try:
            document: Any = json.loads(raw)
        except JSONDecodeError as exc:
            raise RobotConfigurationError(
                f"Invalid JSON in robot configuration {config_path}: "
                f"line {exc.lineno}, column {exc.colno}"
            ) from exc

        if not isinstance(document, dict) or not isinstance(
            document.get("robots"), list
        ):
            raise RobotConfigurationError(
                f"Robot configuration {config_path} must contain a 'robots' list."
            )

        registry = cls()
        for index, item in enumerate(document["robots"]):
            if not isinstance(item, dict):
                raise RobotConfigurationError(
                    f"Robot entry {index} in {config_path} must be an object."
                )
            try:
                registry.register(item)
            except RobotRegistryError as exc:
                raise RobotConfigurationError(
                    f"Invalid robot entry {index} in {config_path}: {exc}"
                ) from exc
        return registry

    def register(self, robot: Robot | dict[str, Any]) -> Robot:
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
        """Strict lookup retained for existing gateway error handling."""
        robot = self.get_robot(robot_id)
        if robot is None:
            raise RobotNotFoundError(f"Robot not found: {robot_id}")
        return robot

    def get_robot(self, robot_id: str) -> Robot | None:
        """Return robot metadata, or None for an unknown ID."""
        return self._robots.get(robot_id)

    def has_robot(self, robot_id: str) -> bool:
        return robot_id in self._robots

    def has_capability(self, robot_id: str, capability: Action | str) -> bool:
        robot = self.get_robot(robot_id)
        if robot is None:
            return False
        try:
            normalized = (
                capability
                if isinstance(capability, Action)
                else Action(str(capability).strip().upper())
            )
        except ValueError:
            return False
        return normalized in robot.capabilities

    def list_all(self) -> list[Robot]:
        return [self._robots[key] for key in sorted(self._robots)]

    def list_online(self) -> list[Robot]:
        return [
            robot for robot in self.list_all() if robot.status is RobotStatus.ONLINE
        ]

    def find_by_capability(self, action: Action | str) -> list[Robot]:
        try:
            capability = (
                action
                if isinstance(action, Action)
                else Action(str(action).strip().upper())
            )
        except ValueError as exc:
            raise RobotRegistryError(f"Unsupported action: {action}") from exc
        return [
            robot for robot in self.list_all() if capability in robot.capabilities
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
