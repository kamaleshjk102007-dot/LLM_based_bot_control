"""Adapter type registry. Only mock is installed in Phase 2."""

from __future__ import annotations

from collections.abc import Callable

from app.adapters.base import RobotAdapter
from app.adapters.mock import MockRobotAdapter
from app.robots.models import Robot

AdapterFactory = Callable[[Robot], RobotAdapter]


class AdapterManagerError(ValueError):
    pass


class AdapterNotFoundError(AdapterManagerError):
    pass


class AdapterManager:
    def __init__(self, register_mock: bool = True) -> None:
        self._factories: dict[str, AdapterFactory] = {}
        if register_mock:
            self.register("mock", MockRobotAdapter)

    def register(self, adapter_type: str, factory: AdapterFactory) -> None:
        key = adapter_type.strip().lower()
        if not key:
            raise AdapterManagerError("Adapter type cannot be blank.")
        if key in self._factories:
            raise AdapterManagerError(f"Adapter already registered: {key}")
        if not callable(factory):
            raise AdapterManagerError("Adapter factory must be callable.")
        self._factories[key] = factory

    def get(self, robot: Robot) -> RobotAdapter:
        try:
            factory = self._factories[robot.adapter_type]
        except KeyError as exc:
            raise AdapterNotFoundError(
                f"Unknown adapter type: {robot.adapter_type}"
            ) from exc
        adapter = factory(robot)
        if not isinstance(adapter, RobotAdapter):
            raise AdapterManagerError(
                f"Factory for {robot.adapter_type} did not return RobotAdapter"
            )
        return adapter

    def supported_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
