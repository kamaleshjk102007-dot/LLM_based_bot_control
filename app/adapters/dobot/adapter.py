"""Universal RobotAdapter implementation for the DOBOT Magician Lite."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.adapters.base import RobotAdapter, RobotAdapterError
from app.adapters.dobot.client import ConnectionState, DobotLinkClient
from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode
from app.adapters.dobot.exceptions import DobotError
from app.adapters.dobot.mapper import SUPPORTED_ACTIONS, map_task
from app.commands.models import Action, UniversalCommand
from app.robots.models import Robot

ConfirmationCallback = Callable[[str, str], bool]


class DobotMagicianLiteAdapter(RobotAdapter):
    simulated = False

    def __init__(
        self,
        robot: Robot,
        client: DobotLinkClient,
        config: DobotConfig,
        confirm: ConfirmationCallback,
    ) -> None:
        super().__init__(robot)
        if config.mode is not OperationMode.REAL:
            raise ValueError("The DOBOT adapter may only be created in real mode.")
        self.client = client
        self.config = config
        self.confirm = confirm

    def validate(self, command: UniversalCommand) -> bool:
        if self.client.state is not ConnectionState.READY:
            return False
        if any(task.action not in SUPPORTED_ACTIONS for task in command.tasks):
            return False
        try:
            # Map every task before executing any task: multi-step commands fail closed.
            [map_task(task, self.config) for task in command.tasks]
        except DobotError:
            return False
        return True

    def prepare(self, command: UniversalCommand) -> list[dict[str, Any]]:
        try:
            return [map_task(task, self.config) for task in command.tasks]
        except DobotError as exc:
            raise RobotAdapterError(str(exc)) from exc

    def _confirmed(self, operation: dict[str, Any]) -> None:
        action = operation["action"]
        if action in {Action.GET_STATUS.value, Action.STOP.value}:
            return
        detail = json.dumps(operation, sort_keys=True)
        if not self.confirm(action, detail):
            raise RobotAdapterError(f"Physical {action} was cancelled by the operator.")

    def execute(self, command: UniversalCommand) -> list[str]:
        operations = self.prepare(command)
        results: list[str] = []
        try:
            for operation in operations:
                self._confirmed(operation)
                action = Action(operation["action"])
                if action is Action.GET_STATUS:
                    result = self.client.get_status()
                elif action is Action.HOME:
                    result = self.client.home()
                elif action is Action.MOVE:
                    result = self.client.move(DobotPosition(**operation["position"]))
                elif action is Action.STOP:
                    result = self.client.stop()
                elif action is Action.GRIP:
                    result = self.client.set_gripper(True)
                elif action is Action.RELEASE:
                    result = self.client.set_gripper(False)
                else:  # protected by map_task, retained as a fail-closed guard
                    raise RobotAdapterError(f"Unsupported DOBOT action: {action.value}")
                results.append(f"[DOBOT REAL] {action.value}: {result!r}")
        except (DobotError, TimeoutError) as exc:
            raise RobotAdapterError(str(exc)) from exc
        except RobotAdapterError:
            raise
        except Exception as exc:
            raise RobotAdapterError(f"DOBOT execution failed: {exc}") from exc
        return results

    def get_status(self) -> str:
        return json.dumps(self.client.get_status(), default=str, sort_keys=True)
