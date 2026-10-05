"""Universal RobotAdapter implementation for the DOBOT Magician Lite."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.adapters.base import RobotAdapter, RobotAdapterError
from app.adapters.dobot.client import ConnectionState, DobotLinkClient
from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode
from app.adapters.dobot.exceptions import DobotError
from app.adapters.dobot.mapper import (
    REAL_LLM_MAX_ROTATION_DEGREES,
    REAL_LLM_MAX_STEP_MM,
    SUPPORTED_ACTIONS,
    map_task,
)
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
        perception: Any | None = None,
    ) -> None:
        super().__init__(robot)
        if config.mode is not OperationMode.REAL:
            raise ValueError("The DOBOT adapter may only be created in real mode.")
        self.client = client
        self.config = config
        self.confirm = confirm
        self.perception = perception

    def validate(self, command: UniversalCommand) -> bool:
        if self.client.state is not ConnectionState.READY:
            return False
        if any(task.action not in SUPPORTED_ACTIONS for task in command.tasks):
            return False
        # Relative single-axis calibration moves must be isolated
        is_relative_single_axis = lambda t: (
            t.action in {Action.MOVE, Action.ROTATE}
            and not (t.parameters and "x" in t.parameters and "y" in t.parameters)
            and t.position != "configured_safe_test_position"
        )
        if any(is_relative_single_axis(task) for task in command.tasks) and len(command.tasks) != 1:
            return False
        try:
            # Map every task before executing any task: multi-step commands fail closed.
            [map_task(task, self.config, self.perception) for task in command.tasks]
        except DobotError:
            return False
        return True

    def prepare(self, command: UniversalCommand) -> list[dict[str, Any]]:
        try:
            return [map_task(task, self.config, self.perception) for task in command.tasks]
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
                action = Action(operation["action"])
                if action is Action.MOVE and operation.get("type") == "absolute":
                    target = DobotPosition(
                        operation["x"], operation["y"], operation["z"], operation["r"]
                    )
                    self._confirmed(operation)
                    result = self.client.move(target)
                elif action is Action.MOVE:
                    before, target = self.client.calibration_preview(
                        operation["axis"], operation["delta_mm"]
                    )
                    confirmation = {
                        **operation,
                        "before": before.as_dict(),
                        "target": target.as_dict(),
                        "hard_max_step_mm": min(
                            self.config.calibration_max_step_mm,
                            REAL_LLM_MAX_STEP_MM,
                        ),
                    }
                    self._confirmed(confirmation)
                    result = self.client.calibrate(
                        operation["axis"], operation["delta_mm"], before
                    )
                elif action is Action.ROTATE:
                    before, target = self.client.rotation_preview(
                        operation["delta_degrees"]
                    )
                    confirmation = {
                        **operation,
                        "before": before.as_dict(),
                        "target": target.as_dict(),
                        "hard_max_step_degrees": min(
                            self.config.rotation_max_step_degrees,
                            REAL_LLM_MAX_ROTATION_DEGREES,
                        ),
                    }
                    self._confirmed(confirmation)
                    result = self.client.calibrate(
                        "r", operation["delta_degrees"], before
                    )
                elif action is Action.PICK:
                    self._confirmed(operation)
                    hover_pos = DobotPosition(
                        operation["x"], operation["y"], operation["hover_z"], operation["r"]
                    )
                    pick_pos = DobotPosition(
                        operation["x"], operation["y"], operation["z"], operation["r"]
                    )
                    # 1. Approach safe hover above block
                    self.client.move(hover_pos)
                    # 2. Descend to block
                    self.client.move(pick_pos)
                    # 3. Grip block
                    self.client.set_gripper(True)
                    # 4. Retract back to hover
                    self.client.move(hover_pos)
                    result = {
                        "picked": True,
                        "position": pick_pos.as_dict(),
                        "object": operation.get("object"),
                    }
                elif action is Action.PLACE:
                    self._confirmed(operation)
                    hover_pos = DobotPosition(
                        operation["x"], operation["y"], operation["hover_z"], operation["r"]
                    )
                    place_pos = DobotPosition(
                        operation["x"], operation["y"], operation["z"], operation["r"]
                    )
                    # 1. Transit to hover above target place position
                    self.client.move(hover_pos)
                    # 2. Descend to release height
                    self.client.move(place_pos)
                    # 3. Release gripper
                    self.client.set_gripper(False)
                    # 4. Retract back to hover
                    self.client.move(hover_pos)
                    result = {
                        "placed": True,
                        "position": place_pos.as_dict(),
                        "target": operation.get("target"),
                    }
                else:
                    self._confirmed(operation)
                    if action is Action.GET_STATUS:
                        result = self.client.get_status()
                    elif action is Action.HOME:
                        result = self.client.home()
                    elif action is Action.STOP:
                        result = self.client.stop()
                    elif action is Action.GRIP:
                        result = self.client.set_gripper(True)
                    elif action is Action.RELEASE:
                        result = self.client.set_gripper(False)
                    else:
                        raise RobotAdapterError(
                            f"Unsupported DOBOT action: {action.value}"
                        )
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
