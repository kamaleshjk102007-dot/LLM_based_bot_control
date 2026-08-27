"""Map universal tasks to the small, verified Magician Lite command surface."""

from __future__ import annotations

from typing import Any

from app.adapters.dobot.config import DobotConfig
from app.adapters.dobot.exceptions import DobotUnsupportedActionError
from app.commands.models import Action, Task

SUPPORTED_ACTIONS = frozenset({
    Action.GET_STATUS, Action.HOME, Action.MOVE,
    Action.STOP, Action.GRIP, Action.RELEASE,
})

# This hard limit cannot be increased through an environment variable.
REAL_LLM_MAX_STEP_MM = 1.0
_VERTICAL_DIRECTIONS = {
    "up": 1.0,
    "upward": 1.0,
    "upwards": 1.0,
    "down": -1.0,
    "downward": -1.0,
    "downwards": -1.0,
}
_MM_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}


def _map_relative_move(task: Task, config: DobotConfig) -> dict[str, Any]:
    if task.position is not None or task.target is not None:
        raise DobotUnsupportedActionError(
            "Real LLM MOVE accepts only an explicit upward or downward distance."
        )
    direction = (task.direction or "").strip().lower()
    if direction not in _VERTICAL_DIRECTIONS:
        raise DobotUnsupportedActionError(
            "Real LLM MOVE currently supports only upward or downward Z movement."
        )
    if task.distance is None or (task.unit or "").strip().lower() not in _MM_UNITS:
        raise DobotUnsupportedActionError(
            "Real LLM MOVE requires an explicit distance in millimeters."
        )
    max_step = min(config.calibration_max_step_mm, REAL_LLM_MAX_STEP_MM)
    if task.distance > max_step:
        raise DobotUnsupportedActionError(
            f"Real LLM MOVE is limited to {max_step:g} mm per command."
        )
    return {
        "action": Action.MOVE.value,
        "axis": "z",
        "delta_mm": _VERTICAL_DIRECTIONS[direction] * task.distance,
    }


def map_task(task: Task, config: DobotConfig) -> dict[str, Any]:
    if task.action not in SUPPORTED_ACTIONS:
        raise DobotUnsupportedActionError(
            f"{task.action.value} is not supported by the Phase 3 DOBOT adapter."
        )
    if task.action is Action.MOVE:
        return _map_relative_move(task, config)

    operation: dict[str, Any] = {"action": task.action.value}
    if task.action is Action.GRIP:
        operation.update({"enable": True, "on": True})
    elif task.action is Action.RELEASE:
        operation.update({"enable": True, "on": False})
    elif task.action is Action.STOP:
        operation["stop_type"] = "software_queue_stop"
    return operation
