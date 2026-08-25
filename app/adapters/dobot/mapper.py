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


def map_task(task: Task, config: DobotConfig) -> dict[str, Any]:
    if task.action not in SUPPORTED_ACTIONS:
        raise DobotUnsupportedActionError(
            f"{task.action.value} is not supported by the Phase 3 DOBOT adapter."
        )
    operation: dict[str, Any] = {"action": task.action.value}
    if task.action is Action.MOVE:
        operation["position"] = config.require_safe_test_position().as_dict()
        operation["ptp_mode"] = config.ptp_mode
    elif task.action is Action.GRIP:
        operation.update({"enable": True, "on": True})
    elif task.action is Action.RELEASE:
        operation.update({"enable": True, "on": False})
    elif task.action is Action.STOP:
        operation["stop_type"] = "software_queue_stop"
    return operation
