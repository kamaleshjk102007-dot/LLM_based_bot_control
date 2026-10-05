"""Map universal tasks to the verified Magician Lite command surface."""

from __future__ import annotations

from typing import Any

from app.adapters.dobot.config import DobotConfig, DobotPosition
from app.adapters.dobot.exceptions import DobotUnsupportedActionError
from app.commands.models import Action, Task

SUPPORTED_ACTIONS = frozenset({
    Action.GET_STATUS, Action.HOME, Action.MOVE,
    Action.ROTATE, Action.STOP, Action.GRIP, Action.RELEASE,
    Action.PICK, Action.PLACE,
})

# Hard limit for single-axis relative calibration moves
REAL_LLM_MAX_STEP_MM = 50.0
REAL_LLM_MAX_ROTATION_DEGREES = 30.0

_CARTESIAN_DIRECTIONS = {
    "x": ("x", 1.0),
    "+x": ("x", 1.0),
    "x+": ("x", 1.0),
    "-x": ("x", -1.0),
    "x-": ("x", -1.0),
    "y": ("y", 1.0),
    "+y": ("y", 1.0),
    "y+": ("y", 1.0),
    "-y": ("y", -1.0),
    "y-": ("y", -1.0),
    "z": ("z", 1.0),
    "+z": ("z", 1.0),
    "z+": ("z", 1.0),
    "-z": ("z", -1.0),
    "z-": ("z", -1.0),
    "up": ("z", 1.0),
    "upward": ("z", 1.0),
    "upwards": ("z", 1.0),
    "down": ("z", -1.0),
    "downward": ("z", -1.0),
    "downwards": ("z", -1.0),
}
_MM_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}
_CM_UNITS = {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}
_INCH_UNITS = {"in", "inch", "inches"}
_METER_UNITS = {"m", "meter", "meters", "metre", "metres"}
_ALL_DISTANCE_UNITS = _MM_UNITS | _CM_UNITS | _INCH_UNITS | _METER_UNITS

_UNIT_TO_MM = {
    **{u: 1.0 for u in _MM_UNITS},
    **{u: 10.0 for u in _CM_UNITS},
    **{u: 25.4 for u in _INCH_UNITS},
    **{u: 1000.0 for u in _METER_UNITS},
}


def _resolve_coordinates(
    action_name: str,
    ref: Any | None,
    parameters: dict[str, Any] | None,
    perception: Any | None,
) -> DobotPosition:
    """Resolve target 3D coordinates from parameters or perception."""
    # 1. Direct coordinates in task parameters
    if parameters:
        x = parameters["x"] if "x" in parameters else parameters.get("target_x")
        y = parameters["y"] if "y" in parameters else parameters.get("target_y")
        z = parameters["z"] if "z" in parameters else parameters.get("target_z")
        r = parameters.get("r", 0.0)
        if x is not None and y is not None and z is not None:
            return DobotPosition(float(x), float(y), float(z), float(r))

    # 2. Query perception with entity reference
    if perception is not None and ref is not None:
        block = perception.find(
            color=getattr(ref, "color", None),
            type=getattr(ref, "type", None),
            name=getattr(ref, "name", None),
            id=getattr(ref, "id", None),
        )
        if block is not None:
            return DobotPosition(block.x, block.y, block.z, block.r)

    raise DobotUnsupportedActionError(
        f"{action_name} is not supported without resolved 3D coordinates from perception or parameters."
    )


def _map_pick(task: Task, config: DobotConfig, perception: Any | None = None) -> dict[str, Any]:
    pos = _resolve_coordinates("PICK", task.object, task.parameters, perception)
    hover_z = pos.z + config.hover_height_mm
    hover_pos = DobotPosition(pos.x, pos.y, hover_z, pos.r)

    if config.safety_limits is not None:
        config.safety_limits.validate(pos, label="PICK position")
        config.safety_limits.validate(hover_pos, label="PICK hover position")

    return {
        "action": Action.PICK.value,
        "x": pos.x,
        "y": pos.y,
        "z": pos.z,
        "r": pos.r,
        "hover_z": hover_z,
        "object": task.object.model_dump() if task.object else None,
    }


def _map_place(task: Task, config: DobotConfig, perception: Any | None = None) -> dict[str, Any]:
    pos = _resolve_coordinates("PLACE", task.target, task.parameters, perception)
    hover_z = pos.z + config.hover_height_mm
    hover_pos = DobotPosition(pos.x, pos.y, hover_z, pos.r)

    if config.safety_limits is not None:
        config.safety_limits.validate(pos, label="PLACE position")
        config.safety_limits.validate(hover_pos, label="PLACE hover position")

    return {
        "action": Action.PLACE.value,
        "x": pos.x,
        "y": pos.y,
        "z": pos.z,
        "r": pos.r,
        "hover_z": hover_z,
        "target": task.target.model_dump() if task.target else None,
    }


def _map_relative_move(task: Task, config: DobotConfig) -> dict[str, Any]:
    if task.position is not None or task.target is not None:
        raise DobotUnsupportedActionError(
            "Real LLM MOVE accepts only an explicit single-axis Cartesian distance."
        )
    direction = (task.direction or "").strip().lower()
    if direction not in _CARTESIAN_DIRECTIONS:
        raise DobotUnsupportedActionError(
            "Real LLM MOVE supports only explicit X, Y, Z, upward, or downward movement."
        )
    if task.distance is None:
        raise DobotUnsupportedActionError(
            "Real LLM MOVE requires an explicit distance."
        )
    unit = (task.unit or "").strip().lower()
    if unit not in _ALL_DISTANCE_UNITS:
        raise DobotUnsupportedActionError(
            "Real LLM MOVE requires a distance unit "
            "(mm, cm, inches, or meters)."
        )
    # Convert to mm
    distance_mm = task.distance * _UNIT_TO_MM[unit]
    max_step = min(config.calibration_max_step_mm, REAL_LLM_MAX_STEP_MM)
    if distance_mm > max_step:
        raise DobotUnsupportedActionError(
            f"Real LLM MOVE is limited to {max_step:g} mm "
            f"({max_step / 10:g} cm) per command. "
            f"You requested {distance_mm:g} mm."
        )
    axis, sign = _CARTESIAN_DIRECTIONS[direction]
    return {
        "action": Action.MOVE.value,
        "type": "relative",
        "axis": axis,
        "delta_mm": sign * distance_mm,
    }


def _map_absolute_move(task: Task, config: DobotConfig) -> dict[str, Any]:
    params = task.parameters or {}
    x = float(params["x"])
    y = float(params["y"])
    z = float(params["z"])
    r = float(params.get("r", 0.0))
    target = DobotPosition(x, y, z, r)
    if config.safety_limits is not None:
        config.safety_limits.validate(target, label="MOVE target")
    return {
        "action": Action.MOVE.value,
        "type": "absolute",
        "x": target.x,
        "y": target.y,
        "z": target.z,
        "r": target.r,
    }


def _map_relative_rotation(task: Task, config: DobotConfig) -> dict[str, Any]:
    direction = (task.direction or "").strip().lower()
    directions = {
        "r": 1.0, "+r": 1.0, "r+": 1.0,
        "-r": -1.0, "r-": -1.0,
    }
    if direction not in directions:
        raise DobotUnsupportedActionError(
            "Real LLM ROTATE supports only an explicit R or -R direction."
        )
    if task.angle is None or (task.unit or "degrees").strip().lower() not in {
        "degree", "degrees", "deg",
    }:
        raise DobotUnsupportedActionError(
            "Real LLM R rotation requires an explicit angle in degrees."
        )
    max_step = min(
        config.rotation_max_step_degrees,
        REAL_LLM_MAX_ROTATION_DEGREES,
    )
    if task.angle > max_step:
        raise DobotUnsupportedActionError(
            f"Real LLM R rotation is limited to {max_step:g} degrees per command."
        )
    return {
        "action": Action.ROTATE.value,
        "axis": "r",
        "delta_degrees": directions[direction] * task.angle,
    }


def map_task(task: Task, config: DobotConfig, perception: Any | None = None) -> dict[str, Any]:
    if task.action not in SUPPORTED_ACTIONS:
        raise DobotUnsupportedActionError(
            f"{task.action.value} is not supported by the DOBOT adapter."
        )
    if task.action is Action.MOVE:
        if task.position == "configured_safe_test_position":
            target = config.require_safe_test_position()
            return {
                "action": Action.MOVE.value,
                "type": "absolute",
                "x": target.x,
                "y": target.y,
                "z": target.z,
                "r": target.r,
            }
        params = task.parameters or {}
        if "x" in params and "y" in params and "z" in params:
            return _map_absolute_move(task, config)
        if task.object is not None and perception is not None:
            pos = _resolve_coordinates("MOVE", task.object, task.parameters, perception)
            hover_z = pos.z + config.hover_height_mm
            return {
                "action": Action.MOVE.value,
                "type": "absolute",
                "x": pos.x,
                "y": pos.y,
                "z": hover_z,
                "r": pos.r,
            }
        return _map_relative_move(task, config)
    if task.action is Action.ROTATE:
        return _map_relative_rotation(task, config)
    if task.action is Action.PICK:
        return _map_pick(task, config, perception)
    if task.action is Action.PLACE:
        return _map_place(task, config, perception)

    operation: dict[str, Any] = {"action": task.action.value}
    if task.action is Action.GRIP:
        operation.update({"enable": True, "on": True})
    elif task.action is Action.RELEASE:
        operation.update({"enable": True, "on": False})
    elif task.action is Action.STOP:
        operation["stop_type"] = "software_queue_stop"
    return operation
