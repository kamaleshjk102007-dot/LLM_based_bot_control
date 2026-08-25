"""Strongly typed, robot-independent Universal Command models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(str, Enum):
    MOVE = "MOVE"
    ROTATE = "ROTATE"
    STOP = "STOP"
    HOME = "HOME"
    PICK = "PICK"
    PLACE = "PLACE"
    GRIP = "GRIP"
    RELEASE = "RELEASE"
    NAVIGATE = "NAVIGATE"
    GET_STATUS = "GET_STATUS"


class EntityReference(BaseModel):
    """Semantic reference only; never a physical pose or vendor identifier."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = Field(default=None, min_length=1)
    id: str | None = Field(default=None, min_length=1)
    color: str | None = Field(default=None, min_length=1)
    name: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_descriptor(self) -> "EntityReference":
        if not any((self.type, self.id, self.color, self.name)):
            raise ValueError("an entity reference requires at least one descriptor")
        return self


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    object: EntityReference | None = None
    target: EntityReference | None = None
    position: str | None = Field(default=None, min_length=1)
    direction: str | None = Field(default=None, min_length=1)
    distance: float | None = Field(default=None, gt=0)
    angle: float | None = Field(default=None, gt=0, le=360)
    unit: str | None = Field(default=None, min_length=1)
    parameters: dict[str, str | int | float | bool] | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Task":
        values = {
            "object": self.object,
            "target": self.target,
            "position": self.position,
            "direction": self.direction,
            "distance": self.distance,
            "angle": self.angle,
            "unit": self.unit,
            "parameters": self.parameters,
        }
        present = {name for name, value in values.items() if value is not None}

        required_any = {
            Action.PICK: {"object"},
            Action.PLACE: {"target"},
            Action.MOVE: {"direction", "target", "position"},
            Action.ROTATE: {"direction", "angle"},
            Action.NAVIGATE: {"target", "position", "direction"},
        }
        if self.action in required_any and not (present & required_any[self.action]):
            choices = ", ".join(sorted(required_any[self.action]))
            raise ValueError(f"{self.action.value} requires one of: {choices}")

        allowed = {
            Action.PICK: {"object", "parameters"},
            Action.PLACE: {"target", "parameters"},
            Action.MOVE: {"target", "position", "direction", "distance", "unit", "parameters"},
            Action.ROTATE: {"direction", "angle", "unit", "parameters"},
            Action.STOP: set(),
            Action.HOME: set(),
            Action.GRIP: {"object", "parameters"},
            Action.RELEASE: {"object", "target", "parameters"},
            Action.NAVIGATE: {"target", "position", "direction", "distance", "unit", "parameters"},
            Action.GET_STATUS: set(),
        }
        incompatible = present - allowed[self.action]
        if incompatible:
            names = ", ".join(sorted(incompatible))
            raise ValueError(f"{self.action.value} does not allow fields: {names}")

        if self.distance is not None and self.unit is None:
            raise ValueError("distance requires a unit")
        if self.angle is not None and self.unit is not None:
            normalized = self.unit.lower()
            if normalized not in {"degree", "degrees", "deg", "radian", "radians", "rad"}:
                raise ValueError("angle unit must be degrees or radians")
        if self.unit is not None and self.distance is None and self.angle is None:
            raise ValueError("unit requires distance or angle")
        return self


class UniversalCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1.0", pattern=r"^1\.0$")
    robot_id: str | None = None
    tasks: list[Task] = Field(min_length=1)


def normalized_command(command: UniversalCommand) -> dict[str, Any]:
    """Return JSON-ready data while omitting unspecified task fields."""

    data = command.model_dump(mode="json", exclude_none=True)
    data["robot_id"] = command.robot_id
    return data
