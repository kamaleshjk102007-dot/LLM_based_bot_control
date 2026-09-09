"""Robot-independent planning data without execution side effects."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.commands.models import Action

NonEmptyText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]


class RobotPosition(BaseModel):
    """Coordinates whose units and frame are agreed by the integration."""

    model_config = ConfigDict(extra="forbid")
    x: FiniteNumber
    y: FiniteNumber
    z: FiniteNumber
    r: FiniteNumber | None = None


class RobotTarget(BaseModel):
    """Perception target with declared validity, not a workspace approval."""

    model_config = ConfigDict(extra="forbid")
    target_id: NonEmptyText
    class_name: NonEmptyText
    position: RobotPosition
    confidence: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=1)]
    timestamp: datetime = Field(strict=True)
    coordinate_frame: NonEmptyText
    valid: bool = Field(strict=True)


class MotionStep(BaseModel):
    """One relative movement or parameter-free action using the existing enum."""

    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(strict=True, ge=0)
    action: Action
    direction: NonEmptyText | None = None
    distance: PositiveNumber | None = None
    angle: Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0, le=360)] | None = None
    unit: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "MotionStep":
        if self.action is Action.MOVE:
            if self.direction is None or self.distance is None or self.unit is None:
                raise ValueError("MOVE requires direction, positive distance, and unit")
            if self.angle is not None:
                raise ValueError("MOVE does not allow angle")
        elif self.action is Action.ROTATE:
            if self.direction is None or self.angle is None or self.unit is None:
                raise ValueError("ROTATE requires direction, positive angle, and unit")
            if self.distance is not None:
                raise ValueError("ROTATE does not allow distance")
            if self.unit.lower() not in {"degree", "degrees", "deg", "radian", "radians", "rad"}:
                raise ValueError("angle unit must be degrees or radians")
        elif self.action in {Action.GRIP, Action.RELEASE, Action.HOME, Action.STOP, Action.GET_STATUS}:
            if any(v is not None for v in (self.direction, self.distance, self.angle, self.unit)):
                raise ValueError("Non-movement steps do not accept movement fields")
        else:
            raise ValueError("This planning representation does not yet support this action")
        return self


class MotionPlan(BaseModel):
    """Ordered planning steps, not the gateway execution result."""

    model_config = ConfigDict(extra="forbid")
    plan_id: NonEmptyText
    robot_id: NonEmptyText
    # Non-target plans, such as HOME, do not require a perception target.
    target_id: NonEmptyText | None = None
    # An already-reached target is represented by an empty plan, never a fake move.
    steps: list[MotionStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_order(self) -> "MotionPlan":
        sequences = [step.sequence for step in self.steps]
        if any(a >= b for a, b in zip(sequences, sequences[1:])):
            raise ValueError("Step sequences must be unique and strictly increasing")
        return self
