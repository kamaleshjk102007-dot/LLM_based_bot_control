"""Deterministic Cartesian planning in fixed X, Y, Z, R axis order."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.commands.models import Action
from motion.models import MotionPlan, MotionStep, NonEmptyText, RobotPosition, RobotTarget


class MotionPlanningError(ValueError):
    """Raised when valid planning inputs cannot produce a plan."""


class MotionPlanningPolicy(BaseModel):
    """Robot-agnostic step sizes and coordinate-frame agreement."""

    model_config = ConfigDict(extra="forbid")

    max_translation_step_mm: Annotated[
        float, Field(strict=True, allow_inf_nan=False, gt=0)
    ]
    max_rotation_step_degrees: Annotated[
        float, Field(strict=True, allow_inf_nan=False, gt=0, le=360)
    ]
    coordinate_frame: NonEmptyText
    translation_zero_tolerance_mm: Annotated[
        float, Field(strict=True, allow_inf_nan=False, ge=0)
    ] = 1e-9
    rotation_zero_tolerance_degrees: Annotated[
        float, Field(strict=True, allow_inf_nan=False, ge=0)
    ] = 1e-9

    @model_validator(mode="after")
    def validate_tolerances(self) -> "MotionPlanningPolicy":
        if self.translation_zero_tolerance_mm >= self.max_translation_step_mm:
            raise ValueError(
                "Translation zero tolerance must be smaller than the maximum step"
            )
        if self.rotation_zero_tolerance_degrees >= self.max_rotation_step_degrees:
            raise ValueError(
                "Rotation zero tolerance must be smaller than the maximum step"
            )
        return self


class MotionPlanner:
    """Create planning data only; this class has no execution dependency."""

    AXIS_ORDER = ("X", "Y", "Z", "R")

    def __init__(self, policy: MotionPlanningPolicy) -> None:
        self.policy = policy

    def plan(
        self,
        current: RobotPosition,
        target: RobotTarget,
        robot_id: str,
    ) -> MotionPlan:
        """Plan relative steps without transforming frames or executing commands."""

        if not isinstance(current, RobotPosition):
            raise MotionPlanningError("current must be a valid RobotPosition")
        if not isinstance(target, RobotTarget):
            raise MotionPlanningError("target must be a valid RobotTarget")
        if not isinstance(robot_id, str) or not robot_id.strip():
            raise MotionPlanningError("robot_id must be a non-empty string")
        try:
            current = RobotPosition.model_validate(current.model_dump())
            target = RobotTarget.model_validate(target.model_dump())
            policy = MotionPlanningPolicy.model_validate(self.policy.model_dump())
        except ValidationError as exc:
            raise MotionPlanningError(f"Invalid planning input: {exc}") from exc
        if not target.valid:
            raise MotionPlanningError("RobotTarget is marked invalid")
        if target.coordinate_frame != policy.coordinate_frame:
            raise MotionPlanningError(
                "Target coordinate frame "
                f"{target.coordinate_frame!r} does not match planning frame "
                f"{policy.coordinate_frame!r}"
            )
        if target.position.r is not None and current.r is None:
            raise MotionPlanningError(
                "Current position requires R when the target includes R"
            )

        deltas = (
            ("X", target.position.x - current.x),
            ("Y", target.position.y - current.y),
            ("Z", target.position.z - current.z),
        )
        steps: list[MotionStep] = []
        for axis, delta in deltas:
            self._append_steps(
                steps,
                axis=axis,
                delta=delta,
                maximum=policy.max_translation_step_mm,
                tolerance=policy.translation_zero_tolerance_mm,
                action=Action.MOVE,
                unit="mm",
            )

        if target.position.r is not None:
            self._append_steps(
                steps,
                axis="R",
                delta=target.position.r - current.r,
                maximum=policy.max_rotation_step_degrees,
                tolerance=policy.rotation_zero_tolerance_degrees,
                action=Action.ROTATE,
                unit="degrees",
            )

        return MotionPlan(
            plan_id=self._plan_id(current, target, robot_id),
            robot_id=robot_id.strip(),
            target_id=target.target_id,
            steps=steps,
        )

    @staticmethod
    def _append_steps(
        steps: list[MotionStep],
        *,
        axis: str,
        delta: float,
        maximum: float,
        tolerance: float,
        action: Action,
        unit: str,
    ) -> None:
        magnitude = abs(delta)
        if magnitude <= tolerance:
            return

        full_steps = int(magnitude // maximum)
        remainder = magnitude - full_steps * maximum
        parts = [maximum] * full_steps
        if remainder > tolerance:
            parts.append(remainder)
        if not parts:
            parts.append(magnitude)

        direction = f"{'+' if delta > 0 else '-'}{axis}"
        for amount in parts:
            values = {
                "sequence": len(steps),
                "action": action,
                "direction": direction,
                "unit": unit,
            }
            values["distance" if action is Action.MOVE else "angle"] = amount
            steps.append(MotionStep.model_validate(values))

    def _plan_id(
        self,
        current: RobotPosition,
        target: RobotTarget,
        robot_id: str,
    ) -> str:
        payload = {
            "current": current.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "robot_id": robot_id.strip(),
            "policy": self.policy.model_dump(mode="json"),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return f"motion-{hashlib.sha256(encoded).hexdigest()[:16]}"
