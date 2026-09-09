"""Pure structural and semantic validation for motion planning artifacts."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.commands.models import Action
from motion.models import MotionPlan, MotionStep
from motion.trajectory import Trajectory


class MotionValidationResult(BaseModel):
    """Deterministic validation outcome for a future motion controller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> "MotionValidationResult":
        if self.valid == bool(self.errors):
            raise ValueError("valid must be true exactly when errors is empty")
        return self


class MotionValidator:
    """Validate planning semantics without gateway, adapter, or hardware access."""

    MOVE_DIRECTIONS = frozenset({"+X", "-X", "+Y", "-Y", "+Z", "-Z"})
    ROTATE_DIRECTIONS = frozenset({"+R", "-R"})
    NON_MOVEMENT_ACTIONS = frozenset({
        Action.GRIP,
        Action.RELEASE,
        Action.HOME,
        Action.STOP,
        Action.GET_STATUS,
    })
    STEP_FIELDS = ("sequence", "action", "direction", "distance", "angle", "unit")

    @classmethod
    def validate_plan(cls, plan: MotionPlan) -> MotionValidationResult:
        errors = cls._validate_artifact(plan, MotionPlan, "MotionPlan")
        return cls._result(errors)

    @classmethod
    def validate_trajectory(
        cls, trajectory: Trajectory
    ) -> MotionValidationResult:
        errors = cls._validate_artifact(trajectory, Trajectory, "Trajectory")
        return cls._result(errors)

    @classmethod
    def validate(
        cls,
        plan: MotionPlan,
        trajectory: Trajectory,
    ) -> MotionValidationResult:
        errors = list(cls.validate_plan(plan).errors)
        errors.extend(cls.validate_trajectory(trajectory).errors)
        if isinstance(plan, MotionPlan) and isinstance(trajectory, Trajectory):
            cls._validate_consistency(plan, trajectory, errors)
        return cls._result(errors)

    @classmethod
    def _validate_artifact(
        cls,
        artifact: Any,
        expected_type: type[MotionPlan] | type[Trajectory],
        label: str,
    ) -> list[str]:
        if artifact is None:
            return [f"{label} is required."]
        if not isinstance(artifact, expected_type):
            return [f"{label} must be a {label} instance."]

        errors: list[str] = []
        cls._validate_identifier(artifact, "plan_id", label, errors)
        cls._validate_identifier(artifact, "robot_id", label, errors)
        cls._validate_identifier(artifact, "target_id", label, errors, optional=True)
        if label == "Trajectory":
            cls._validate_identifier(
                artifact, "trajectory_id", label, errors
            )

        steps = getattr(artifact, "steps", None)
        if not isinstance(steps, (list, tuple)):
            errors.append(f"{label} steps must be an ordered collection.")
            return errors

        cls._validate_sequences(steps, label, errors)
        for index, step in enumerate(steps):
            cls._validate_step(step, index, errors)

        try:
            expected_type.model_validate(artifact.model_dump(warnings=False))
        except (ValidationError, TypeError, ValueError):
            if not errors:
                errors.append(f"{label} is structurally invalid.")
        return errors

    @staticmethod
    def _validate_identifier(
        artifact: Any,
        field: str,
        label: str,
        errors: list[str],
        *,
        optional: bool = False,
    ) -> None:
        value = getattr(artifact, field, None)
        if optional and value is None:
            return
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label} {field} must be a non-empty string.")

    @classmethod
    def _validate_sequences(
        cls,
        steps: list[Any] | tuple[Any, ...],
        label: str,
        errors: list[str],
    ) -> None:
        sequences: list[int] = []
        for index, step in enumerate(steps):
            sequence = getattr(step, "sequence", None)
            if (
                not isinstance(step, MotionStep)
                or not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 0
            ):
                errors.append(f"MotionStep at index {index} has invalid sequence.")
                continue
            sequences.append(sequence)

        if len(sequences) != len(set(sequences)):
            errors.append(f"{label} contains duplicate sequence numbers.")
        if any(a > b for a, b in zip(sequences, sequences[1:])):
            errors.append(f"{label} sequence numbers must be strictly increasing.")
        if any(b != a + 1 for a, b in zip(sequences, sequences[1:])):
            errors.append(f"{label} sequence numbers must be contiguous.")

    @classmethod
    def _validate_step(
        cls,
        step: Any,
        index: int,
        errors: list[str],
    ) -> None:
        if not isinstance(step, MotionStep):
            errors.append(f"Step at index {index} must be a MotionStep.")
            return

        action = step.action
        if action is Action.MOVE:
            if step.direction not in cls.MOVE_DIRECTIONS:
                errors.append(f"MOVE step {step.sequence} has invalid direction.")
            if not cls._positive_finite(step.distance):
                errors.append(f"MOVE step {step.sequence} has non-positive distance.")
            if step.unit != "mm":
                errors.append(f"MOVE step {step.sequence} must use unit 'mm'.")
        elif action is Action.ROTATE:
            if step.direction not in cls.ROTATE_DIRECTIONS:
                errors.append(f"ROTATE step {step.sequence} has invalid direction.")
            if not cls._positive_finite(step.angle):
                errors.append(f"ROTATE step {step.sequence} has non-positive angle.")
            if step.unit != "degrees":
                errors.append(
                    f"ROTATE step {step.sequence} must use unit 'degrees'."
                )
        elif action in cls.NON_MOVEMENT_ACTIONS:
            if any(
                value is not None
                for value in (
                    step.direction,
                    step.distance,
                    step.angle,
                    step.unit,
                )
            ):
                errors.append(
                    f"Non-movement step {step.sequence} contains movement fields."
                )
        else:
            errors.append(f"MotionStep {step.sequence} has unsupported action.")

    @staticmethod
    def _positive_finite(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )

    @classmethod
    def _validate_consistency(
        cls,
        plan: MotionPlan,
        trajectory: Trajectory,
        errors: list[str],
    ) -> None:
        for field in ("plan_id", "robot_id", "target_id"):
            if getattr(trajectory, field, None) != getattr(plan, field, None):
                errors.append(
                    f"Trajectory {field} does not match MotionPlan {field}."
                )

        plan_steps = getattr(plan, "steps", ())
        trajectory_steps = getattr(trajectory, "steps", ())
        if not isinstance(plan_steps, (list, tuple)) or not isinstance(
            trajectory_steps, (list, tuple)
        ):
            return
        if len(trajectory_steps) != len(plan_steps):
            errors.append(
                "Trajectory step count does not match MotionPlan step count."
            )

        for index, (plan_step, trajectory_step) in enumerate(
            zip(plan_steps, trajectory_steps)
        ):
            for field in cls.STEP_FIELDS:
                if getattr(trajectory_step, field, None) != getattr(
                    plan_step, field, None
                ):
                    errors.append(
                        f"Trajectory step {index} {field} does not match "
                        f"MotionPlan step {index} {field}."
                    )

    @staticmethod
    def _result(errors: list[str]) -> MotionValidationResult:
        return MotionValidationResult(valid=not errors, errors=tuple(errors))
