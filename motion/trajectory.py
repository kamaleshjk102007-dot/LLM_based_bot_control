"""Deterministic, execution-free trajectory representation."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from motion.models import MotionPlan, MotionStep, NonEmptyText


class TrajectoryError(ValueError):
    """Raised when a motion plan cannot form a structurally valid trajectory."""


class Trajectory(BaseModel):
    """An ordered, execution-free snapshot of a validated motion plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_id: NonEmptyText
    plan_id: NonEmptyText
    robot_id: NonEmptyText
    target_id: NonEmptyText | None = None
    steps: tuple[MotionStep, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_step_order(self) -> "Trajectory":
        sequences = [step.sequence for step in self.steps]
        pairs = zip(sequences, sequences[1:])
        if any(current >= following for current, following in pairs):
            raise ValueError(
                "Trajectory step sequences must be unique and strictly increasing"
            )
        pairs = zip(sequences, sequences[1:])
        if any(following != current + 1 for current, following in pairs):
            raise ValueError("Trajectory step sequences must be contiguous")
        return self

    @classmethod
    def from_plan(cls, plan: MotionPlan) -> "Trajectory":
        """Validate and copy a plan without mutating or executing it."""

        if not isinstance(plan, MotionPlan):
            raise TrajectoryError("plan must be a valid MotionPlan")
        try:
            validated = MotionPlan.model_validate(plan.model_dump())
            trajectory = cls(
                trajectory_id=cls._trajectory_id(validated),
                plan_id=validated.plan_id,
                robot_id=validated.robot_id,
                target_id=validated.target_id,
                steps=tuple(step.model_copy(deep=True) for step in validated.steps),
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise TrajectoryError(f"Invalid MotionPlan for trajectory: {exc}") from exc
        return trajectory

    def is_empty(self) -> bool:
        return not self.steps

    def total_steps(self) -> int:
        return len(self.steps)

    def get_steps(self) -> tuple[MotionStep, ...]:
        """Return copies so callers cannot mutate this trajectory through the helper."""

        return tuple(step.model_copy(deep=True) for step in self.steps)

    @staticmethod
    def _trajectory_id(plan: MotionPlan) -> str:
        encoded = json.dumps(
            plan.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"trajectory-{hashlib.sha256(encoded).hexdigest()[:16]}"
