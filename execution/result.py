from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.commands.models import Action


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    STOPPED = "STOPPED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


class ExecutionResult(BaseModel):
    """Standard result returned by the execution layer."""

    execution_id: str
    robot_id: str
    action: Action
    status: ExecutionStatus
    message: str = ""

    requested_axis: str | None = None
    requested_distance: float | None = None
    requested_unit: str | None = None

    actual_axis: str | None = None
    actual_distance: float | None = None
    actual_unit: str | None = None

    error: float | None = None
    tolerance: float | None = None
    verification: bool | None = None

    # Initial robot pose before execution
    initial_x: float | None = None
    initial_y: float | None = None
    initial_z: float | None = None
    initial_r: float | None = None

    # Final robot pose after execution
    final_x: float | None = None
    final_y: float | None = None
    final_z: float | None = None
    final_r: float | None = None

    failure_reason: str | None = None