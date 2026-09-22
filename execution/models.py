from __future__ import annotations

from pydantic import BaseModel


class Pose(BaseModel):
    """Robot Cartesian pose used for execution verification."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0


class RequestedMovement(BaseModel):
    """Movement requested by the UniversalCommand."""

    axis: str
    distance: float
    unit: str
    direction: int = 1


class ActualMovement(BaseModel):
    """Movement measured from initial and final robot poses."""

    axis: str
    distance: float
    unit: str
    direction: int = 1


class VerificationResult(BaseModel):
    """Result of comparing requested and actual movement."""

    passed: bool
    error: float | None = None
    tolerance: float | None = None
    message: str = ""


class ExecutionMeasurement(BaseModel):
    """Complete movement measurement used by the executor."""

    initial_pose: Pose
    final_pose: Pose
    requested: RequestedMovement | None = None
    actual: ActualMovement | None = None
    verification: VerificationResult | None = None