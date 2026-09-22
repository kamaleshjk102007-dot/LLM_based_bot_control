from __future__ import annotations

from execution.models import (
    ActualMovement,
    RequestedMovement,
    VerificationResult,
)


class MovementVerifier:
    def __init__(self, tolerance: float = 0.2) -> None:
        self.tolerance = tolerance

    def verify(
        self,
        requested: RequestedMovement,
        actual: ActualMovement,
    ) -> VerificationResult:

        requested_axis = requested.axis.upper()
        actual_axis = actual.axis.upper()

        # ========================================================
        # CHECK 1: AXIS
        # ========================================================

        if requested_axis != actual_axis:
            return VerificationResult(
                passed=False,
                error=None,
                tolerance=self.tolerance,
                message=(
                    f"Expected movement on {requested_axis}, "
                    f"but actual movement was on {actual_axis}"
                ),
            )

        # ========================================================
        # CHECK 2: NO MOVEMENT
        # ========================================================

        if actual.distance == 0:
            error = round(
                abs(requested.distance - actual.distance),
                10,
            )

            return VerificationResult(
                passed=False,
                error=error,
                tolerance=self.tolerance,
                message=(
                    f"No movement detected. "
                    f"Movement error {error:.3f} "
                    f"{requested.unit} exceeds tolerance "
                    f"{self.tolerance:.3f} {requested.unit}"
                ),
            )

        # ========================================================
        # CHECK 3: DIRECTION
        # ========================================================

        if requested.direction != actual.direction:
            expected_direction = (
                "+" if requested.direction > 0 else "-"
            )

            actual_direction = (
                "+" if actual.direction > 0 else "-"
            )

            return VerificationResult(
                passed=False,
                error=None,
                tolerance=self.tolerance,
                message=(
                    f"Expected direction "
                    f"{expected_direction}{requested_axis}, "
                    f"but actual direction was "
                    f"{actual_direction}{actual_axis}"
                ),
            )

        # ========================================================
        # CHECK 4: DISTANCE
        # ========================================================

        error = round(
            abs(requested.distance - actual.distance),
            10,
        )

        if error <= self.tolerance + 1e-9:
            return VerificationResult(
                passed=True,
                error=error,
                tolerance=self.tolerance,
                message=(
                    "Requested axis, direction, and movement "
                    "distance are within tolerance"
                ),
            )

        # ========================================================
        # DISTANCE VERIFICATION FAILED
        # ========================================================

        return VerificationResult(
            passed=False,
            error=error,
            tolerance=self.tolerance,
            message=(
                f"Movement error {error:.3f} "
                f"{requested.unit} exceeds tolerance "
                f"{self.tolerance:.3f} {requested.unit}"
            ),
        )