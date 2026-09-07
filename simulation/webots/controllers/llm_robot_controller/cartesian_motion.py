"""Pure Cartesian helpers shared by the Webots controller and unit tests."""

import math

MM_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}
X_DIRECTIONS = {
    "x": 1.0,
    "+x": 1.0,
    "x+": 1.0,
    "-x": -1.0,
    "x-": -1.0,
}


def requested_x_metres(task):
    """Return a signed X displacement in metres, or None for legacy directions."""
    direction = str(task.get("direction", "")).strip().lower()
    if direction not in X_DIRECTIONS:
        return None
    unit = str(task.get("unit", "")).strip().lower()
    if unit not in MM_UNITS:
        raise ValueError("Cartesian X movement currently requires millimeters.")
    distance = float(task.get("distance", 0.0))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("Cartesian X movement requires a positive finite distance.")
    return X_DIRECTIONS[direction] * distance / 1000.0


def damped_xz_step(columns, error_xz, damping=1e-5, max_step=0.04):
    """Solve a bounded 2-joint damped least-squares X/Z correction."""
    (j11, j21), (j12, j22) = columns
    ex, ez = error_xz
    a = j11 * j11 + j12 * j12 + damping
    b = j11 * j21 + j12 * j22
    d = j21 * j21 + j22 * j22 + damping
    determinant = a * d - b * b
    if abs(determinant) < 1e-12:
        raise ValueError("Cartesian Jacobian is singular at the current pose.")
    ux = (d * ex - b * ez) / determinant
    uz = (-b * ex + a * ez) / determinant
    dq1 = j11 * ux + j21 * uz
    dq2 = j12 * ux + j22 * uz
    largest = max(abs(dq1), abs(dq2), 1e-12)
    if largest > max_step:
        scale = max_step / largest
        dq1 *= scale
        dq2 *= scale
    return dq1, dq2


def displacement_report(before_m, after_m, requested_x_m, tolerance_m=0.00075):
    """Measure X displacement and fail if it does not match the request."""
    actual_x_m = float(after_m[0]) - float(before_m[0])
    error_m = actual_x_m - requested_x_m
    return {
        "requested_x_mm": requested_x_m * 1000.0,
        "actual_x_mm": actual_x_m * 1000.0,
        "error_mm": error_m * 1000.0,
        "tolerance_mm": tolerance_m * 1000.0,
        "verified": abs(error_m) <= tolerance_m,
        "before_mm": [float(value) * 1000.0 for value in before_m],
        "after_mm": [float(value) * 1000.0 for value in after_m],
    }
