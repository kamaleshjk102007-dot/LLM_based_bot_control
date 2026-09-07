"""Pure Cartesian helpers shared by the Webots controller and unit tests."""

import math

MM_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}
AXIS_DIRECTIONS = {
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
}
X_DIRECTIONS = {
    direction: sign
    for direction, (axis, sign) in AXIS_DIRECTIONS.items()
    if axis == "x"
}


def requested_axis_metres(task):
    """Return (axis, signed displacement) for supported Cartesian millimeter moves."""
    direction = str(task.get("direction", "")).strip().lower()
    axis_and_sign = AXIS_DIRECTIONS.get(direction)
    if axis_and_sign is None:
        return None
    unit = str(task.get("unit", "")).strip().lower()
    if unit not in MM_UNITS:
        raise ValueError("Cartesian movement currently requires millimeters.")
    distance = float(task.get("distance", 0.0))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("Cartesian movement requires a positive finite distance.")
    axis, sign = axis_and_sign
    return axis, sign * distance / 1000.0


def requested_x_metres(task):
    """Return a signed X displacement in metres, or None for legacy directions."""
    requested = requested_axis_metres(task)
    if requested is None or requested[0] != "x":
        return None
    return requested[1]


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


def damped_scalar_step(jacobian, error, damping=1e-5, max_step=0.04):
    """Return a bounded damped least-squares step for one Cartesian component."""
    denominator = jacobian * jacobian + damping
    if denominator <= damping:
        raise ValueError("Cartesian Jacobian is singular at the current pose.")
    step = jacobian * error / denominator
    return max(-max_step, min(max_step, step))


def axis_displacement_report(
    before_m, after_m, axis, requested_m, tolerance_m=0.00075
):
    """Measure displacement on one Cartesian axis."""
    indices = {"x": 0, "y": 1, "z": 2}
    normalized_axis = str(axis).lower()
    if normalized_axis not in indices:
        raise ValueError("Unsupported Cartesian reporting axis.")
    index = indices[normalized_axis]
    actual_m = float(after_m[index]) - float(before_m[index])
    error_m = actual_m - requested_m
    return {
        "axis": normalized_axis,
        "requested_mm": requested_m * 1000.0,
        "actual_mm": actual_m * 1000.0,
        f"requested_{normalized_axis}_mm": requested_m * 1000.0,
        f"actual_{normalized_axis}_mm": actual_m * 1000.0,
        "error_mm": error_m * 1000.0,
        "tolerance_mm": tolerance_m * 1000.0,
        "verified": abs(error_m) <= tolerance_m,
        "before_mm": [float(value) * 1000.0 for value in before_m],
        "after_mm": [float(value) * 1000.0 for value in after_m],
    }


def displacement_report(before_m, after_m, requested_x_m, tolerance_m=0.00075):
    """Backward-compatible X displacement report."""
    return axis_displacement_report(
        before_m, after_m, "x", requested_x_m, tolerance_m
    )
