"""Pure guarded wrist-rotation helpers for Webots and unit tests."""

import math

_DEGREE_UNITS = {"degree", "degrees", "deg"}
_RADIAN_UNITS = {"radian", "radians", "rad"}
_R_DIRECTIONS = {
    "r": 1.0,
    "+r": 1.0,
    "r+": 1.0,
    "-r": -1.0,
    "r-": -1.0,
    "counterclockwise": 1.0,
    "counter-clockwise": 1.0,
    "anticlockwise": 1.0,
    "left": 1.0,
    "clockwise": -1.0,
    "right": -1.0,
}


def requested_r_radians(task):
    """Return a signed wrist rotation, or None for a non-R rotation."""
    direction = str(task.get("direction", "")).strip().lower()
    sign = _R_DIRECTIONS.get(direction)
    if sign is None:
        return None
    angle = float(task.get("angle", 0.0))
    if not math.isfinite(angle) or angle <= 0:
        raise ValueError("Wrist rotation requires a positive finite angle.")
    unit = str(task.get("unit", "degrees")).strip().lower()
    if unit in _DEGREE_UNITS:
        radians = math.radians(angle)
    elif unit in _RADIAN_UNITS:
        radians = angle
    else:
        raise ValueError("Wrist rotation unit must be degrees or radians.")
    return sign * radians


def angular_report(before_rad, after_rad, requested_rad, tolerance_deg=0.25):
    """Measure wrist rotation and compare it with the signed request."""
    actual_rad = float(after_rad) - float(before_rad)
    requested_deg = math.degrees(requested_rad)
    actual_deg = math.degrees(actual_rad)
    error_deg = actual_deg - requested_deg
    return {
        "axis": "r",
        "before_r_deg": math.degrees(float(before_rad)),
        "after_r_deg": math.degrees(float(after_rad)),
        "requested_r_deg": requested_deg,
        "actual_r_deg": actual_deg,
        "error_deg": error_deg,
        "tolerance_deg": tolerance_deg,
        "verified": abs(error_deg) <= tolerance_deg,
    }
