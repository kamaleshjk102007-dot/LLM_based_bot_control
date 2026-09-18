"""Member 3's explicit boundary test against unmodified Member 4 planning."""

from datetime import datetime, timedelta, timezone

import pytest

from motion.models import RobotPosition
from motion.planner import MotionPlanner, MotionPlanningPolicy
from vision.models import Point3D, RobotTarget, TargetStatus
from vision.target import MotionTargetConversionError, to_motion_target


def vision_target(**overrides):
    values = {
        "target_id": "vision-target-1",
        "class_name": "red_block",
        "position": Point3D(x=210.0, y=60.0, z=35.0),
        "confidence": 0.94,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "dobot_base",
        "valid": True,
        "status": TargetStatus.VALID,
    }
    values.update(overrides)
    return RobotTarget(**values)


def planner():
    return MotionPlanner(MotionPlanningPolicy(
        max_translation_step_mm=5.0,
        max_rotation_step_degrees=5.0,
        coordinate_frame="dobot_base",
    ))


def test_valid_vision_target_converts_to_member4_canonical_target():
    converted = to_motion_target(vision_target())
    assert converted.__class__.__module__ == "motion.models"
    assert converted.position == RobotPosition(x=210.0, y=60.0, z=35.0)
    assert converted.coordinate_frame == "dobot_base"
    assert converted.valid is True


def test_member4_motion_planner_accepts_converted_vision_target():
    plan = planner().plan(
        RobotPosition(x=200.0, y=50.0, z=30.0, r=0.0),
        to_motion_target(vision_target()),
        "robot-1",
    )
    assert [(step.direction, step.distance) for step in plan.steps] == [
        ("+X", 5.0),
        ("+X", 5.0),
        ("+Y", 5.0),
        ("+Y", 5.0),
        ("+Z", 5.0),
    ]


@pytest.mark.parametrize("target", [
    vision_target(valid=False, status=TargetStatus.INVALID),
    vision_target(confidence=0.2, valid=False, status=TargetStatus.LOW_CONFIDENCE),
    vision_target(valid=False, status=TargetStatus.UNSTABLE),
    vision_target(valid=False, status=TargetStatus.STALE),
    vision_target(coordinate_frame="camera_frame"),
    vision_target(timestamp=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), valid=False, status=TargetStatus.STALE),
])
def test_invalid_vision_targets_cannot_cross_motion_boundary(target):
    with pytest.raises(MotionTargetConversionError):
        to_motion_target(target)


def test_non_finite_coordinates_are_rejected_at_motion_boundary():
    target = vision_target(position=Point3D(x=float("nan"), y=60.0, z=35.0))
    with pytest.raises(MotionTargetConversionError):
        to_motion_target(target)
