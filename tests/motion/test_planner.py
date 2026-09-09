"""Offline tests for deterministic Cartesian motion planning."""

import ast
import inspect
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.commands.models import Action
import motion.planner as planner_module
from motion.models import RobotPosition, RobotTarget
from motion.planner import MotionPlanner, MotionPlanningError, MotionPlanningPolicy


def target(*, x=0.0, y=0.0, z=0.0, r=None, frame="robot_base", valid=True):
    return RobotTarget(
        target_id="target-1",
        class_name="cube",
        position=RobotPosition(x=x, y=y, z=z, r=r),
        confidence=0.9,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        coordinate_frame=frame,
        valid=valid,
    )


def planner(translation=5.0, rotation=5.0, frame="robot_base"):
    return MotionPlanner(
        MotionPlanningPolicy(
            max_translation_step_mm=translation,
            max_rotation_step_degrees=rotation,
            coordinate_frame=frame,
        )
    )


@pytest.mark.parametrize(
    "axis,value,direction",
    [
        ("x", 3.0, "+X"), ("x", -3.0, "-X"),
        ("y", 3.0, "+Y"), ("y", -3.0, "-Y"),
        ("z", 3.0, "+Z"), ("z", -3.0, "-Z"),
    ],
)
def test_single_axis_translation(axis, value, direction):
    coordinates = dict(x=0.0, y=0.0, z=0.0)
    coordinates[axis] = value
    step = planner().plan(RobotPosition(x=0, y=0, z=0), target(**coordinates), "r").steps[0]
    assert (step.action, step.direction, step.distance, step.unit) == (
        Action.MOVE, direction, 3.0, "mm"
    )


@pytest.mark.parametrize("value,direction", [(3.0, "+R"), (-3.0, "-R")])
def test_rotation(value, direction):
    step = planner().plan(
        RobotPosition(x=0, y=0, z=0, r=0),
        target(r=value),
        "r",
    ).steps[0]
    assert (step.action, step.direction, step.angle, step.unit) == (
        Action.ROTATE, direction, 3.0, "degrees"
    )


def test_fixed_axis_order_and_sequences():
    plan = planner().plan(
        RobotPosition(x=0, y=0, z=0, r=0),
        target(x=1, y=-2, z=3, r=-4),
        "r",
    )
    assert [step.direction for step in plan.steps] == ["+X", "-Y", "+Z", "-R"]
    assert [step.sequence for step in plan.steps] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "delta,maximum,expected",
    [
        (20.0, 5.0, [5.0, 5.0, 5.0, 5.0]),
        (5.0, 5.0, [5.0]),
        (3.0, 5.0, [3.0]),
        (12.0, 5.0, [5.0, 5.0, 2.0]),
    ],
)
def test_step_splitting(delta, maximum, expected):
    plan = planner(translation=maximum).plan(
        RobotPosition(x=0, y=0, z=0), target(x=delta), "r"
    )
    assert [step.distance for step in plan.steps] == expected
    assert all(step.distance > 0 for step in plan.steps)


def test_zero_axes_and_no_op_create_no_fake_steps():
    y_only = planner().plan(
        RobotPosition(x=1, y=2, z=3), target(x=1, y=5, z=3), "r"
    )
    assert [step.direction for step in y_only.steps] == ["+Y"]
    no_op = planner().plan(
        RobotPosition(x=1, y=2, z=3), target(x=1, y=2, z=3), "r"
    )
    assert no_op.steps == []


def test_floating_noise_is_zero():
    policy = MotionPlanningPolicy(
        max_translation_step_mm=5,
        max_rotation_step_degrees=5,
        coordinate_frame="robot_base",
        translation_zero_tolerance_mm=1e-6,
    )
    plan = MotionPlanner(policy).plan(
        RobotPosition(x=1, y=2, z=3),
        target(x=1 + 1e-8, y=2, z=3),
        "r",
    )
    assert plan.steps == []


def test_plan_is_deterministic():
    current = RobotPosition(x=0, y=0, z=0, r=0)
    destination = target(x=6, y=2, z=-1, r=3)
    first = planner().plan(current, destination, "robot-1")
    second = planner().plan(current, destination, "robot-1")
    assert first == second


def test_rotation_requires_current_r():
    with pytest.raises(MotionPlanningError, match="requires R"):
        planner().plan(RobotPosition(x=0, y=0, z=0), target(r=1), "r")


def test_invalid_or_incompatible_input_is_rejected():
    with pytest.raises(MotionPlanningError, match="marked invalid"):
        planner().plan(RobotPosition(x=0, y=0, z=0), target(valid=False), "r")
    with pytest.raises(MotionPlanningError, match="coordinate frame"):
        planner().plan(RobotPosition(x=0, y=0, z=0), target(frame="camera"), "r")
    with pytest.raises(MotionPlanningError, match="robot_id"):
        planner().plan(RobotPosition(x=0, y=0, z=0), target(), " ")
    with pytest.raises(MotionPlanningError, match="current"):
        planner().plan({"x": 0, "y": 0, "z": 0}, target(), "r")


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_translation_step_mm", 0),
        ("max_translation_step_mm", float("inf")),
        ("max_rotation_step_degrees", -1),
        ("max_rotation_step_degrees", 361),
        ("coordinate_frame", ""),
        ("translation_zero_tolerance_mm", -1),
        ("translation_zero_tolerance_mm", 5),
        ("rotation_zero_tolerance_degrees", 5),
    ],
)
def test_invalid_policy(field, value):
    data = dict(
        max_translation_step_mm=5,
        max_rotation_step_degrees=5,
        coordinate_frame="robot_base",
    )
    data[field] = value
    with pytest.raises(ValidationError):
        MotionPlanningPolicy(**data)


def test_invalid_target_values_are_rejected_by_model():
    with pytest.raises(ValidationError):
        target(x=float("nan"))
    with pytest.raises(ValidationError):
        RobotTarget(
            target_id="t",
            class_name="cube",
            position=RobotPosition(x=0, y=0, z=0),
            confidence=1.1,
            timestamp=datetime.now(timezone.utc),
            coordinate_frame="robot_base",
            valid=True,
        )


def test_mutated_input_is_revalidated_at_planner_boundary():
    destination = target(x=1)
    destination.confidence = float("nan")
    with pytest.raises(MotionPlanningError, match="Invalid planning input"):
        planner().plan(RobotPosition(x=0, y=0, z=0), destination, "r")


def test_planner_has_no_gateway_or_adapter_dependency():
    tree = ast.parse(inspect.getsource(planner_module))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any(
        name.startswith(("app.gateway", "app.adapters")) for name in imported
    )
