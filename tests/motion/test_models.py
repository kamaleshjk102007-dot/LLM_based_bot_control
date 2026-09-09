"""Offline tests for planning data and existing command compatibility."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.commands.models import Action
from app.commands.validator import validate_command
from motion.models import MotionPlan, MotionStep, RobotPosition, RobotTarget


def target_data():
    return dict(
        target_id="t", class_name="cube", position=dict(x=1, y=2, z=3),
        confidence=0.9, timestamp=datetime.now(timezone.utc),
        coordinate_frame="robot_base", valid=True,
    )


def test_positions():
    assert RobotPosition(x=1, y=2, z=3).r is None
    assert RobotPosition(x=1, y=2, z=3, r=-5).r == -5


@pytest.mark.parametrize("axis", ["x", "y", "z", "r"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "1", True])
def test_invalid_position(axis, value):
    data = dict(x=1, y=2, z=3, r=0)
    data[axis] = value
    with pytest.raises(ValidationError):
        RobotPosition(**data)


def test_valid_target():
    assert RobotTarget(**target_data()).valid is True
    data = target_data()
    data["valid"] = False
    assert RobotTarget(**data).valid is False


@pytest.mark.parametrize("field,value", [
    ("target_id", ""), ("class_name", " "), ("coordinate_frame", ""),
    ("confidence", -0.1), ("confidence", 1.1), ("confidence", float("nan")),
    ("position", dict(x=float("inf"), y=2, z=3)), ("position", {}),
    ("valid", "true"), ("timestamp", 123),
])
def test_invalid_target(field, value):
    data = target_data()
    data[field] = value
    with pytest.raises(ValidationError):
        RobotTarget(**data)


def test_target_requires_valid():
    data = target_data()
    del data["valid"]
    with pytest.raises(ValidationError):
        RobotTarget(**data)


@pytest.mark.parametrize("fields", [
    dict(action="MOVE", direction="-X", distance=5, unit="mm"),
    dict(action="ROTATE", direction="-R", angle=5, unit="degrees"),
    *[dict(action=a) for a in ("GRIP", "RELEASE", "HOME", "STOP", "GET_STATUS")],
])
def test_step_compatible_with_existing_command(fields):
    step = MotionStep(sequence=0, **fields)
    assert isinstance(step.action, Action)
    payload = step.model_dump(mode="json", exclude={"sequence"}, exclude_none=True)
    task = validate_command({"tasks": [payload]}).tasks[0]
    assert task.action is step.action
    assert task.direction == step.direction


@pytest.mark.parametrize("action,quantity,unit", [
    ("MOVE", "distance", "mm"), ("ROTATE", "angle", "deg"),
])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "5", True, None])
def test_invalid_movement_quantity(action, quantity, unit, value):
    with pytest.raises(ValidationError):
        MotionStep(sequence=0, action=action, direction="X", unit=unit, **{quantity: value})


@pytest.mark.parametrize("fields", [
    dict(action="MOVE", distance=1, unit="mm"),
    dict(action="MOVE", direction=" ", distance=1, unit="mm"),
    dict(action="MOVE", direction="X", distance=1),
    dict(action="MOVE", direction="X", distance=1, unit="mm", angle=1),
    dict(action="ROTATE", angle=1, unit="deg"),
    dict(action="ROTATE", direction="R", angle=1, unit="mm"),
    dict(action="ROTATE", direction="R", angle=361, unit="deg"),
    dict(action="ROTATE", direction="R", angle=1, unit="deg", distance=1),
    dict(action="GRIP", distance=1, unit="mm"),
    dict(action="PICK"),
])
def test_invalid_step_fields(fields):
    with pytest.raises(ValidationError):
        MotionStep(sequence=0, **fields)


@pytest.mark.parametrize("sequence", [-1, True, "1", 1.5])
def test_invalid_sequence(sequence):
    with pytest.raises(ValidationError):
        MotionStep(sequence=sequence, action="HOME")


def test_plan_retains_order():
    steps = [MotionStep(sequence=1, action="GRIP"), MotionStep(sequence=3, action="RELEASE")]
    plan = MotionPlan(plan_id="p", robot_id="r", target_id="t", steps=steps)
    assert plan.steps == steps
    assert plan.target_id == "t"
    assert MotionPlan(plan_id="p", robot_id="r", steps=steps).target_id is None


@pytest.mark.parametrize("field,value", [
    ("plan_id", ""), ("robot_id", " "), ("target_id", ""),
])
def test_invalid_plan(field, value):
    data = dict(plan_id="p", robot_id="r", steps=[dict(sequence=0, action="HOME")])
    data[field] = value
    with pytest.raises(ValidationError):
        MotionPlan(**data)


@pytest.mark.parametrize("field", ["plan_id", "robot_id"])
def test_missing_plan_fields(field):
    data = dict(plan_id="p", robot_id="r", steps=[dict(sequence=0, action="HOME")])
    del data[field]
    with pytest.raises(ValidationError):
        MotionPlan(**data)


@pytest.mark.parametrize("sequences", [(1, 1), (2, 1)])
def test_invalid_order(sequences):
    with pytest.raises(ValidationError):
        MotionPlan(plan_id="p", robot_id="r", steps=[
            MotionStep(sequence=s, action="HOME") for s in sequences
        ])


def test_empty_plan_represents_no_movement():
    assert MotionPlan(plan_id="p", robot_id="r", target_id="t").steps == []
