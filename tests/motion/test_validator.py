"""Offline tests for pure motion planning validation."""

import ast
import inspect

import pytest

import motion.validator as validator_module
from app.commands.models import Action
from motion.models import MotionPlan, MotionStep
from motion.trajectory import Trajectory
from motion.validator import MotionValidationResult, MotionValidator


def move(sequence=0, direction="+X", distance=5.0, unit="mm"):
    return MotionStep(
        sequence=sequence,
        action=Action.MOVE,
        direction=direction,
        distance=distance,
        unit=unit,
    )


def rotate(sequence=1, direction="+R", angle=5.0, unit="degrees"):
    return MotionStep(
        sequence=sequence,
        action=Action.ROTATE,
        direction=direction,
        angle=angle,
        unit=unit,
    )


def plan(steps=None):
    return MotionPlan(
        plan_id="plan-1",
        robot_id="robot-1",
        target_id="target-1",
        steps=[move(), rotate()] if steps is None else steps,
    )


def trajectory(source=None):
    return Trajectory.from_plan(plan() if source is None else source)


def changed_trajectory(source, **updates):
    return Trajectory.from_plan(source).model_copy(update=updates, deep=True)


def test_valid_plan_trajectory_and_consistency_pass():
    source = plan()
    path = trajectory(source)
    assert MotionValidator.validate_plan(source) == MotionValidationResult(
        valid=True
    )
    assert MotionValidator.validate_trajectory(path).valid
    assert MotionValidator.validate(source, path).valid


def test_empty_plan_and_trajectory_pass():
    source = plan([])
    path = trajectory(source)
    assert MotionValidator.validate_plan(source).valid
    assert MotionValidator.validate_trajectory(path).valid
    assert MotionValidator.validate(source, path).valid


def test_none_and_wrong_types_fail_cleanly():
    assert MotionValidator.validate_plan(None).errors == (
        "MotionPlan is required.",
    )
    assert not MotionValidator.validate_plan({}).valid
    assert MotionValidator.validate_trajectory(None).errors == (
        "Trajectory is required.",
    )


@pytest.mark.parametrize(
    "sequences,message",
    [
        ((0, 0), "MotionPlan contains duplicate sequence numbers."),
        ((1, 0), "MotionPlan sequence numbers must be strictly increasing."),
        ((0, 2), "MotionPlan sequence numbers must be contiguous."),
    ],
)
def test_invalid_plan_sequences_fail(sequences, message):
    source = plan()
    for step, sequence in zip(source.steps, sequences):
        step.sequence = sequence
    result = MotionValidator.validate_plan(source)
    assert not result.valid
    assert message in result.errors


@pytest.mark.parametrize(
    "action,field,value,message",
    [
        (Action.MOVE, "direction", "+R", "MOVE step 0 has invalid direction."),
        (Action.MOVE, "distance", 0, "MOVE step 0 has non-positive distance."),
        (Action.MOVE, "distance", -1, "MOVE step 0 has non-positive distance."),
        (Action.MOVE, "distance", float("nan"), "MOVE step 0 has non-positive distance."),
        (Action.MOVE, "distance", float("inf"), "MOVE step 0 has non-positive distance."),
        (Action.MOVE, "unit", "cm", "MOVE step 0 must use unit 'mm'."),
        (Action.ROTATE, "direction", "+X", "ROTATE step 0 has invalid direction."),
        (Action.ROTATE, "angle", 0, "ROTATE step 0 has non-positive angle."),
        (Action.ROTATE, "angle", -1, "ROTATE step 0 has non-positive angle."),
        (Action.ROTATE, "angle", float("-inf"), "ROTATE step 0 has non-positive angle."),
        (Action.ROTATE, "unit", "rad", "ROTATE step 0 must use unit 'degrees'."),
    ],
)
def test_invalid_movement_semantics_fail(action, field, value, message):
    step = move() if action is Action.MOVE else rotate(sequence=0)
    setattr(step, field, value)
    result = MotionValidator.validate_plan(plan([step]))
    assert not result.valid
    assert message in result.errors


@pytest.mark.parametrize("field", ["plan_id", "robot_id"])
def test_missing_or_empty_required_plan_id_fails(field):
    source = plan()
    setattr(source, field, "")
    result = MotionValidator.validate_plan(source)
    assert not result.valid
    assert f"MotionPlan {field} must be a non-empty string." in result.errors


def test_non_movement_actions_pass_without_fake_motion_fields():
    source = plan([
        MotionStep(sequence=index, action=action)
        for index, action in enumerate(
            (
                Action.GRIP,
                Action.RELEASE,
                Action.HOME,
                Action.STOP,
                Action.GET_STATUS,
            )
        )
    ])
    assert MotionValidator.validate_plan(source).valid


@pytest.mark.parametrize("field", ["plan_id", "robot_id", "target_id"])
def test_inconsistent_identifiers_fail(field):
    source = plan()
    path = changed_trajectory(source, **{field: f"different-{field}"})
    result = MotionValidator.validate(source, path)
    assert not result.valid
    assert (
        f"Trajectory {field} does not match MotionPlan {field}."
        in result.errors
    )


def test_different_step_count_fails():
    source = plan()
    path = changed_trajectory(source, steps=(source.steps[0].model_copy(deep=True),))
    result = MotionValidator.validate(source, path)
    assert "Trajectory step count does not match MotionPlan step count." in result.errors


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", 3),
        ("action", Action.ROTATE),
        ("direction", "-X"),
        ("distance", 4.0),
        ("angle", 1.0),
        ("unit", "cm"),
    ],
)
def test_changed_step_field_fails_consistency(field, value):
    source = plan()
    path = trajectory(source)
    replacement = path.steps[0].model_copy(deep=True)
    setattr(replacement, field, value)
    changed = path.model_copy(
        update={"steps": (replacement, path.steps[1].model_copy(deep=True))}
    )
    result = MotionValidator.validate(source, changed)
    assert not result.valid
    assert (
        f"Trajectory step 0 {field} does not match "
        f"MotionPlan step 0 {field}."
    ) in result.errors


def test_different_step_order_fails():
    source = plan()
    path = trajectory(source)
    changed = path.model_copy(update={"steps": tuple(reversed(path.steps))})
    result = MotionValidator.validate(source, changed)
    assert not result.valid
    assert any("Trajectory step 0" in error for error in result.errors)


def test_validation_does_not_mutate_inputs():
    source = plan()
    path = trajectory(source)
    plan_before = source.model_dump()
    trajectory_before = path.model_dump()
    MotionValidator.validate(source, path)
    assert source.model_dump() == plan_before
    assert path.model_dump() == trajectory_before


def test_invalid_errors_are_deterministic():
    source = plan()
    source.steps[0].direction = "+R"
    source.steps[0].distance = 0
    first = MotionValidator.validate_plan(source)
    second = MotionValidator.validate_plan(source)
    assert first == second
    assert first.errors[:2] == (
        "MOVE step 0 has invalid direction.",
        "MOVE step 0 has non-positive distance.",
    )


def test_module_has_no_execution_hardware_or_perception_dependencies():
    tree = ast.parse(inspect.getsource(validator_module))
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
    forbidden = (
        "app.gateway",
        "app.adapters",
        "robots",
        "dobot",
        "dobotlink",
        "webots",
        "rclpy",
        "ros",
        "vision",
        "cv2",
        "yolo",
    )
    assert not any(name.lower().startswith(forbidden) for name in imported)


def test_validator_exposes_no_execution_method_or_physical_limits():
    assert {"execute", "run", "send", "send_to_gateway"}.isdisjoint(
        dir(MotionValidator)
    )
    source = inspect.getsource(validator_module)
    assert "DOBOT_MIN_" not in source
    assert "DOBOT_MAX_" not in source
