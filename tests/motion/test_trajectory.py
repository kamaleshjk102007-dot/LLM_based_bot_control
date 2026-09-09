"""Offline tests for deterministic, execution-free trajectories."""

import ast
import inspect

import pytest
from pydantic import ValidationError

import motion.trajectory as trajectory_module
from app.commands.models import Action
from motion.models import MotionPlan, MotionStep
from motion.trajectory import Trajectory, TrajectoryError


def move(sequence=0, direction="+X", distance=5.0):
    return MotionStep(
        sequence=sequence,
        action=Action.MOVE,
        direction=direction,
        distance=distance,
        unit="mm",
    )


def rotate(sequence=1, direction="+R", angle=5.0):
    return MotionStep(
        sequence=sequence,
        action=Action.ROTATE,
        direction=direction,
        angle=angle,
        unit="degrees",
    )


def plan(steps=None):
    return MotionPlan(
        plan_id="plan-1",
        robot_id="robot-1",
        target_id="target-1",
        steps=[move(), rotate()] if steps is None else steps,
    )


def test_from_plan_preserves_metadata_and_exact_step_semantics():
    source = plan()
    trajectory = Trajectory.from_plan(source)

    assert trajectory.plan_id == source.plan_id
    assert trajectory.robot_id == source.robot_id
    assert trajectory.target_id == source.target_id
    assert [step.model_dump() for step in trajectory.steps] == [
        step.model_dump() for step in source.steps
    ]
    assert [step.sequence for step in trajectory.steps] == [0, 1]
    assert [step.action for step in trajectory.steps] == [Action.MOVE, Action.ROTATE]
    assert [step.direction for step in trajectory.steps] == ["+X", "+R"]
    assert trajectory.steps[0].distance == 5.0
    assert trajectory.steps[1].angle == 5.0
    assert [step.unit for step in trajectory.steps] == ["mm", "degrees"]


def test_multiple_steps_remain_in_original_order():
    steps = [
        move(0, "+X", 1),
        move(1, "-Y", 2),
        move(2, "+Z", 3),
        rotate(3, "-R", 4),
    ]
    trajectory = Trajectory.from_plan(plan(steps))
    assert [step.direction for step in trajectory.steps] == [
        "+X", "-Y", "+Z", "-R"
    ]


def test_source_plan_and_steps_are_not_mutated_or_shared():
    source = plan()
    before = source.model_dump()
    trajectory = Trajectory.from_plan(source)

    assert source.model_dump() == before
    assert all(
        trajectory_step is not source_step
        for trajectory_step, source_step in zip(trajectory.steps, source.steps)
    )
    trajectory.steps[0].direction = "-X"
    assert source.steps[0].direction == "+X"


def test_get_steps_returns_independent_copies():
    trajectory = Trajectory.from_plan(plan())
    returned = trajectory.get_steps()
    returned[0].direction = "-X"
    assert trajectory.steps[0].direction == "+X"


def test_empty_plan_creates_empty_trajectory():
    trajectory = Trajectory.from_plan(plan([]))
    assert trajectory.steps == ()
    assert trajectory.is_empty()
    assert trajectory.total_steps() == 0


def test_helpers_report_nonempty_trajectory():
    trajectory = Trajectory.from_plan(plan())
    assert not trajectory.is_empty()
    assert trajectory.total_steps() == 2


@pytest.mark.parametrize("sequences", [(0, 0), (1, 0), (0, 2)])
def test_invalid_order_or_noncontiguous_sequence_is_rejected(sequences):
    with pytest.raises(ValidationError):
        Trajectory(
            trajectory_id="trajectory-1",
            plan_id="plan-1",
            robot_id="robot-1",
            steps=tuple(move(sequence) for sequence in sequences),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("trajectory_id", ""),
        ("plan_id", " "),
        ("robot_id", ""),
        ("target_id", " "),
    ],
)
def test_invalid_trajectory_metadata_is_rejected(field, value):
    values = {
        "trajectory_id": "trajectory-1",
        "plan_id": "plan-1",
        "robot_id": "robot-1",
        "target_id": "target-1",
        "steps": (),
    }
    values[field] = value
    with pytest.raises(ValidationError):
        Trajectory(**values)


@pytest.mark.parametrize("field", ["trajectory_id", "plan_id", "robot_id"])
def test_required_trajectory_metadata_cannot_be_missing(field):
    values = {
        "trajectory_id": "trajectory-1",
        "plan_id": "plan-1",
        "robot_id": "robot-1",
    }
    del values[field]
    with pytest.raises(ValidationError):
        Trajectory(**values)


def test_none_and_non_plan_inputs_are_rejected():
    with pytest.raises(TrajectoryError, match="valid MotionPlan"):
        Trajectory.from_plan(None)
    with pytest.raises(TrajectoryError, match="valid MotionPlan"):
        Trajectory.from_plan({"plan_id": "plan-1"})


def test_mutated_invalid_plan_metadata_is_rejected():
    source = plan()
    source.plan_id = ""
    with pytest.raises(TrajectoryError, match="Invalid MotionPlan"):
        Trajectory.from_plan(source)


def test_mutated_invalid_step_is_rejected():
    source = plan()
    source.steps[0].distance = -1
    with pytest.raises(TrajectoryError, match="Invalid MotionPlan"):
        Trajectory.from_plan(source)


@pytest.mark.parametrize("sequences", [(0, 0), (1, 0)])
def test_mutated_invalid_plan_order_is_rejected(sequences):
    source = plan()
    for step, sequence in zip(source.steps, sequences):
        step.sequence = sequence
    with pytest.raises(TrajectoryError, match="Invalid MotionPlan"):
        Trajectory.from_plan(source)


def test_noncontiguous_valid_plan_is_rejected_by_trajectory():
    source = plan([move(0), rotate(2)])
    with pytest.raises(TrajectoryError, match="contiguous"):
        Trajectory.from_plan(source)


def test_same_plan_produces_same_trajectory():
    source = plan()
    first = Trajectory.from_plan(source)
    second = Trajectory.from_plan(source)
    assert first == second
    assert first.trajectory_id == second.trajectory_id


def test_target_id_remains_optional():
    source = MotionPlan(plan_id="plan-1", robot_id="robot-1", steps=[])
    assert Trajectory.from_plan(source).target_id is None


def test_module_has_no_execution_or_robot_integration_imports():
    tree = ast.parse(inspect.getsource(trajectory_module))
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
        "DobotLink",
        "webots",
    )
    assert not any(name.startswith(forbidden) for name in imported)


def test_trajectory_exposes_no_execution_method():
    forbidden = {"execute", "run", "send", "send_to_gateway"}
    assert forbidden.isdisjoint(dir(Trajectory))
