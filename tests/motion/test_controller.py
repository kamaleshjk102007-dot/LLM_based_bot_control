"""Offline tests for the motion orchestration boundary."""

import ast
import inspect
from datetime import datetime, timezone

import pytest

import motion.controller as controller_module
from app.commands.models import Action, UniversalCommand
from app.commands.validator import CommandValidationError
from app.gateway.command_router import ExecutionPlan, PlanStatus
from motion.controller import ControllerStatus, MotionController
from motion.models import MotionPlan, MotionStep, RobotPosition, RobotTarget
from motion.planner import MotionPlanner, MotionPlanningPolicy
from motion.trajectory import Trajectory
from motion.validator import MotionValidationResult, MotionValidator


class FakeGateway:
    def __init__(self, outcomes=None, events=None):
        self.outcomes = list(outcomes or [])
        self.calls = []
        self.events = events

    def process(self, command):
        if self.events is not None:
            self.events.append(("execute", command.tasks[0].direction))
        self.calls.append(command)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return gateway_result(PlanStatus.READY)


def gateway_result(status, reason=None):
    return ExecutionPlan(
        plan_id="gateway-plan",
        status=status,
        reason=reason,
    )


def target(*, x=12.0, y=2.0, z=0.0, valid=True):
    return RobotTarget(
        target_id="target-1",
        class_name="cube",
        position=RobotPosition(x=x, y=y, z=z),
        confidence=0.9,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        coordinate_frame="robot_base",
        valid=valid,
    )


def planner():
    return MotionPlanner(
        MotionPlanningPolicy(
            max_translation_step_mm=5,
            max_rotation_step_degrees=5,
            coordinate_frame="robot_base",
        )
    )


def controller(gateway=None, validator=MotionValidator):
    return MotionController(planner(), gateway or FakeGateway(), validator)


def test_valid_target_runs_complete_pipeline_and_propagates_robot_id():
    gateway = FakeGateway()
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )

    assert result.status is ControllerStatus.SUCCESS
    assert isinstance(result.plan, MotionPlan)
    assert isinstance(result.trajectory, Trajectory)
    assert all(isinstance(command, UniversalCommand) for command in result.prepared_commands)
    assert all(command.robot_id == "robot-1" for command in result.prepared_commands)
    assert [command.tasks[0].direction for command in result.prepared_commands] == [
        "+X", "+X", "+X", "+Y"
    ]
    assert [command.tasks[0].distance for command in result.prepared_commands] == [
        5.0, 5.0, 2.0, 2.0
    ]
    assert gateway.calls == list(result.prepared_commands)
    assert result.submitted_commands == result.prepared_commands
    assert len(result.gateway_results) == 4
    assert result.successful


def test_commands_execute_strictly_in_trajectory_order():
    gateway = FakeGateway()
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(x=-6, y=7, z=2),
        "robot-1",
    )
    expected = [
        (Action.MOVE, "-X", 5.0),
        (Action.MOVE, "-X", 1.0),
        (Action.MOVE, "+Y", 5.0),
        (Action.MOVE, "+Y", 2.0),
        (Action.MOVE, "+Z", 2.0),
    ]
    actual = [
        (command.tasks[0].action, command.tasks[0].direction, command.tasks[0].distance)
        for command in gateway.calls
    ]
    assert actual == expected
    assert [
        step.sequence for step in result.trajectory.steps
    ] == list(range(len(expected)))


def test_empty_trajectory_returns_no_motion_without_gateway_call():
    gateway = FakeGateway()
    result = controller(gateway).execute_target(
        RobotPosition(x=1, y=2, z=3),
        target(x=1, y=2, z=3),
        "robot-1",
    )
    assert result.status is ControllerStatus.NO_MOTION
    assert result.successful
    assert result.trajectory.steps == ()
    assert result.prepared_commands == ()
    assert gateway.calls == []


def test_invalid_target_causes_no_execution():
    gateway = FakeGateway()
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(valid=False),
        "robot-1",
    )
    assert result.status is ControllerStatus.PLANNING_FAILED
    assert gateway.calls == []


def test_invalid_motion_plan_causes_no_execution():
    class InvalidPlanner:
        def plan(self, current, destination, robot_id):
            result = MotionPlan(
                plan_id="plan-1",
                robot_id=robot_id,
                target_id=destination.target_id,
                steps=[MotionStep(
                    sequence=0,
                    action=Action.MOVE,
                    direction="+X",
                    distance=1,
                    unit="mm",
                )],
            )
            result.steps[0].distance = -1
            return result

    gateway = FakeGateway()
    result = MotionController(InvalidPlanner(), gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )
    assert result.status is ControllerStatus.VALIDATION_FAILED
    assert gateway.calls == []


def test_invalid_trajectory_validation_causes_no_execution(monkeypatch):
    original = Trajectory.from_plan

    def invalid_trajectory(plan):
        result = original(plan)
        result.steps[0].direction = "+R"
        return result

    monkeypatch.setattr(
        controller_module.Trajectory,
        "from_plan",
        invalid_trajectory,
    )
    gateway = FakeGateway()
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )
    assert result.status is ControllerStatus.VALIDATION_FAILED
    assert gateway.calls == []


def test_explicit_validator_failure_happens_before_execution():
    class RejectingValidator:
        @classmethod
        def validate(cls, plan, trajectory):
            return MotionValidationResult(
                valid=False,
                errors=("trajectory rejected",),
            )

    gateway = FakeGateway()
    result = controller(gateway, RejectingValidator).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )
    assert result.status is ControllerStatus.VALIDATION_FAILED
    assert "trajectory rejected" in result.error
    assert gateway.calls == []


def test_all_commands_are_validated_before_first_gateway_call(monkeypatch):
    events = []
    real_validate = controller_module.validate_command

    def tracking_validate(payload):
        events.append(("validate", payload["tasks"][0]["direction"]))
        return real_validate(payload)

    monkeypatch.setattr(controller_module, "validate_command", tracking_validate)
    gateway = FakeGateway(events=events)
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )

    assert result.status is ControllerStatus.SUCCESS
    assert [event[0] for event in events] == [
        "validate", "validate", "validate", "validate",
        "execute", "execute", "execute", "execute",
    ]


def test_invalid_generated_command_prevents_all_execution(monkeypatch):
    real_validate = controller_module.validate_command
    calls = 0

    def fail_later(payload):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise CommandValidationError("bad third command")
        return real_validate(payload)

    monkeypatch.setattr(controller_module, "validate_command", fail_later)
    gateway = FakeGateway()
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )
    assert result.status is ControllerStatus.VALIDATION_FAILED
    assert calls == 3
    assert gateway.calls == []


def test_first_gateway_failure_stops_remaining_commands():
    gateway = FakeGateway([
        gateway_result(PlanStatus.READY),
        gateway_result(PlanStatus.REJECTED, "adapter rejected step"),
        gateway_result(PlanStatus.READY),
    ])
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )

    assert result.status is ControllerStatus.EXECUTION_FAILED
    assert not result.successful
    assert len(gateway.calls) == 2
    assert len(result.submitted_commands) == 2
    assert len(result.gateway_results) == 2
    assert result.failed_step_sequence == 1
    assert result.remaining_step_sequences == (2, 3)
    assert result.error == "adapter rejected step"


def test_gateway_exception_becomes_failure_and_stops():
    gateway = FakeGateway([RuntimeError("offline")])
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )
    assert result.status is ControllerStatus.EXECUTION_FAILED
    assert result.failed_step_sequence == 0
    assert result.remaining_step_sequences == (1, 2, 3)
    assert result.gateway_results == ()
    assert "RuntimeError: offline" in result.error
    assert len(gateway.calls) == 1


def test_invalid_gateway_result_becomes_failure():
    gateway = FakeGateway([object()])
    result = controller(gateway).execute_target(
        RobotPosition(x=0, y=0, z=0),
        target(),
        "robot-1",
    )
    assert result.status is ControllerStatus.EXECUTION_FAILED
    assert "invalid execution result" in result.error
    assert len(gateway.calls) == 1


def test_same_input_produces_same_command_order():
    current = RobotPosition(x=0, y=0, z=0)
    destination = target()
    first = controller().execute_target(current, destination, "robot-1")
    second = controller().execute_target(current, destination, "robot-1")
    first_tasks = [
        command.tasks[0].model_dump() for command in first.prepared_commands
    ]
    second_tasks = [
        command.tasks[0].model_dump() for command in second.prepared_commands
    ]
    assert first_tasks == second_tasks


def test_controller_does_not_mutate_planning_inputs_or_artifacts():
    current = RobotPosition(x=0, y=0, z=0)
    destination = target()
    current_before = current.model_dump()
    target_before = destination.model_dump()
    result = controller().execute_target(current, destination, "robot-1")

    plan_before = result.plan.model_dump()
    trajectory_before = result.trajectory.model_dump()
    assert current.model_dump() == current_before
    assert destination.model_dump() == target_before
    assert result.plan.model_dump() == plan_before
    assert result.trajectory.model_dump() == trajectory_before


def test_controller_has_only_generic_execution_dependencies():
    tree = ast.parse(inspect.getsource(controller_module))
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
    lowered = {name.lower() for name in imported}
    assert not any(name.startswith(forbidden) for name in lowered)


def test_controller_has_no_robot_or_verification_methods():
    forbidden = {
        "move",
        "calibrate",
        "verify",
        "measure_pose",
        "connect",
        "stop_robot",
    }
    assert forbidden.isdisjoint(dir(MotionController))
