"""End-to-end software tests using the real gateway and an in-memory robot."""

from datetime import datetime, timezone

import pytest

import motion.controller as controller_module
from app.commands.models import Action, UniversalCommand
from app.gateway.adapter_manager import AdapterManager
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.robots.models import Robot
from motion.controller import ControllerStatus, MotionController
from motion.models import RobotPosition, RobotTarget
from motion.planner import MotionPlanner, MotionPlanningPolicy
from robots.interface import RobotInterface
from fakes.fake_robot import FakeRobotAdapter


def target(position: RobotPosition) -> RobotTarget:
    return RobotTarget(
        target_id="target-1",
        class_name="test-target",
        position=position,
        confidence=1.0,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        coordinate_frame="robot_base",
        valid=True,
    )


def system(
    initial: RobotPosition,
    *,
    translation_step: float = 5.0,
    rotation_step: float = 5.0,
    fail_on_execution: int | None = None,
):
    robot = Robot(
        robot_id="fake-1",
        name="Deterministic Fake Robot",
        robot_type="robotic_arm",
        adapter_type="fake-motion",
        capabilities=frozenset({
            Action.MOVE,
            Action.ROTATE,
            Action.HOME,
            Action.STOP,
            Action.GET_STATUS,
            Action.GRIP,
            Action.RELEASE,
        }),
        status="ONLINE",
    )
    fake = FakeRobotAdapter(
        robot,
        initial,
        fail_on_execution=fail_on_execution,
    )
    registry = RobotRegistry()
    registry.register(robot)
    adapters = AdapterManager(register_mock=False)
    # Returning this one instance preserves test-only in-memory state across
    # the Controller's one-command-per-step gateway calls.
    adapters.register("fake-motion", lambda configured_robot: fake)
    gateway = UniversalGateway(registry, adapters)
    planner = MotionPlanner(MotionPlanningPolicy(
        max_translation_step_mm=translation_step,
        max_rotation_step_degrees=rotation_step,
        coordinate_frame="robot_base",
    ))
    return MotionController(planner, gateway), gateway, fake


def history_semantics(fake):
    return [
        (record.action, record.direction, record.distance, record.angle)
        for record in fake.execution_history
    ]


def test_fake_adapter_satisfies_robot_interface():
    controller, gateway, fake = system(RobotPosition(x=0, y=0, z=0, r=0))
    assert isinstance(fake, RobotInterface)
    assert gateway.adapters.get(fake.robot) is fake
    assert controller.gateway is gateway


def test_complete_controller_gateway_fake_robot_pipeline():
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    controller, gateway, fake = system(initial)

    result = controller.execute_target(
        initial,
        target(RobotPosition(x=210, y=60, z=35, r=0)),
        "fake-1",
    )

    assert result.status is ControllerStatus.SUCCESS
    assert fake.position == RobotPosition(x=210, y=60, z=35, r=0)
    assert [step.direction for step in result.plan.steps] == [
        "+X", "+X", "+Y", "+Y", "+Z"
    ]
    assert [step.direction for step in result.trajectory.steps] == [
        "+X", "+X", "+Y", "+Y", "+Z"
    ]
    assert history_semantics(fake) == [
        (Action.MOVE, "+X", 5.0, None),
        (Action.MOVE, "+X", 5.0, None),
        (Action.MOVE, "+Y", 5.0, None),
        (Action.MOVE, "+Y", 5.0, None),
        (Action.MOVE, "+Z", 5.0, None),
    ]
    assert all(plan.simulated for plan in result.gateway_results)
    assert all(plan.adapter_type == "fake-motion" for plan in result.gateway_results)
    assert result.submitted_commands == result.prepared_commands
    assert controller.gateway is gateway


def test_rotation_updates_r_in_deterministic_steps():
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    controller, _, fake = system(initial)
    result = controller.execute_target(
        initial,
        target(RobotPosition(x=200, y=50, z=30, r=10)),
        "fake-1",
    )
    assert result.status is ControllerStatus.SUCCESS
    assert [record.direction for record in fake.execution_history] == ["+R", "+R"]
    assert [record.angle for record in fake.execution_history] == [5.0, 5.0]
    assert fake.position.r == pytest.approx(10.0)


def test_multi_axis_order_and_final_state():
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    controller, _, fake = system(initial)
    result = controller.execute_target(
        initial,
        target(RobotPosition(x=210, y=45, z=32, r=0)),
        "fake-1",
    )
    assert result.status is ControllerStatus.SUCCESS
    assert [record.direction for record in fake.execution_history] == [
        "+X", "+X", "-Y", "+Z"
    ]
    assert fake.position.x == pytest.approx(210)
    assert fake.position.y == pytest.approx(45)
    assert fake.position.z == pytest.approx(32)
    assert fake.position.r == pytest.approx(0)


def test_injected_failure_stops_later_gateway_submissions_and_state_updates():
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    controller, _, fake = system(initial, fail_on_execution=3)
    result = controller.execute_target(
        initial,
        target(RobotPosition(x=210, y=60, z=35, r=0)),
        "fake-1",
    )

    assert result.status is ControllerStatus.EXECUTION_FAILED
    assert result.failed_step_sequence == 2
    assert result.remaining_step_sequences == (3, 4)
    assert len(result.submitted_commands) == 3
    assert len(result.gateway_results) == 3
    assert [record.succeeded for record in fake.execution_history] == [True, True, False]
    assert fake.position == RobotPosition(x=210, y=50, z=30, r=0)
    assert "Injected failure on execution 3" in result.error


def test_pre_execution_command_validation_failure_executes_nothing(monkeypatch):
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    controller, _, fake = system(initial)
    real_validate = controller_module.validate_command
    validation_count = 0

    def reject_third(payload):
        nonlocal validation_count
        validation_count += 1
        if validation_count == 3:
            payload["tasks"][0]["distance"] = -1
        return real_validate(payload)

    monkeypatch.setattr(controller_module, "validate_command", reject_third)
    result = controller.execute_target(
        initial,
        target(RobotPosition(x=210, y=60, z=35, r=0)),
        "fake-1",
    )
    assert result.status is ControllerStatus.VALIDATION_FAILED
    assert validation_count == 3
    assert fake.execution_history == []
    assert fake.position == initial


def test_no_motion_does_not_call_fake_robot():
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    controller, _, fake = system(initial)
    result = controller.execute_target(initial, target(initial), "fake-1")
    assert result.status is ControllerStatus.NO_MOTION
    assert result.plan.steps == []
    assert result.trajectory.steps == ()
    assert fake.execution_history == []
    assert fake.position == initial


def test_fake_home_stop_status_grip_and_release_use_existing_commands():
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    _, gateway, fake = system(initial)
    fake.position = RobotPosition(x=205, y=45, z=31, r=5)

    def command(action):
        return UniversalCommand(robot_id="fake-1", tasks=[{"action": action}])

    assert gateway.process(command(Action.HOME)).status.value == "READY"
    assert fake.position == initial
    assert gateway.process(command(Action.GRIP)).status.value == "READY"
    assert fake.gripped is True
    assert gateway.process(command(Action.RELEASE)).status.value == "READY"
    assert fake.gripped is False
    assert gateway.process(command(Action.STOP)).status.value == "READY"
    assert fake.stopped is True
    status = gateway.process(command(Action.GET_STATUS))
    assert status.status.value == "READY"
    assert '"state": "STOPPED"' in status.results[0]


def test_fake_rejects_malformed_moves_without_state_change():
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    _, _, fake = system(initial)
    malformed = UniversalCommand(
        robot_id="fake-1",
        tasks=[{"action": "MOVE", "direction": "X", "distance": 5, "unit": "mm"}],
    )
    assert fake.validate(malformed) is False
    assert fake.execution_history == []
    assert fake.position == initial

