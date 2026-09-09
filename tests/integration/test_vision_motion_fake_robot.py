"""Software-only proof of the Member 3 -> Member 4 -> fake-robot boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.commands.models import Action
from app.gateway.adapter_manager import AdapterManager
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.robots.models import Robot
from fakes.fake_robot import FakeRobotAdapter
from motion.controller import ControllerStatus, MotionController
from motion.models import RobotPosition
from motion.planner import MotionPlanner, MotionPlanningPolicy
from vision.models import Point3D, RobotTarget, TargetStatus
from vision.target import (
    MotionTargetConversionError,
    TargetFreshnessChecker,
    to_motion_target,
)


class GatewaySpy:
    """Record gateway submissions while delegating to the existing gateway."""

    def __init__(self, gateway: UniversalGateway) -> None:
        self._gateway = gateway
        self.calls = []

    def process(self, command):
        self.calls.append(command)
        return self._gateway.process(command)


def system(initial: RobotPosition):
    robot = Robot(
        robot_id="fake-vision-1",
        name="Vision Integration Fake Robot",
        robot_type="robotic_arm",
        adapter_type="fake-vision-motion",
        capabilities=frozenset({Action.MOVE, Action.ROTATE}),
        status="ONLINE",
    )
    fake = FakeRobotAdapter(robot, initial)
    registry = RobotRegistry()
    registry.register(robot)
    adapters = AdapterManager(register_mock=False)
    adapters.register("fake-vision-motion", lambda configured_robot: fake)
    spy = GatewaySpy(UniversalGateway(registry, adapters))
    planner = MotionPlanner(MotionPlanningPolicy(
        max_translation_step_mm=5.0,
        max_rotation_step_degrees=5.0,
        coordinate_frame="dobot_base",
    ))
    return MotionController(planner, spy), spy, fake


def vision_target(**overrides) -> RobotTarget:
    values = {
        "target_id": "vision-target-integration-1",
        "class_name": "red_block",
        "position": Point3D(x=210.0, y=60.0, z=35.0),
        "confidence": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "dobot_base",
        "valid": True,
        "status": TargetStatus.VALID,
    }
    values.update(overrides)
    return RobotTarget(**values)


def test_valid_vision_target_runs_member4_controller_through_gateway_to_fake_robot():
    initial = RobotPosition(x=200.0, y=50.0, z=30.0, r=0.0)
    controller, gateway, fake = system(initial)

    canonical_target = to_motion_target(vision_target())
    result = controller.execute_target(initial, canonical_target, "fake-vision-1")

    assert result.status is ControllerStatus.SUCCESS
    assert [(step.direction, step.distance) for step in result.plan.steps] == [
        ("+X", 5.0), ("+X", 5.0), ("+Y", 5.0), ("+Y", 5.0), ("+Z", 5.0),
    ]
    assert [command.tasks[0].direction for command in result.prepared_commands] == [
        "+X", "+X", "+Y", "+Y", "+Z",
    ]
    assert gateway.calls == list(result.prepared_commands)
    assert len(fake.execution_history) == 5
    assert fake.position == RobotPosition(x=210.0, y=60.0, z=35.0, r=0.0)


def test_already_reached_vision_target_sends_no_gateway_command():
    initial = RobotPosition(x=200.0, y=50.0, z=30.0, r=0.0)
    controller, gateway, fake = system(initial)
    target = vision_target(position=Point3D(x=200.0, y=50.0, z=30.0))

    result = controller.execute_target(initial, to_motion_target(target), "fake-vision-1")

    assert result.status is ControllerStatus.NO_MOTION
    assert gateway.calls == []
    assert fake.execution_history == []
    assert fake.position == initial


@pytest.mark.parametrize(
    "target",
    [
        vision_target(valid=False, status=TargetStatus.INVALID),
        vision_target(confidence=0.2, valid=False, status=TargetStatus.LOW_CONFIDENCE),
        vision_target(valid=False, status=TargetStatus.UNSTABLE),
        vision_target(coordinate_frame="camera_frame"),
        vision_target(position=Point3D(x=float("nan"), y=60.0, z=35.0)),
    ],
    ids=["invalid", "low-confidence", "unstable", "wrong-frame", "non-finite"],
)
def test_invalid_vision_targets_are_rejected_before_gateway_execution(target):
    initial = RobotPosition(x=200.0, y=50.0, z=30.0, r=0.0)
    controller, gateway, fake = system(initial)

    with pytest.raises(MotionTargetConversionError):
        to_motion_target(target)

    assert gateway.calls == []
    assert fake.execution_history == []
    assert fake.position == initial


def test_stale_vision_target_is_rejected_before_planning_or_execution():
    initial = RobotPosition(x=200.0, y=50.0, z=30.0, r=0.0)
    controller, gateway, fake = system(initial)
    stale = vision_target(
        timestamp=(datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(),
        valid=False,
        status=TargetStatus.STALE,
    )

    fresh, _, _ = TargetFreshnessChecker(max_age_seconds=1.5).check(stale)
    assert fresh is False
    with pytest.raises(MotionTargetConversionError):
        to_motion_target(stale)

    assert gateway.calls == []
    assert fake.execution_history == []
    assert fake.position == initial
