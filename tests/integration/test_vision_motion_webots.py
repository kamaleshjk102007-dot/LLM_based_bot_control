"""Live simulation proof of the Member 3 -> Member 4 -> Webots boundary.

These tests use the repository's existing Webots world and TCP adapter.  They
never import physical DOBOT or DobotLink code.  Start the world and press Play,
then set ``RUN_WEBOTS_INTEGRATION_TESTS=1`` to run them.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

import motion.controller as controller_module
from app.adapters.webots import WebotsClient, WebotsRobotAdapter
from app.commands.models import Action
from app.commands.validator import CommandValidationError
from app.gateway.adapter_manager import AdapterManager
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.robots.models import Robot
from motion.controller import ControllerStatus, MotionController
from motion.models import RobotPosition, RobotTarget as MotionRobotTarget
from motion.planner import MotionPlanner, MotionPlanningPolicy
from vision.models import Point3D, RobotTarget, TargetStatus
from vision.target import MotionTargetConversionError, to_motion_target


RUN_WEBOTS = os.getenv("RUN_WEBOTS_INTEGRATION_TESTS") == "1"
requires_webots = pytest.mark.skipif(
    not RUN_WEBOTS,
    reason=(
        "set RUN_WEBOTS_INTEGRATION_TESTS=1 after opening "
        "simulation/webots/worlds/magician_lite.wbt and pressing Play"
    ),
)


class GatewaySpy:
    """Observe existing-gateway calls without changing its behavior."""

    def __init__(self, gateway: UniversalGateway) -> None:
        self._gateway = gateway
        self.calls = []

    def process(self, command):
        self.calls.append(command)
        return self._gateway.process(command)


def webots_system():
    robot = Robot(
        robot_id="webots_001",
        name="Virtual Magician Lite",
        robot_type="robotic_arm",
        manufacturer="DOBOT-inspired",
        model="simplified_visual_model",
        adapter_type="webots",
        capabilities=frozenset({Action.MOVE, Action.ROTATE, Action.HOME, Action.STOP, Action.GET_STATUS}),
        status="ONLINE",
    )
    client = WebotsClient(timeout=30.0)
    adapter = WebotsRobotAdapter(robot, client)
    registry = RobotRegistry()
    registry.register(robot)
    adapters = AdapterManager(register_mock=False)
    adapters.register("webots", lambda configured_robot: adapter)
    gateway = GatewaySpy(UniversalGateway(registry, adapters))
    planner = MotionPlanner(MotionPlanningPolicy(
        max_translation_step_mm=5.0,
        max_rotation_step_degrees=5.0,
        coordinate_frame="dobot_base",
    ))
    return MotionController(planner, gateway), gateway, client


def measured_position(client: WebotsClient) -> RobotPosition:
    status = client.request({"type": "status"})
    position = status["end_effector_position_mm"]
    return RobotPosition(
        x=float(position["x"]),
        y=float(position["y"]),
        z=float(position["z"]),
        r=float(status["measured_wrist_r_degrees"]),
    )


def vision_target(position: RobotPosition, **overrides) -> RobotTarget:
    values = {
        "target_id": "vision-webots-target",
        "class_name": "red_block",
        "position": Point3D(x=position.x, y=position.y, z=position.z),
        "confidence": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "dobot_base",
        "valid": True,
        "status": TargetStatus.VALID,
    }
    values.update(overrides)
    return RobotTarget(**values)


def canonical_rotation_target(position: RobotPosition) -> MotionRobotTarget:
    """R is a robot-state value, not a value currently emitted by Member 3."""

    return MotionRobotTarget(
        target_id="webots-rotation-target",
        class_name="wrist-orientation",
        position=position,
        confidence=1.0,
        timestamp=datetime.now(timezone.utc),
        coordinate_frame="dobot_base",
        valid=True,
    )


def assert_pose(actual: RobotPosition, expected: RobotPosition, *, tolerance_mm: float) -> None:
    assert actual.x == pytest.approx(expected.x, abs=tolerance_mm)
    assert actual.y == pytest.approx(expected.y, abs=tolerance_mm)
    assert actual.z == pytest.approx(expected.z, abs=tolerance_mm)
    assert actual.r == pytest.approx(expected.r, abs=0.25)


@pytest.mark.live
@requires_webots
def test_live_vision_target_single_x_step_reaches_measured_webots_pose():
    controller, gateway, client = webots_system()
    before = measured_position(client)
    destination = before.model_copy(update={"x": before.x + 5.0})

    result = controller.execute_target(
        before, to_motion_target(vision_target(destination)), "webots_001"
    )
    after = measured_position(client)

    assert result.status is ControllerStatus.SUCCESS
    assert [(step.direction, step.distance) for step in result.plan.steps] == [("+X", 5.0)]
    assert [command.tasks[0].direction for command in gateway.calls] == ["+X"]
    assert_pose(after, destination, tolerance_mm=0.75)


@pytest.mark.live
@requires_webots
def test_live_vision_target_multi_axis_path_reaches_measured_webots_pose():
    controller, gateway, client = webots_system()
    before = measured_position(client)
    destination = RobotPosition(
        x=before.x + 10.0,
        y=before.y + 10.0,
        z=before.z + 5.0,
        r=before.r,
    )

    result = controller.execute_target(
        before, to_motion_target(vision_target(destination)), "webots_001"
    )
    after = measured_position(client)

    assert result.status is ControllerStatus.SUCCESS
    assert [(step.direction, step.distance) for step in result.plan.steps] == [
        ("+X", 5.0), ("+X", 5.0), ("+Y", 5.0), ("+Y", 5.0), ("+Z", 5.0),
    ]
    assert [command.tasks[0].direction for command in gateway.calls] == [
        "+X", "+X", "+Y", "+Y", "+Z",
    ]
    # Existing Webots tests permit 0.75 mm per verified step; its established
    # multi-step assertion therefore uses 2.25 mm for accumulated drift.
    assert_pose(after, destination, tolerance_mm=2.25)


@pytest.mark.live
@requires_webots
def test_live_vision_target_rotation_reaches_measured_webots_wrist_pose():
    controller, gateway, client = webots_system()
    before = measured_position(client)
    if before.r > 134.5:
        pytest.skip("current wrist pose has insufficient +R joint range")
    destination = before.model_copy(update={"r": before.r + 10.0})

    result = controller.execute_target(
        before, canonical_rotation_target(destination), "webots_001"
    )
    after = measured_position(client)

    assert result.status is ControllerStatus.SUCCESS
    assert [command.tasks[0].direction for command in gateway.calls] == ["+R", "+R"]
    assert after.r == pytest.approx(destination.r, abs=0.5)


@pytest.mark.live
@requires_webots
def test_live_vision_target_at_current_pose_sends_no_gateway_command():
    controller, gateway, client = webots_system()
    before = measured_position(client)

    result = controller.execute_target(
        before, to_motion_target(vision_target(before)), "webots_001"
    )
    after = measured_position(client)

    assert result.status is ControllerStatus.NO_MOTION
    assert gateway.calls == []
    assert_pose(after, before, tolerance_mm=0.75)


@pytest.mark.live
@requires_webots
def test_invalid_vision_target_never_reaches_webots_gateway():
    controller, gateway, client = webots_system()
    before = measured_position(client)
    invalid = vision_target(before, valid=False, status=TargetStatus.INVALID)

    with pytest.raises(MotionTargetConversionError):
        to_motion_target(invalid)

    after = measured_position(client)
    assert gateway.calls == []
    assert_pose(after, before, tolerance_mm=0.75)


@pytest.mark.live
@requires_webots
def test_invalid_later_command_blocks_all_webots_execution(monkeypatch):
    controller, gateway, client = webots_system()
    before = measured_position(client)
    destination = before.model_copy(update={"x": before.x + 10.0})
    real_validate = controller_module.validate_command
    validation_count = 0

    def reject_second_command(payload):
        nonlocal validation_count
        validation_count += 1
        if validation_count == 2:
            payload["tasks"][0]["distance"] = -1
        return real_validate(payload)

    monkeypatch.setattr(controller_module, "validate_command", reject_second_command)
    result = controller.execute_target(
        before, to_motion_target(vision_target(destination)), "webots_001"
    )
    after = measured_position(client)

    assert result.status is ControllerStatus.VALIDATION_FAILED
    assert gateway.calls == []
    assert_pose(after, before, tolerance_mm=0.75)
