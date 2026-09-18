"""MotionController integration with the existing Webots adapter.

Offline tests use a protocol stub at the WebotsClient boundary. Tests marked
``live`` require the existing Webots world to be open and playing; enable them
explicitly with ``RUN_WEBOTS_INTEGRATION_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from app.adapters.base import RobotAdapterError
from app.adapters.webots import WebotsClient, WebotsRobotAdapter
from app.commands.models import Action
from app.gateway.adapter_manager import AdapterManager
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.robots.models import Robot
from motion.controller import ControllerStatus, MotionController
from motion.models import RobotPosition, RobotTarget
from motion.planner import MotionPlanner, MotionPlanningPolicy
from robots.interface import RobotInterface


class RecordingWebotsClient:
    """Offline Webots protocol stub; it is not a replacement robot adapter."""

    timeout = 5.0

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.response_timeouts: list[float | None] = []

    def request(self, payload, response_timeout=None):
        self.requests.append(payload)
        self.response_timeouts.append(response_timeout)
        if payload["type"] == "status":
            return {
                "ok": True,
                "state": "READY",
                "end_effector_position_mm": {"x": 200, "y": 50, "z": 30},
                "measured_wrist_r_degrees": 0,
            }
        task = payload["tasks"][0]
        return {
            "ok": True,
            "state": "READY",
            "results": [f"[WEBOTS] {task['action']} accepted"],
        }


class UnavailableWebotsClient(RecordingWebotsClient):
    def request(self, payload, response_timeout=None):
        raise RobotAdapterError("Webots test controller is unavailable.")


def webots_robot() -> Robot:
    return Robot(
        robot_id="webots_001",
        name="Virtual Magician Lite",
        robot_type="robotic_arm",
        manufacturer="DOBOT-inspired",
        model="simplified_visual_model",
        adapter_type="webots",
        capabilities=frozenset({
            Action.MOVE,
            Action.ROTATE,
            Action.HOME,
            Action.STOP,
            Action.GET_STATUS,
        }),
        status="ONLINE",
    )


def integration_system(client):
    robot = webots_robot()
    adapter = WebotsRobotAdapter(robot, client)
    registry = RobotRegistry()
    registry.register(robot)
    adapters = AdapterManager(register_mock=False)
    adapters.register("webots", lambda configured_robot: adapter)
    gateway = UniversalGateway(registry, adapters)
    planner = MotionPlanner(MotionPlanningPolicy(
        max_translation_step_mm=5.0,
        max_rotation_step_degrees=5.0,
        coordinate_frame="robot_base",
    ))
    return MotionController(planner, gateway), gateway, adapter


def target(position: RobotPosition) -> RobotTarget:
    return RobotTarget(
        target_id="webots-target",
        class_name="integration-target",
        position=position,
        confidence=1.0,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        coordinate_frame="robot_base",
        valid=True,
    )


def executed_tasks(client: RecordingWebotsClient) -> list[dict]:
    return [
        request["tasks"][0]
        for request in client.requests
        if request["type"] == "execute"
    ]


def test_same_controller_selects_existing_webots_robot_interface():
    client = RecordingWebotsClient()
    controller, gateway, adapter = integration_system(client)
    assert isinstance(adapter, WebotsRobotAdapter)
    assert isinstance(adapter, RobotInterface)
    assert gateway.adapters.get(webots_robot()) is adapter
    assert controller.gateway is gateway


def test_single_move_flows_through_existing_webots_adapter():
    client = RecordingWebotsClient()
    controller, _, adapter = integration_system(client)
    initial = RobotPosition(x=200, y=50, z=30, r=0)

    result = controller.execute_target(
        initial,
        target(RobotPosition(x=205, y=50, z=30, r=0)),
        "webots_001",
    )

    assert result.status is ControllerStatus.SUCCESS
    assert result.gateway_results[0].adapter_type == "webots"
    assert result.gateway_results[0].simulated is True
    assert executed_tasks(client) == [{
        "action": "MOVE",
        "direction": "+X",
        "distance": 5.0,
        "unit": "mm",
    }]
    assert controller.gateway.adapters.get(webots_robot()) is adapter


def test_multi_step_commands_retain_planner_order_at_webots_boundary():
    client = RecordingWebotsClient()
    controller, _, _ = integration_system(client)
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    result = controller.execute_target(
        initial,
        target(RobotPosition(x=210, y=60, z=35, r=0)),
        "webots_001",
    )

    assert result.status is ControllerStatus.SUCCESS
    tasks = executed_tasks(client)
    assert [(task["direction"], task["distance"]) for task in tasks] == [
        ("+X", 5.0),
        ("+X", 5.0),
        ("+Y", 5.0),
        ("+Y", 5.0),
        ("+Z", 5.0),
    ]
    assert len(result.gateway_results) == 5
    assert all(len(request["tasks"]) == 1 for request in client.requests)


def test_rotation_uses_existing_webots_r_command_shape():
    client = RecordingWebotsClient()
    controller, _, _ = integration_system(client)
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    result = controller.execute_target(
        initial,
        target(RobotPosition(x=200, y=50, z=30, r=10)),
        "webots_001",
    )
    assert result.status is ControllerStatus.SUCCESS
    assert executed_tasks(client) == [
        {"action": "ROTATE", "direction": "+R", "angle": 5.0, "unit": "degrees"},
        {"action": "ROTATE", "direction": "+R", "angle": 5.0, "unit": "degrees"},
    ]


def test_webots_unavailable_is_clear_execution_failure():
    controller, _, _ = integration_system(UnavailableWebotsClient())
    initial = RobotPosition(x=200, y=50, z=30, r=0)
    result = controller.execute_target(
        initial,
        target(RobotPosition(x=205, y=50, z=30, r=0)),
        "webots_001",
    )
    assert result.status is ControllerStatus.EXECUTION_FAILED
    assert not result.successful
    assert result.failed_step_sequence == 0
    assert "Webots test controller is unavailable" in result.error


RUN_WEBOTS = os.getenv("RUN_WEBOTS_INTEGRATION_TESTS") == "1"
requires_webots = pytest.mark.skipif(
    not RUN_WEBOTS,
    reason=(
        "set RUN_WEBOTS_INTEGRATION_TESTS=1 after opening the existing "
        "magician_lite.wbt world and pressing Play"
    ),
)


def live_system():
    # One Cartesian correction may run the controller's full bounded solver
    # loop (roughly 20 seconds of simulation time). The production adapter
    # already honors an injected client's timeout, so give live smoke tests a
    # long enough response window without changing adapter behavior.
    client = WebotsClient(timeout=30.0)
    controller, gateway, adapter = integration_system(client)
    return controller, gateway, adapter, client


def measured_position(status: dict) -> RobotPosition:
    position = status["end_effector_position_mm"]
    return RobotPosition(
        x=float(position["x"]),
        y=float(position["y"]),
        z=float(position["z"]),
        r=float(status["measured_wrist_r_degrees"]),
    )


def verified_report(result_text: str) -> dict:
    return json.loads(result_text.split(" verified ", 1)[1])


@pytest.mark.live
@requires_webots
def test_live_webots_single_five_mm_x_move_uses_actual_pose():
    controller, _, _, client = live_system()
    before = measured_position(client.request({"type": "status"}))
    destination = RobotPosition(
        x=before.x + 5.0,
        y=before.y,
        z=before.z,
        r=before.r,
    )
    result = controller.execute_target(before, target(destination), "webots_001")
    after = measured_position(client.request({"type": "status"}))

    assert result.status is ControllerStatus.SUCCESS
    report = verified_report(result.gateway_results[0].results[0])
    assert report["verified"] is True
    assert after.x == pytest.approx(destination.x, abs=0.75)
    assert after.y == pytest.approx(destination.y, abs=0.75)
    assert after.z == pytest.approx(destination.z, abs=0.75)


@pytest.mark.live
@requires_webots
def test_live_webots_multi_step_move_compares_actual_final_pose():
    controller, _, _, client = live_system()
    before = measured_position(client.request({"type": "status"}))
    destination = RobotPosition(
        x=before.x + 10.0,
        y=before.y + 10.0,
        z=before.z + 5.0,
        r=before.r,
    )
    result = controller.execute_target(before, target(destination), "webots_001")
    after = measured_position(client.request({"type": "status"}))

    assert result.status is ControllerStatus.SUCCESS
    reports = [
        verified_report(plan.results[0]) for plan in result.gateway_results
    ]
    assert len(reports) == 5
    assert all(report["verified"] for report in reports)
    # Each existing Webots move permits 0.75 mm axis/cross-axis error. This
    # bound accounts for accumulated drift across the five measured steps.
    assert after.x == pytest.approx(destination.x, abs=2.25)
    assert after.y == pytest.approx(destination.y, abs=2.25)
    assert after.z == pytest.approx(destination.z, abs=2.25)
    assert after.r == pytest.approx(destination.r, abs=0.25)


@pytest.mark.live
@requires_webots
def test_live_webots_two_step_r_rotation_uses_measured_wrist_state():
    controller, _, _, client = live_system()
    before = measured_position(client.request({"type": "status"}))
    if before.r > 134.5:
        pytest.skip("current wrist pose has insufficient +R joint range")
    destination = before.model_copy(update={"r": before.r + 10.0})
    result = controller.execute_target(before, target(destination), "webots_001")
    after = measured_position(client.request({"type": "status"}))

    assert result.status is ControllerStatus.SUCCESS
    reports = [
        verified_report(plan.results[0]) for plan in result.gateway_results
    ]
    assert len(reports) == 2
    assert all(report["verified"] for report in reports)
    assert after.r == pytest.approx(destination.r, abs=0.5)
