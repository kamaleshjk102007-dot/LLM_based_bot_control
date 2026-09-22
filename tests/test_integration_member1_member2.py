"""Integration test suite: Member 1 (DOBOT Magician Lite Adapter) + Member 2 (Universal ExecutionEngine)."""

from __future__ import annotations

import json
import pytest

from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
from app.adapters.dobot.capabilities import build_dobot_robot
from app.adapters.dobot.client import ConnectionState
from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode, SafetyLimits
from app.commands.models import UniversalCommand
from execution.executor import ExecutionEngine
from execution.logger import ExecutionLogger
from execution.result import ExecutionStatus
from robots.interface import RobotInterface


class MockDobotClient:
    """Mock client simulating DobotLink Magician Lite hardware with internal pose tracking."""
    state = ConnectionState.READY

    def __init__(self, initial_position: DobotPosition | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.current_pose = initial_position or DobotPosition(100.0, 0.0, 50.0, 0.0)

    def is_connected(self) -> bool:
        return True

    def get_status(self) -> dict:
        self.calls.append(("status",))
        return {
            "state": "READY",
            "connected": True,
            "port_name": "COM3",
            "pose": self.current_pose.as_dict(),
        }

    def calibration_preview(self, axis: str, delta_mm: float) -> tuple[DobotPosition, DobotPosition]:
        self.calls.append(("preview", axis, delta_mm))
        before = self.current_pose
        values = before.as_dict()
        values[axis] += delta_mm
        target = DobotPosition(**values)
        return before, target

    def rotation_preview(self, delta_degrees: float) -> tuple[DobotPosition, DobotPosition]:
        self.calls.append(("rotation_preview", delta_degrees))
        before = self.current_pose
        values = before.as_dict()
        values["r"] += delta_degrees
        return before, DobotPosition(**values)

    def calibrate(self, axis: str, delta_mm: float, expected_before: DobotPosition) -> dict:
        self.calls.append(("calibrate", axis, delta_mm))
        values = self.current_pose.as_dict()
        values[axis] += delta_mm
        self.current_pose = DobotPosition(**values)
        return {"verified": True, "new_pose": self.current_pose.as_dict()}

    def move(self, position: DobotPosition) -> dict:
        self.calls.append(("move", position.as_dict()))
        self.current_pose = position
        return {"verified": True}

    def home(self) -> str:
        self.calls.append(("home",))
        self.current_pose = DobotPosition(100.0, 0.0, 50.0, 0.0)
        return "ok"

    def stop(self) -> str:
        self.calls.append(("stop",))
        return "software-stopped"

    def set_gripper(self, on: bool) -> str:
        self.calls.append(("gripper", on))
        return "ok"


@pytest.fixture
def integrated_environment(tmp_path):
    """Integrates Member 1 adapter with Member 2 ExecutionEngine."""
    config = DobotConfig(
        mode=OperationMode.REAL,
        calibration_max_step_mm=5.0,
        rotation_max_step_degrees=30.0,
        safety_limits=SafetyLimits(50, 150, -50, 50, 20, 100, -90, 90),
    )
    client = MockDobotClient(DobotPosition(100.0, 0.0, 50.0, 0.0))
    robot_meta = build_dobot_robot()

    # Member 1: Adapter satisfying RobotInterface
    adapter = DobotMagicianLiteAdapter(
        robot=robot_meta,
        client=client,
        config=config,
        confirm=lambda action, detail: True,  # Auto-confirm for automated tests
    )
    assert isinstance(adapter, RobotInterface)

    # Member 2: Execution Logger
    log_file = tmp_path / "execution_log.json"
    logger = ExecutionLogger(str(log_file))

    # Member 2: Universal Execution Engine
    engine = ExecutionEngine(robot=adapter, tolerance=0.2, logger=logger)

    return engine, adapter, client, logger


def test_member1_member2_verified_z_move(integrated_environment):
    """Member 2 executes and verifies a 5mm Z move executed by Member 1 DOBOT."""
    engine, _, client, logger = integrated_environment

    cmd = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": "MOVE", "direction": "Z", "distance": 5.0, "unit": "mm"}],
    })

    results = engine.execute(cmd)

    assert len(results) == 1
    res = results[0]

    # Verification checks performed by Member 2
    assert res.status == ExecutionStatus.SUCCESS
    assert res.verification is True
    assert res.requested_axis == "Z"
    assert res.requested_distance == 5.0
    assert res.actual_distance == 5.0
    assert res.initial_z == 50.0
    assert res.final_z == 55.0
    assert res.error == pytest.approx(0.0)

    # Hardware state verification in Member 1 client
    assert client.current_pose.z == 55.0

    # Member 2 persistent logging verification
    logs = logger.read_logs()
    assert len(logs) == 1
    assert logs[0]["status"] == "SUCCESS"
    assert logs[0]["requested_distance"] == 5.0


def test_member1_member2_negative_x_move(integrated_environment):
    """Member 2 executes and verifies a -2mm X move on Member 1 DOBOT."""
    engine, _, client, logger = integrated_environment

    cmd = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": "MOVE", "direction": "-X", "distance": 2.0, "unit": "mm"}],
    })

    results = engine.execute(cmd)

    assert len(results) == 1
    res = results[0]

    assert res.status == ExecutionStatus.SUCCESS
    assert res.verification is True
    assert res.requested_axis == "X"
    assert res.initial_x == 100.0
    assert res.final_x == 98.0
    assert res.actual_distance == 2.0
    assert client.current_pose.x == 98.0


def test_member1_member2_step_limit_safety_rejection(integrated_environment):
    """Member 1 safety limits reject a 20mm move; Member 2 logs REJECTED without moving hardware."""
    engine, _, client, logger = integrated_environment

    cmd = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": "MOVE", "direction": "Z", "distance": 20.0, "unit": "mm"}],
    })

    results = engine.execute(cmd)

    assert len(results) == 1
    res = results[0]
    assert res.status == ExecutionStatus.REJECTED
    assert "rejected" in res.message.lower() or "failed" in res.failure_reason.lower()

    # Hardware must not have moved
    assert client.current_pose.z == 50.0
    assert not any(c[0] == "calibrate" for c in client.calls)

    # REJECTED result logged
    logs = logger.read_logs()
    assert len(logs) == 1
    assert logs[0]["status"] == "REJECTED"


def test_member1_member2_home_command(integrated_environment):
    """Member 2 sends HOME to Member 1 and verifies SUCCESS."""
    engine, _, client, logger = integrated_environment

    # Offset initial position
    client.current_pose = DobotPosition(120.0, 10.0, 60.0, 0.0)

    cmd = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": "HOME"}],
    })

    results = engine.execute(cmd)

    assert len(results) == 1
    assert results[0].status == ExecutionStatus.SUCCESS
    assert client.current_pose == DobotPosition(100.0, 0.0, 50.0, 0.0)


def test_member1_member2_stop_command(integrated_environment):
    """Member 2 sends STOP to Member 1 and returns STOPPED status."""
    engine, _, client, logger = integrated_environment

    cmd = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": "STOP"}],
    })

    results = engine.execute(cmd)

    assert len(results) == 1
    assert results[0].status == ExecutionStatus.STOPPED
    assert any(c[0] == "stop" for c in client.calls)


def test_member1_member2_gripper_cycle(integrated_environment):
    """Member 2 executes GRIP then RELEASE on Member 1."""
    engine, _, client, _ = integrated_environment

    cmd = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": "GRIP"}, {"action": "RELEASE"}],
    })

    results = engine.execute(cmd)

    assert len(results) == 2
    assert results[0].status == ExecutionStatus.SUCCESS
    assert results[1].status == ExecutionStatus.SUCCESS
    assert ("gripper", True) in client.calls
    assert ("gripper", False) in client.calls
