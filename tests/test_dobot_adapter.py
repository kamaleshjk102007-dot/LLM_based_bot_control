from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.adapters.base import RobotAdapterError
from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
from app.adapters.dobot.capabilities import build_dobot_robot
from app.adapters.dobot.client import ConnectionState
from app.adapters.dobot.config import (
    DobotConfig, DobotPosition, OperationMode, SafetyLimits,
)
from app.commands.models import UniversalCommand


class FakeClient:
    state = ConnectionState.READY

    def __init__(self):
        self.calls = []

    def get_status(self):
        self.calls.append("status")
        return {"state": "READY"}

    def home(self):
        self.calls.append("home")
        return "ok"

    def move(self, position):
        self.calls.append(("move", position))
        return "ok"

    def stop(self):
        self.calls.append("stop")
        return "software-stopped"

    def set_gripper(self, on):
        self.calls.append(("gripper", on))
        return "ok"


@pytest.fixture
def config():
    return DobotConfig(
        mode=OperationMode.REAL,
        test_position=DobotPosition(100, 0, 50, 0),
        safety_limits=SafetyLimits(50, 150, -50, 50, 20, 100, -90, 90),
    )


def command(action, **fields):
    return UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": action, **fields}],
    })


def test_move_requires_confirmation(config):
    client = FakeClient()
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, config, confirm=lambda *_: False
    )
    with pytest.raises(RobotAdapterError, match="cancelled"):
        adapter.execute(command("MOVE", position="configured"))
    assert client.calls == []


def test_stop_is_software_stop_and_needs_no_confirmation(config):
    client = FakeClient()
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, config,
        confirm=lambda *_: pytest.fail("STOP must not prompt"),
    )
    result = adapter.execute(command("STOP"))
    assert "software-stopped" in result[0]


def test_pick_and_place_are_unsupported(config):
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), FakeClient(), config, confirm=lambda *_: True
    )
    assert adapter.validate(command("PICK", object={"name": "cube"})) is False


def test_multi_step_is_prevalidated_before_hardware(config):
    client = FakeClient()
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, config, confirm=lambda *_: True
    )
    mixed = UniversalCommand.model_validate({
        "tasks": [
            {"action": "HOME"},
            {"action": "PLACE", "target": {"name": "box"}},
        ]
    })
    with pytest.raises(RobotAdapterError, match="not supported"):
        adapter.execute(mixed)
    assert client.calls == []
