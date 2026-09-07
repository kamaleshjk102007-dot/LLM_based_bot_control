from __future__ import annotations

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
        self.before = DobotPosition(100, 0, 50, 0)

    def get_status(self):
        self.calls.append("status")
        return {"state": "READY"}

    def home(self):
        self.calls.append("home")
        return "ok"

    def calibration_preview(self, axis, delta_mm):
        self.calls.append(("preview", axis, delta_mm))
        values = self.before.as_dict()
        values[axis] += delta_mm
        target = DobotPosition(**values)
        return self.before, target

    def rotation_preview(self, delta_degrees):
        self.calls.append(("rotation_preview", delta_degrees))
        values = self.before.as_dict()
        values["r"] += delta_degrees
        return self.before, DobotPosition(**values)

    def calibrate(self, axis, delta_mm, expected_before):
        self.calls.append(("calibrate", axis, delta_mm, expected_before))
        return {"verified": True}

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
        calibration_max_step_mm=5,
        safety_limits=SafetyLimits(50, 150, -50, 50, 20, 100, -90, 90),
    )


def command(action, **fields):
    return UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [{"action": action, **fields}],
    })


def test_five_mm_upward_move_requires_detailed_confirmation(config):
    client = FakeClient()
    prompts = []
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, config,
        confirm=lambda action, detail: prompts.append((action, detail)) or True,
    )
    result = adapter.execute(command(
        "MOVE", direction="upward", distance=5, unit="mm"
    ))
    assert "verified" in result[0]
    assert client.calls[0] == ("preview", "z", 5.0)
    assert client.calls[1][0:3] == ("calibrate", "z", 5.0)
    assert '"hard_max_step_mm": 5.0' in prompts[0][1]
    assert '"target"' in prompts[0][1]


@pytest.mark.parametrize(
    ("direction", "expected_axis", "expected_delta"),
    [
        ("X", "x", 1.0),
        ("-X", "x", -1.0),
        ("Y", "y", 1.0),
        ("-Y", "y", -1.0),
        ("Z", "z", 1.0),
        ("-Z", "z", -1.0),
    ],
)
def test_explicit_cartesian_move_uses_guarded_calibration(
    config, direction, expected_axis, expected_delta
):
    client = FakeClient()
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, config, confirm=lambda *_: True
    )
    result = adapter.execute(command(
        "MOVE", direction=direction, distance=1, unit="mm"
    ))
    assert "verified" in result[0]
    assert client.calls[0] == ("preview", expected_axis, expected_delta)
    assert client.calls[1][0:3] == (
        "calibrate", expected_axis, expected_delta
    )


@pytest.mark.parametrize(
    ("direction", "expected_delta"),
    [("R", 1.0), ("-R", -1.0)],
)
def test_real_r_rotation_requires_preview_confirmation_and_verification(
    config, direction, expected_delta
):
    client = FakeClient()
    prompts = []
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, config,
        confirm=lambda action, detail: prompts.append((action, detail)) or True,
    )
    result = adapter.execute(command(
        "ROTATE", direction=direction, angle=1, unit="degrees"
    ))
    assert "verified" in result[0]
    assert client.calls[0] == ("rotation_preview", expected_delta)
    assert client.calls[1][0:3] == ("calibrate", "r", expected_delta)
    assert '"hard_max_step_degrees": 5.0' in prompts[0][1]
    assert '"target"' in prompts[0][1]


def test_real_r_rotation_above_five_degrees_is_rejected(config):
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), FakeClient(), config, confirm=lambda *_: True
    )
    assert adapter.validate(command(
        "ROTATE", direction="R", angle=5.01, unit="degrees"
    )) is False


def test_real_r_rotation_cannot_be_combined_with_status(config):
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), FakeClient(), config, confirm=lambda *_: True
    )
    mixed = UniversalCommand.model_validate({"tasks": [
        {"action": "ROTATE", "direction": "R", "angle": 1, "unit": "degrees"},
        {"action": "GET_STATUS"},
    ]})
    assert adapter.validate(mixed) is False


def test_cancelled_move_never_calibrates(config):
    client = FakeClient()
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, config, confirm=lambda *_: False
    )
    with pytest.raises(RobotAdapterError, match="cancelled"):
        adapter.execute(command(
            "MOVE", direction="downward", distance=1, unit="mm"
        ))
    assert client.calls == [("preview", "z", -1.0)]


@pytest.mark.parametrize("fields", [
    {"direction": "upward", "distance": 5.01, "unit": "mm"},
    {"direction": "left", "distance": 1, "unit": "mm"},
    {"direction": "upward", "distance": 1, "unit": "centimeters"},
])
def test_real_llm_move_rejects_unsafe_shape(config, fields):
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), FakeClient(), config, confirm=lambda *_: True
    )
    assert adapter.validate(command("MOVE", **fields)) is False


def test_move_cannot_be_combined_with_another_task(config):
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), FakeClient(), config, confirm=lambda *_: True
    )
    mixed = UniversalCommand.model_validate({"tasks": [
        {"action": "MOVE", "direction": "upward", "distance": 1, "unit": "mm"},
        {"action": "GET_STATUS"},
    ]})
    assert adapter.validate(mixed) is False


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
