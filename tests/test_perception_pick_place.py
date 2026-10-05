"""Unit tests for perception integration, coordinate resolution, and PICK/PLACE execution."""

from __future__ import annotations

import pytest

from app.adapters.base import RobotAdapterError
from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
from app.adapters.dobot.capabilities import build_dobot_robot
from app.adapters.dobot.client import ConnectionState
from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode, SafetyLimits
from app.adapters.dobot.exceptions import DobotSafetyError, DobotUnsupportedActionError
from app.adapters.dobot.mapper import map_task
from app.commands.models import Action, EntityReference, Task, UniversalCommand
from app.perception.manager import BlockObject, PerceptionManager


class FakeDobotClient:
    state = ConnectionState.READY

    def __init__(self):
        self.calls = []
        self.current_pose = DobotPosition(200.0, 0.0, 50.0, 0.0)

    def get_status(self):
        self.calls.append("status")
        return {"state": "READY", "pose": self.current_pose.as_dict()}

    def home(self):
        self.calls.append("home")
        return "ok"

    def move(self, position: DobotPosition):
        self.calls.append(("move", position.as_dict()))
        self.current_pose = position
        return {"verified": True, "target": position.as_dict()}

    def set_gripper(self, on: bool):
        self.calls.append(("gripper", on))
        return "ok"

    def stop(self):
        self.calls.append("stop")
        return "software-stopped"


@pytest.fixture
def sample_perception():
    return PerceptionManager([
        BlockObject(id="red_cube_1", color="red", type="cube", x=220.0, y=-60.0, z=20.0),
        BlockObject(id="blue_cube_1", color="blue", type="cube", x=240.0, y=60.0, z=20.0),
        BlockObject(id="green_bin_1", color="green", type="bin", x=200.0, y=120.0, z=20.0),
    ])


@pytest.fixture
def real_config():
    return DobotConfig(
        mode=OperationMode.REAL,
        hover_height_mm=40.0,
        safety_limits=SafetyLimits(
            min_x=120.0, max_x=330.0,
            min_y=-250.0, max_y=250.0,
            min_z=-50.0, max_z=160.0,
            min_r=-180.0, max_r=180.0,
        ),
    )


def test_perception_manager_queries(sample_perception):
    red = sample_perception.find(color="red")
    assert red is not None
    assert red.id == "red_cube_1"
    assert red.x == 220.0 and red.y == -60.0

    green_bin = sample_perception.find(color="green", type="bin")
    assert green_bin is not None
    assert green_bin.id == "green_bin_1"

    missing = sample_perception.find(color="yellow")
    assert missing is None


def test_perception_prompt_formatting(sample_perception):
    context = sample_perception.to_prompt_context()
    assert "red cube" in context
    assert "green bin" in context
    assert "X=220.0 mm" in context


def test_mapper_resolves_pick_from_parameters(real_config):
    task = Task(
        action=Action.PICK,
        object=EntityReference(name="red block"),
        parameters={"x": 220.0, "y": -60.0, "z": 20.0},
    )
    mapped = map_task(task, real_config)
    assert mapped["action"] == "PICK"
    assert mapped["x"] == 220.0
    assert mapped["y"] == -60.0
    assert mapped["z"] == 20.0
    assert mapped["hover_z"] == 60.0  # 20 + 40mm hover


def test_mapper_resolves_pick_from_perception(real_config, sample_perception):
    task = Task(
        action=Action.PICK,
        object=EntityReference(color="red", type="cube"),
    )
    mapped = map_task(task, real_config, perception=sample_perception)
    assert mapped["action"] == "PICK"
    assert mapped["x"] == 220.0
    assert mapped["y"] == -60.0
    assert mapped["z"] == 20.0
    assert mapped["hover_z"] == 60.0


def test_mapper_resolves_place_from_perception(real_config, sample_perception):
    task = Task(
        action=Action.PLACE,
        target=EntityReference(color="green", type="bin"),
    )
    mapped = map_task(task, real_config, perception=sample_perception)
    assert mapped["action"] == "PLACE"
    assert mapped["x"] == 200.0
    assert mapped["y"] == 120.0
    assert mapped["z"] == 20.0
    assert mapped["hover_z"] == 60.0


def test_mapper_rejects_out_of_bounds_block(real_config):
    task = Task(
        action=Action.PICK,
        object=EntityReference(name="far object"),
        parameters={"x": 500.0, "y": 0.0, "z": 20.0},  # Max X is 330.0
    )
    with pytest.raises(DobotSafetyError, match="outside the allowed range"):
        map_task(task, real_config)


def test_adapter_executes_pick_and_place_sequence(real_config, sample_perception):
    client = FakeDobotClient()
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, real_config, confirm=lambda *_: True, perception=sample_perception
    )
    command = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [
            {
                "action": "PICK",
                "object": {"color": "red", "type": "cube"},
            },
            {
                "action": "PLACE",
                "target": {"color": "green", "type": "bin"},
            },
        ],
    })

    assert adapter.validate(command) is True
    results = adapter.execute(command)
    assert len(results) == 2
    assert "PICK" in results[0]
    assert "PLACE" in results[1]

    # Verify trajectory sequence:
    # PICK: 1. hover (x=220, y=-60, z=60) -> 2. grip height (z=20) -> 3. gripper ON -> 4. retract hover (z=60)
    # PLACE: 5. hover (x=200, y=120, z=60) -> 6. release height (z=20) -> 7. gripper OFF -> 8. retract hover (z=60)
    expected_calls = [
        ("move", {"x": 220.0, "y": -60.0, "z": 60.0, "r": 0.0}),
        ("move", {"x": 220.0, "y": -60.0, "z": 20.0, "r": 0.0}),
        ("gripper", True),
        ("move", {"x": 220.0, "y": -60.0, "z": 60.0, "r": 0.0}),
        ("move", {"x": 200.0, "y": 120.0, "z": 60.0, "r": 0.0}),
        ("move", {"x": 200.0, "y": 120.0, "z": 20.0, "r": 0.0}),
        ("gripper", False),
        ("move", {"x": 200.0, "y": 120.0, "z": 60.0, "r": 0.0}),
    ]
    assert client.calls == expected_calls


def test_adapter_absolute_move(real_config):
    client = FakeDobotClient()
    adapter = DobotMagicianLiteAdapter(
        build_dobot_robot(), client, real_config, confirm=lambda *_: True
    )
    command = UniversalCommand.model_validate({
        "robot_id": "dobot_001",
        "tasks": [
            {
                "action": "MOVE",
                "position": "coordinate_target",
                "parameters": {"x": 210.0, "y": 50.0, "z": 30.0},
            }
        ],
    })
    assert adapter.validate(command) is True
    results = adapter.execute(command)
    assert len(results) == 1
    assert client.calls == [("move", {"x": 210.0, "y": 50.0, "z": 30.0, "r": 0.0})]
