from types import SimpleNamespace

import pytest

from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
from app.adapters.dobot.config import DobotConfig, OperationMode
from app.adapters.webots import WebotsRobotAdapter
from app.commands.models import UniversalCommand
from app.gateway.adapter_manager import AdapterManager, AdapterManagerError
from app.robots.models import Robot
from robots.interface import RobotInterface
from robots.registry import RobotRegistry


class IncompleteRobot(RobotInterface):
    pass


class FakeWebotsClient:
    timeout = 5.0

    def request(self, payload, response_timeout=None):
        if payload["type"] == "status":
            return {"ok": True, "state": "READY"}
        return {"ok": True, "results": ["[WEBOTS] STOP accepted"]}


class NotAnInterface:
    pass


def webots_robot():
    return Robot.model_validate({
        "robot_id": "webots_001",
        "name": "Virtual Magician Lite",
        "robot_type": "robotic_arm",
        "adapter_type": "webots",
        "capabilities": ["MOVE", "ROTATE", "HOME", "STOP", "GET_STATUS"],
        "status": "ONLINE",
    })


def test_robot_interface_is_abstract():
    with pytest.raises(TypeError):
        RobotInterface()


def test_missing_interface_methods_prevent_instantiation():
    with pytest.raises(TypeError):
        IncompleteRobot()


def test_webots_adapter_satisfies_interface_and_preserves_command_flow():
    adapter = WebotsRobotAdapter(webots_robot(), FakeWebotsClient())
    command = UniversalCommand.model_validate({"tasks": [{"action": "STOP"}]})

    assert isinstance(adapter, RobotInterface)
    assert adapter.validate(command) is True
    assert adapter.execute(command) == ["[WEBOTS] STOP accepted"]
    assert adapter.get_status() == "READY"


def test_dobot_adapter_satisfies_interface_without_touching_hardware():
    registry = RobotRegistry.from_config()
    robot = registry.get("dobot_001")
    fake_client = SimpleNamespace()
    config = DobotConfig(mode=OperationMode.REAL)

    adapter = DobotMagicianLiteAdapter(
        robot, fake_client, config, confirm=lambda *_: False
    )

    assert isinstance(adapter, RobotInterface)
    assert registry.has_capability("dobot_001", "MOVE")
    assert registry.has_capability("dobot_001", "ROTATE")


def test_registry_robot_resolves_to_interface_through_adapter_manager():
    registry = RobotRegistry.from_config()
    robot = registry.get("dobot_001")
    manager = AdapterManager(register_mock=False)
    config = DobotConfig(mode=OperationMode.REAL)
    manager.register(
        robot.adapter_type,
        lambda configured_robot: DobotMagicianLiteAdapter(
            configured_robot,
            SimpleNamespace(),
            config,
            confirm=lambda *_: False,
        ),
    )

    assert isinstance(manager.get(robot), RobotInterface)


def test_adapter_manager_rejects_non_interface_factory():
    manager = AdapterManager(register_mock=False)
    manager.register("invalid", lambda _: NotAnInterface())
    invalid_robot = webots_robot().model_copy(
        update={"adapter_type": "invalid"}
    )

    with pytest.raises(AdapterManagerError, match="RobotInterface"):
        manager.get(invalid_robot)
