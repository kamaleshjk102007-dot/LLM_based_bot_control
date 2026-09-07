from app.adapters.webots import WebotsRobotAdapter
from app.commands.models import UniversalCommand
from app.robots.models import Robot


class FakeClient:
    def __init__(self):
        self.payload = None
        self.timeout = 5.0
        self.response_timeout = None

    def request(self, payload, response_timeout=None):
        self.payload = payload
        self.response_timeout = response_timeout
        if payload["type"] == "status":
            return {"ok": True, "state": "READY"}
        return {"ok": True, "results": ["[WEBOTS] MOVE accepted"]}


def robot():
    return Robot.model_validate({
        "robot_id": "webots_001",
        "name": "Virtual Magician Lite",
        "robot_type": "robotic_arm",
        "manufacturer": "DOBOT-inspired",
        "model": "simplified_visual_model",
        "adapter_type": "webots",
        "capabilities": ["MOVE", "ROTATE", "HOME", "STOP", "GET_STATUS"],
        "status": "ONLINE",
    })


def test_webots_adapter_sends_universal_tasks():
    client = FakeClient()
    adapter = WebotsRobotAdapter(robot(), client)
    command = UniversalCommand.model_validate({
        "tasks": [{"action": "MOVE", "direction": "forward", "distance": 20, "unit": "centimeters"}]
    })
    assert adapter.execute(command) == ["[WEBOTS] MOVE accepted"]
    assert client.payload == {
        "type": "execute",
        "tasks": [{"action": "MOVE", "direction": "forward", "distance": 20.0, "unit": "centimeters"}],
    }
    assert client.response_timeout == 10.0
    assert adapter.simulated is True


def test_webots_adapter_allows_time_for_each_task():
    client = FakeClient()
    client.request = lambda payload, response_timeout=None: {
        "ok": True,
        "results": ["one", "two", "three"],
        "response_timeout": response_timeout,
    }
    adapter = WebotsRobotAdapter(robot(), client)
    command = UniversalCommand.model_validate({
        "tasks": [
            {"action": "MOVE", "direction": axis, "distance": 5, "unit": "mm"}
            for axis in ("X", "Y", "Z")
        ]
    })
    assert adapter.execute(command) == ["one", "two", "three"]


def test_webots_adapter_status():
    assert WebotsRobotAdapter(robot(), FakeClient()).get_status() == "READY"
