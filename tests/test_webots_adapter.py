from app.adapters.webots import WebotsRobotAdapter
from app.commands.models import UniversalCommand
from app.robots.models import Robot


class FakeClient:
    def __init__(self):
        self.payload = None

    def request(self, payload):
        self.payload = payload
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
    assert adapter.simulated is True


def test_webots_adapter_status():
    assert WebotsRobotAdapter(robot(), FakeClient()).get_status() == "READY"
