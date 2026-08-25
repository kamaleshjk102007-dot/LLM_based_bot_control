from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.adapters.dobot.client import ConnectionState, DobotLinkClient
from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode
from app.adapters.dobot.exceptions import DobotConfigurationError


class FakeLite:
    def __init__(self):
        self.calls = []

    def SearchDobot(self):
        self.calls.append(("SearchDobot", {}))
        return [{"portName": "COM7"}]

    def __getattr__(self, name):
        def call(**kwargs):
            self.calls.append((name, kwargs))
            if name == "GetPose":
                return {"x": 100, "y": 0, "z": 50, "r": 0}
            if name == "GetEndEffectorType":
                return 2
            return {"ok": True}
        return call


def make_client(lite=None):
    lite = lite or FakeLite()
    config = DobotConfig(mode=OperationMode.REAL, max_retries=0)
    client = DobotLinkClient(
        config,
        backend_factory=lambda _: SimpleNamespace(MagicianLite=lite),
        sleep=lambda _: None,
    )
    return client, lite


def test_simulation_mode_never_touches_backend():
    touched = False

    def factory(_):
        nonlocal touched
        touched = True

    client = DobotLinkClient(DobotConfig(), backend_factory=factory)
    with pytest.raises(DobotConfigurationError):
        client.connect()
    assert touched is False


def test_lifecycle_and_verified_rpc_mapping():
    client, lite = make_client()
    assert client.state is ConnectionState.DISCONNECTED
    client.connect()
    assert client.state is ConnectionState.READY
    assert client.port_name == "COM7"
    client.home()
    client.move(DobotPosition(100, 0, 50, 0))
    client.set_gripper(True)
    client.stop()
    names = [name for name, _ in lite.calls]
    assert names == [
        "SearchDobot", "ConnectDobot", "GetPose", "SetHOMECmd",
        "SetPTPCmd", "GetEndEffectorType", "SetEndEffectorGripper",
        "QueuedCmdStop",
    ]
    client.disconnect()
    assert client.state is ConnectionState.DISCONNECTED


def test_connection_retries_are_bounded():
    attempts = 0

    def failing(_):
        nonlocal attempts
        attempts += 1
        raise OSError("DobotLink unavailable")

    config = DobotConfig(mode=OperationMode.REAL, max_retries=2)
    client = DobotLinkClient(config, backend_factory=failing, sleep=lambda _: None)
    with pytest.raises(Exception, match="3 attempt"):
        client.connect()
    assert attempts == 3
    assert client.state is ConnectionState.ERROR
