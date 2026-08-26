from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.adapters.dobot.client import ConnectionState, DobotLinkClient
from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode
from app.adapters.dobot.exceptions import (
    DobotConfigurationError,
    DobotSafetyError,
)


class FakeLite:
    def __init__(self, pose=None):
        self.calls = []
        self.pose = pose or {"x": 100, "y": 0, "z": 50, "r": 0}

    def SearchDobot(self):
        self.calls.append(("SearchDobot", {}))
        return [{"portName": "COM7"}]

    def __getattr__(self, name):
        def call(**kwargs):
            self.calls.append((name, kwargs))
            if name == "GetPose":
                return dict(self.pose)
            if name == "GetEndEffectorType":
                return 2
            return {"ok": True}
        return call


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.1
        return self.value


def make_client(lite=None, config=None, clock=None):
    lite = lite or FakeLite()
    config = config or DobotConfig(
        mode=OperationMode.REAL,
        max_retries=0,
        verification_start_delay_seconds=0,
    )
    client = DobotLinkClient(
        config,
        backend_factory=lambda _: SimpleNamespace(MagicianLite=lite),
        sleep=lambda _: None,
        clock=clock or FakeClock(),
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
    result = client.move(DobotPosition(100, 0, 50, 0))
    assert result["verified"] is True
    assert result["before"] == result["target"] == result["final"]
    client.set_gripper(True)
    client.stop()
    names = [name for name, _ in lite.calls]
    for required in (
        "SearchDobot", "ConnectDobot", "GetPose", "SetHOMECmd",
        "SetPTPCmd", "QueuedCmdClear", "GetEndEffectorType",
        "SetEndEffectorGripper", "QueuedCmdStop",
    ):
        assert required in names
    assert names.count("GetPose") >= 5
    client.disconnect()
    assert client.state is ConnectionState.DISCONNECTED


def test_move_mismatch_stops_clears_and_enters_error():
    lite = FakeLite({"x": 100, "y": 25, "z": 50, "r": 0})
    config = DobotConfig(
        mode=OperationMode.REAL,
        max_retries=0,
        verification_timeout_seconds=0.2,
        verification_start_delay_seconds=0,
        verification_samples=2,
    )
    client, _ = make_client(lite, config, FakeClock())
    client.connect()

    with pytest.raises(DobotSafetyError, match="verification failed") as caught:
        client.move(DobotPosition(100, 0, 50, 0))

    assert "target={'x': 100" in str(caught.value)
    names = [name for name, _ in lite.calls]
    assert "QueuedCmdStop" in names
    assert "QueuedCmdClear" in names
    assert client.state is ConnectionState.ERROR
    assert client.last_error


def test_connection_retries_are_bounded():
    attempts = 0

    def failing(_):
        nonlocal attempts
        attempts += 1
        raise OSError("DobotLink unavailable")

    config = DobotConfig(mode=OperationMode.REAL, max_retries=2)
    client = DobotLinkClient(
        config,
        backend_factory=failing,
        sleep=lambda _: None,
    )
    with pytest.raises(Exception, match="3 attempt"):
        client.connect()
    assert attempts == 3
    assert client.state is ConnectionState.ERROR
