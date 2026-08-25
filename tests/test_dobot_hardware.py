"""Opt-in physical integration checks; never collected as ordinary CI hardware work."""

import os

import pytest

pytestmark = pytest.mark.hardware


def test_real_connection_and_status(require_dobot_hardware):
    client = require_dobot_hardware
    client.connect()
    try:
        status = client.get_status()
        assert status["state"] == "READY"
        assert status["pose"] is not None
    finally:
        client.disconnect()


@pytest.fixture
def require_dobot_hardware():
    if os.getenv("RUN_DOBOT_HARDWARE_TESTS") != "1":
        pytest.skip("Set RUN_DOBOT_HARDWARE_TESTS=1 to enable physical tests.")
    if os.getenv("DOBOT_HARDWARE_CONFIRMED") != "YES":
        pytest.skip(
            "Physical testing requires DOBOT_HARDWARE_CONFIRMED=YES after "
            "clearing the workspace and supervising the robot."
        )
    from app.adapters.dobot.client import DobotLinkClient
    from app.adapters.dobot.config import DobotConfig
    return DobotLinkClient(DobotConfig.from_env("real"))
