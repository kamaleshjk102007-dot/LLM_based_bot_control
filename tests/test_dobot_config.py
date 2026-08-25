from __future__ import annotations

import pytest

from app.adapters.dobot.config import DobotConfig, OperationMode
from app.adapters.dobot.exceptions import DobotConfigurationError, DobotSafetyError


MOVE_ENV = {
    "DOBOT_TEST_X": "100", "DOBOT_TEST_Y": "0",
    "DOBOT_TEST_Z": "50", "DOBOT_TEST_R": "0",
    "DOBOT_MIN_X": "50", "DOBOT_MAX_X": "150",
    "DOBOT_MIN_Y": "-50", "DOBOT_MAX_Y": "50",
    "DOBOT_MIN_Z": "20", "DOBOT_MAX_Z": "100",
    "DOBOT_MIN_R": "-90", "DOBOT_MAX_R": "90",
}


def test_simulation_is_default_and_does_not_require_hardware():
    assert DobotConfig.from_env().mode is OperationMode.SIMULATION


def test_move_requires_explicit_position_and_limits(monkeypatch):
    for name in MOVE_ENV:
        monkeypatch.delenv(name, raising=False)
    config = DobotConfig.from_env("real")
    with pytest.raises(DobotConfigurationError, match="MOVE is disabled"):
        config.require_safe_test_position()


def test_safe_test_position(monkeypatch):
    for name, value in MOVE_ENV.items():
        monkeypatch.setenv(name, value)
    position = DobotConfig.from_env("real").require_safe_test_position()
    assert position.as_dict() == {"x": 100.0, "y": 0.0, "z": 50.0, "r": 0.0}


def test_out_of_bounds_position_is_rejected(monkeypatch):
    for name, value in MOVE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DOBOT_TEST_X", "151")
    with pytest.raises(DobotSafetyError, match="outside"):
        DobotConfig.from_env("real").require_safe_test_position()
