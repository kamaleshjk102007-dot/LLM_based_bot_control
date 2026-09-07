import json

import pytest

from robots.registry import RobotConfigurationError, RobotRegistry


REQUIRED_CAPABILITIES = ("MOVE", "ROTATE", "GRIP", "RELEASE", "HOME", "STOP")


def test_configured_dobot_can_be_retrieved():
    registry = RobotRegistry.from_config()

    robot = registry.get_robot("dobot_001")

    assert robot is not None
    assert robot.robot_id == "dobot_001"
    assert robot.robot_type == "DOBOT_MAGician_Lite"


def test_unknown_robot_is_handled_cleanly():
    registry = RobotRegistry.from_config()

    assert registry.get_robot("unknown_robot") is None
    assert registry.has_robot("unknown_robot") is False
    assert registry.has_capability("unknown_robot", "MOVE") is False


def test_registered_robot_and_required_capabilities():
    registry = RobotRegistry.from_config()

    assert registry.has_robot("dobot_001") is True
    for capability in REQUIRED_CAPABILITIES:
        assert registry.has_capability("dobot_001", capability) is True


def test_capability_checks_are_case_safe_and_reject_unsupported_actions():
    registry = RobotRegistry.from_config()

    assert registry.has_capability("dobot_001", "move") is True
    assert registry.has_capability("dobot_001", " grip ") is True
    assert registry.has_capability("dobot_001", "FLY") is False


def test_missing_configuration_fails_cleanly(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(RobotConfigurationError, match="Could not read"):
        RobotRegistry.from_config(missing)


def test_invalid_json_fails_cleanly(tmp_path):
    invalid = tmp_path / "robots.json"
    invalid.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RobotConfigurationError, match="Invalid JSON"):
        RobotRegistry.from_config(invalid)


def test_invalid_configuration_shape_fails_cleanly(tmp_path):
    invalid = tmp_path / "robots.json"
    invalid.write_text(json.dumps({"robots": {}}), encoding="utf-8")

    with pytest.raises(RobotConfigurationError, match="'robots' list"):
        RobotRegistry.from_config(invalid)
