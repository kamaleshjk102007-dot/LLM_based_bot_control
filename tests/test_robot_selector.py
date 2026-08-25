import pytest

from app.gateway.capability_manager import CapabilityManager
from app.gateway.robot_registry import RobotRegistry
from app.gateway.robot_selector import (
    NoRobotError,
    RobotSelector,
    RobotUnavailableError,
    UnsupportedRobotError,
)
from conftest import command, robot


def selector(*robots):
    registry = RobotRegistry()
    for item in robots:
        registry.register(item)
    return RobotSelector(registry, CapabilityManager())


def test_explicit_robot_is_used():
    select = selector(
        robot("arm", priority=1),
        robot("mobile", robot_type="mobile_robot", capabilities=("NAVIGATE",), priority=2),
    )
    cmd = command(
        {"action": "NAVIGATE", "target": {"type": "location", "id": "table"}},
        robot_id="mobile",
    )
    assert select.select(cmd).robot_id == "mobile"


def test_explicit_missing_offline_and_unsupported_are_rejected():
    cmd = command({"action": "PICK", "object": {"type": "cube"}}, robot_id="target")
    with pytest.raises(RobotUnavailableError):
        selector().select(cmd)
    with pytest.raises(RobotUnavailableError):
        selector(robot("target", status="OFFLINE")).select(cmd)
    with pytest.raises(UnsupportedRobotError):
        selector(robot("target", capabilities=("MOVE",))).select(cmd)


def test_automatic_selection_requires_all_capabilities():
    select = selector(
        robot("mobile", robot_type="mobile_robot", capabilities=("MOVE", "NAVIGATE")),
        robot("arm", capabilities=("PICK", "PLACE", "MOVE")),
    )
    cmd = command(
        {"action": "PICK", "object": {"type": "cube"}},
        {"action": "PLACE", "target": {"type": "box"}},
        {"action": "MOVE", "direction": "forward"},
    )
    assert select.select(cmd).robot_id == "arm"


def test_no_compatible_online_robot():
    select = selector(
        robot("mobile", capabilities=("MOVE",), status="ONLINE"),
        robot("arm", capabilities=("PICK",), status="OFFLINE"),
    )
    with pytest.raises(NoRobotError):
        select.select(command({"action": "PICK", "object": {"type": "cube"}}))


def test_priority_then_robot_id_tie_break_is_deterministic():
    cmd = command({"action": "STOP"})
    assert selector(
        robot("z", priority=1), robot("a", priority=2)
    ).select(cmd).robot_id == "z"
    assert selector(
        robot("z", priority=1), robot("a", priority=1)
    ).select(cmd).robot_id == "a"
