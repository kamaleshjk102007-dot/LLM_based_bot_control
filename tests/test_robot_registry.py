import pytest

from app.gateway.robot_registry import (
    DuplicateRobotError,
    RobotNotFoundError,
    RobotRegistry,
)
from app.robots.models import RobotStatus
from conftest import robot


def test_registration_retrieval_and_unregister():
    registry = RobotRegistry()
    item = registry.register(robot())
    assert registry.get(item.robot_id) == item
    assert registry.unregister(item.robot_id) == item
    with pytest.raises(RobotNotFoundError):
        registry.get(item.robot_id)


def test_duplicate_id_is_rejected():
    registry = RobotRegistry()
    registry.register(robot())
    with pytest.raises(DuplicateRobotError):
        registry.register(robot())


def test_status_online_and_capability_filters():
    registry = RobotRegistry()
    registry.register(robot("arm"))
    registry.register(
        robot(
            "mobile",
            robot_type="mobile_robot",
            capabilities=("MOVE", "NAVIGATE", "STOP"),
            status="OFFLINE",
        )
    )
    assert [item.robot_id for item in registry.list_online()] == ["arm"]
    assert [item.robot_id for item in registry.find_by_capability("NAVIGATE")] == [
        "mobile"
    ]
    updated = registry.update_status("mobile", RobotStatus.ONLINE)
    assert updated.status is RobotStatus.ONLINE
    assert [item.robot_id for item in registry.list_online()] == ["arm", "mobile"]
