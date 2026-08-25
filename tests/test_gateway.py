from app.gateway.command_router import PlanStatus
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from conftest import command, robot


def gateway(*robots):
    registry = RobotRegistry()
    for item in robots:
        registry.register(item)
    return UniversalGateway(registry)


def test_pick_selects_arm_and_returns_ready_plan():
    plan = gateway(robot("robot_001")).process(
        command({"action": "PICK", "object": {"type": "cube", "color": "red"}})
    )
    assert plan.status is PlanStatus.READY
    assert plan.robot_id == "robot_001"


def test_mobile_only_cannot_pick():
    plan = gateway(
        robot(
            "robot_002",
            robot_type="mobile_robot",
            capabilities=("MOVE", "ROTATE", "NAVIGATE", "STOP", "GET_STATUS"),
        )
    ).process(command({"action": "PICK", "object": {"type": "cube"}}))
    assert plan.status is PlanStatus.NO_ROBOT
    assert plan.results == []


def test_navigation_selects_mobile_robot():
    plan = gateway(
        robot("arm", capabilities=("PICK", "PLACE")),
        robot(
            "robot_002",
            robot_type="mobile_robot",
            capabilities=("MOVE", "ROTATE", "NAVIGATE", "STOP", "GET_STATUS"),
        ),
    ).process(
        command(
            {"action": "NAVIGATE", "target": {"type": "location", "id": "table"}}
        )
    )
    assert plan.status is PlanStatus.READY
    assert plan.robot_id == "robot_002"


def test_explicit_robot_and_offline_rules():
    target = robot(
        "robot_002",
        robot_type="mobile_robot",
        capabilities=("NAVIGATE",),
    )
    cmd = command(
        {"action": "NAVIGATE", "target": {"type": "location", "id": "table"}},
        robot_id="robot_002",
    )
    assert gateway(robot("other", capabilities=("NAVIGATE",)), target).process(
        cmd
    ).robot_id == "robot_002"

    rejected = gateway(target.model_copy(update={"status": "OFFLINE"})).process(cmd)
    assert rejected.status is PlanStatus.REJECTED


def test_invalid_payload_returns_invalid_plan():
    plan = gateway(robot()).process(
        {"version": "1.0", "tasks": [{"action": "PICK"}]}
    )
    assert plan.status is PlanStatus.INVALID
