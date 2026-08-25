from app.gateway.adapter_manager import AdapterManager
from app.gateway.capability_manager import CapabilityManager
from app.gateway.command_router import CommandRouter, PlanStatus
from app.gateway.robot_registry import RobotRegistry
from app.gateway.robot_selector import RobotSelector
from app.gateway.safety_validator import SafetyValidator
from conftest import command, robot


def router(*robots):
    registry = RobotRegistry()
    for item in robots:
        registry.register(item)
    capabilities = CapabilityManager()
    return CommandRouter(
        RobotSelector(registry, capabilities),
        capabilities,
        SafetyValidator(capabilities),
        AdapterManager(),
    )


def test_complete_routing_flow_creates_ready_plan():
    route = router(robot("arm", capabilities=("PICK", "PLACE", "MOVE")))
    cmd = command(
        {"action": "PICK", "object": {"type": "cube", "color": "red"}},
        {"action": "PLACE", "target": {"type": "box", "id": "A"}},
        {"action": "MOVE", "direction": "forward", "distance": 20, "unit": "cm"},
    )
    plan = route.route(cmd)
    assert plan.status is PlanStatus.READY
    assert plan.robot_id == "arm"
    assert plan.adapter_type == "mock"
    assert plan.safety_passed
    assert plan.simulated
    assert plan.capability_checks == {"PICK": True, "PLACE": True, "MOVE": True}
    assert len(plan.results) == 3


def test_no_robot_and_no_partial_approval():
    route = router(robot("mobile", capabilities=("MOVE", "NAVIGATE")))
    plan = route.route(
        command(
            {"action": "PICK", "object": {"type": "cube"}},
            {"action": "MOVE", "direction": "forward"},
        )
    )
    assert plan.status is PlanStatus.NO_ROBOT
    assert not plan.results
