"""Shared Phase 2 test builders."""

from app.commands.models import UniversalCommand
from app.robots.models import Robot


def robot(
    robot_id="robot_001",
    *,
    robot_type="robotic_arm",
    capabilities=("MOVE", "PICK", "PLACE", "GRIP", "RELEASE", "HOME", "STOP", "GET_STATUS"),
    status="ONLINE",
    priority=10,
    adapter_type="mock",
):
    return Robot.model_validate(
        {
            "robot_id": robot_id,
            "name": robot_id,
            "robot_type": robot_type,
            "manufacturer": "generic",
            "model": "demo",
            "adapter_type": adapter_type,
            "capabilities": capabilities,
            "status": status,
            "priority": priority,
        }
    )


def command(*tasks, robot_id=None):
    return UniversalCommand.model_validate(
        {"version": "1.0", "robot_id": robot_id, "tasks": list(tasks)}
    )
