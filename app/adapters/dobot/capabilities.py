"""Phase 3 capability declaration for DOBOT Magician Lite."""

from app.commands.models import Action
from app.robots.models import Robot

DOBOT_CAPABILITIES = frozenset({
    Action.GET_STATUS,
    Action.HOME,
    Action.MOVE,
    Action.STOP,
    Action.GRIP,
    Action.RELEASE,
})


def build_dobot_robot() -> Robot:
    return Robot(
        robot_id="dobot_001",
        name="DOBOT Magician Lite",
        robot_type="robotic_arm",
        manufacturer="DOBOT",
        model="Magician Lite",
        adapter_type="dobot_magician_lite",
        capabilities=DOBOT_CAPABILITIES,
        status="UNKNOWN",
        priority=10,
    )
