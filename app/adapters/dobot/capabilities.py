"""DOBOT Magician Lite registry compatibility helper."""

from app.robots.models import Robot
from robots.registry import RobotConfigurationError, RobotRegistry


def build_dobot_robot() -> Robot:
    """Return the configured DOBOT without duplicating its capabilities."""
    robot = RobotRegistry.from_config().get_robot("dobot_001")
    if robot is None:
        raise RobotConfigurationError(
            "Required robot 'dobot_001' is missing from config/robots.json."
        )
    return robot
