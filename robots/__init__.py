"""Universal robot contracts and configuration registry."""

from robots.interface import RobotInterface
from robots.registry import (
    DuplicateRobotError,
    RobotConfigurationError,
    RobotNotFoundError,
    RobotRegistry,
    RobotRegistryError,
)

__all__ = [
    "DuplicateRobotError",
    "RobotConfigurationError",
    "RobotInterface",
    "RobotNotFoundError",
    "RobotRegistry",
    "RobotRegistryError",
]
