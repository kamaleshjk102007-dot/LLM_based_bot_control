"""Robot registry package.

Hardware-independent robot discovery and capability metadata live here.
"""

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
    "RobotNotFoundError",
    "RobotRegistry",
    "RobotRegistryError",
]
