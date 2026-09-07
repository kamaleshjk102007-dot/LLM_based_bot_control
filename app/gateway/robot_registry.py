"""Compatibility imports for the central Robot Registry.

Gateway callers keep their existing import path while the implementation and
configuration-loading responsibility live in robots.registry.
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
