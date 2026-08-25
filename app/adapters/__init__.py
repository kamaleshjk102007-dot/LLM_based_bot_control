"""Generic adapter boundary and simulation adapter."""

from app.adapters.base import RobotAdapter
from app.adapters.mock import MockRobotAdapter

__all__ = ["RobotAdapter", "MockRobotAdapter"]
