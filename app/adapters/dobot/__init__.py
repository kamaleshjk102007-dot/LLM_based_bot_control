"""DOBOT Magician Lite adapter package.

Importing this package does not import the optional DOBOT SDK or touch hardware.
"""

from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
from app.adapters.dobot.capabilities import DOBOT_CAPABILITIES, build_dobot_robot
from app.adapters.dobot.client import ConnectionState, DobotLinkClient
from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode

__all__ = (
    "DOBOT_CAPABILITIES",
    "ConnectionState",
    "DobotConfig",
    "DobotLinkClient",
    "DobotMagicianLiteAdapter",
    "DobotPosition",
    "OperationMode",
    "build_dobot_robot",
)
