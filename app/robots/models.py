"""Robot-independent registry models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.commands.models import Action


class RobotType(str, Enum):
    ROBOTIC_ARM = "robotic_arm"
    MOBILE_ROBOT = "mobile_robot"
    DRONE = "drone"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class RobotStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    BUSY = "BUSY"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class Robot(BaseModel):
    """Generic robot metadata; it contains no communication details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    robot_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    robot_type: RobotType | str
    manufacturer: str = Field(default="generic", min_length=1)
    model: str = Field(default="unknown", min_length=1)
    adapter_type: str = Field(min_length=1)
    capabilities: frozenset[Action] = Field(default_factory=frozenset)
    status: RobotStatus = RobotStatus.UNKNOWN
    priority: int = Field(default=100, ge=0)

    @field_validator("robot_id", "name", "manufacturer", "model", "adapter_type")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value

    @field_validator("adapter_type")
    @classmethod
    def normalize_adapter_type(cls, value: str) -> str:
        return value.lower()
