"""
Data models for Member 3: Vision + Sensors + Perception.
Defines foundational schemas for frames, detections, coordinate representations,
and the RobotTarget contract consumed by Member 4.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Point2D(BaseModel):
    """Represents a 2D coordinate in pixel space (u, v)."""
    x: float = Field(..., description="Horizontal pixel coordinate (u)")
    y: float = Field(..., description="Vertical pixel coordinate (v)")

    model_config = ConfigDict(frozen=True)


class Point3D(BaseModel):
    """Represents a 3D coordinate in physical space (millimeters)."""
    x: float = Field(..., description="X coordinate in millimeters")
    y: float = Field(..., description="Y coordinate in millimeters")
    z: float = Field(..., description="Z coordinate in millimeters")

    model_config = ConfigDict(frozen=True)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def distance_to(self, other: Point3D) -> float:
        """Euclidean distance in 3D space."""
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2))

    def planar_distance_to(self, other: Point3D) -> float:
        """Euclidean distance on the XY plane."""
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates (top-left x1, y1 to bottom-right x2, y2)."""
    x1: int = Field(..., description="Top-left X pixel coordinate")
    y1: int = Field(..., description="Top-left Y pixel coordinate")
    x2: int = Field(..., description="Bottom-right X pixel coordinate")
    y2: int = Field(..., description="Bottom-right Y pixel coordinate")

    model_config = ConfigDict(frozen=True)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)


class Frame(BaseModel):
    """
    Standard camera frame containing metadata and raw image array.
    """
    frame_id: str = Field(..., description="Unique frame identifier")
    timestamp: float = Field(..., description="UNIX timestamp in seconds")
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    image: Any = Field(..., description="NumPy array containing raw image (BGR)")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_image(cls, image: np.ndarray, frame_id: Optional[str] = None, timestamp: Optional[float] = None) -> Frame:
        """Helper to create a Frame directly from an OpenCV image array."""
        import time
        import uuid
        h, w = image.shape[:2]
        fid = frame_id or f"frame_{uuid.uuid4().hex[:8]}"
        ts = timestamp if timestamp is not None else time.time()
        return cls(frame_id=fid, timestamp=ts, width=w, height=h, image=image)


class Detection(BaseModel):
    """
    Represents an object detected in an image frame.
    """
    detection_id: str = Field(..., description="Unique detection identifier")
    class_name: str = Field(..., description="Classification label (e.g. red_block)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    bbox: BoundingBox = Field(..., description="2D pixel bounding box")
    center: Point2D = Field(..., description="Centroid in pixel coordinates")

    model_config = ConfigDict(frozen=True)


class TargetStatus(str, Enum):
    """Evaluation status for target selection and validation."""
    VALID = "VALID"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNSTABLE = "UNSTABLE"
    STALE = "STALE"
    OUT_OF_REACH = "OUT_OF_REACH"
    INVALID = "INVALID"


class RobotTarget(BaseModel):
    """
    Official contract between Member 3 (Vision) and Member 4 (Motion Planning).
    Describes the physical target in robot base coordinates.
    """
    target_id: str = Field(..., description="Unique target identifier")
    class_name: str = Field(..., description="Target object class name (e.g. red_block)")
    position: Point3D = Field(..., description="Physical position (X, Y, Z in mm)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Perception confidence")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of target generation"
    )
    coordinate_frame: str = Field(
        default="dobot_base",
        description="Reference coordinate frame. Strictly 'dobot_base'."
    )
    valid: bool = Field(..., description="True if target is safe and ready for motion planning")
    status: TargetStatus = Field(..., description="Validation status")
    stability_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Temporal stability score [0.0, 1.0]"
    )
    source_detection_id: Optional[str] = Field(
        default=None, description="Detection ID from which target was generated"
    )
    message: Optional[str] = Field(
        default=None, description="Diagnostic message or reason if invalid"
    )

    model_config = ConfigDict(frozen=True)
