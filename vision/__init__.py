"""
Member 3: Vision + Sensors + Perception Package
For Robot 1 — DOBOT Magician Lite
"""

from .calibration import TabletopCalibration, ValidationMetrics
from .camera import BaseCamera, FileCamera, MockCamera, USBCamera, WebotsCamera
from .coordinate_transform import CoordinateTransformer
from .detector import BaseDetector, ColorShapeDetector, YOLODetector
from .models import (
    BoundingBox,
    Detection,
    Frame,
    Point2D,
    Point3D,
    RobotTarget,
    TargetStatus,
)
from .pipeline import VisionPipeline
from .target import (
    TargetFreshnessChecker,
    TargetSelector,
    TemporalStabilityTracker,
    create_invalid_robot_target,
)
from .visualization import VisionVisualizer

__version__ = "1.0.0"

__all__ = [
    # Models & Contracts
    "Point2D",
    "Point3D",
    "BoundingBox",
    "Frame",
    "Detection",
    "TargetStatus",
    "RobotTarget",
    # Camera Abstraction
    "BaseCamera",
    "USBCamera",
    "FileCamera",
    "MockCamera",
    "WebotsCamera",
    # Detection
    "BaseDetector",
    "ColorShapeDetector",
    "YOLODetector",
    # Calibration & Transforms
    "TabletopCalibration",
    "ValidationMetrics",
    "CoordinateTransformer",
    # Target Selection & Stability
    "TargetSelector",
    "TemporalStabilityTracker",
    "TargetFreshnessChecker",
    "create_invalid_robot_target",
    # Visualization & Pipeline
    "VisionVisualizer",
    "VisionPipeline",
]
