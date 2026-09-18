"""
Target selection, ambiguity resolution, temporal stability, and freshness checking.
Enforces safety policies before generating a RobotTarget for Member 4.
"""

from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
import uuid
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from motion.models import RobotPosition, RobotTarget as MotionRobotTarget

from .models import Detection, Point3D, RobotTarget, TargetStatus


class MotionTargetConversionError(ValueError):
    """Raised when a perception target cannot cross into motion planning."""


def to_motion_target(target: RobotTarget) -> MotionRobotTarget:
    """Convert one validated vision target to Member 4's canonical model.

    The conversion is deliberately explicit: perception-only status metadata
    is retained inside ``vision`` and only robot-frame pose data crosses the
    Member 3 -> Member 4 boundary.
    """

    if not isinstance(target, RobotTarget):
        raise MotionTargetConversionError("target must be a vision RobotTarget")
    if not target.valid or target.status is not TargetStatus.VALID:
        raise MotionTargetConversionError("only VALID vision targets may be planned")
    if target.coordinate_frame != "dobot_base":
        raise MotionTargetConversionError(
            "vision target must be expressed in the dobot_base frame"
        )
    try:
        timestamp = datetime.fromisoformat(target.timestamp)
    except (TypeError, ValueError) as exc:
        raise MotionTargetConversionError("vision target timestamp is invalid") from exc

    try:
        return MotionRobotTarget(
            target_id=target.target_id,
            class_name=target.class_name,
            position=RobotPosition(
                x=target.position.x,
                y=target.position.y,
                z=target.position.z,
            ),
            confidence=target.confidence,
            timestamp=timestamp,
            coordinate_frame=target.coordinate_frame,
            valid=target.valid,
        )
    except ValueError as exc:
        raise MotionTargetConversionError(
            "vision target does not satisfy the canonical motion contract"
        ) from exc


class TargetSelector:
    """
    Evaluates raw detections against a task query.
    Enforces the core rule:
      0 matches -> NOT_FOUND
      1 match   -> VALID candidate
      2+ matches -> AMBIGUOUS (Never randomly pick!)
    """

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence

    def select(
        self,
        detections: List[Detection],
        requested_class: str,
    ) -> Tuple[Optional[Detection], TargetStatus, str]:
        """
        Select single matching detection for requested class.

        :param detections: All detections in current frame.
        :param requested_class: Object label requested (e.g. 'red_block').
        :return: (Selected detection or None, TargetStatus, explanatory message)
        """
        # Filter detections matching requested class
        class_matches = [d for d in detections if d.class_name == requested_class]

        if len(class_matches) == 0:
            return None, TargetStatus.NOT_FOUND, f"No object of class '{requested_class}' detected."

        # Check confidence threshold
        confident_matches = [d for d in class_matches if d.confidence >= self.min_confidence]

        if len(confident_matches) == 0:
            max_conf = max(d.confidence for d in class_matches)
            return (
                None,
                TargetStatus.LOW_CONFIDENCE,
                f"Candidate '{requested_class}' found but confidence {max_conf:.2f} < threshold {self.min_confidence:.2f}.",
            )

        if len(confident_matches) > 1:
            return (
                None,
                TargetStatus.AMBIGUOUS,
                f"Multiple ({len(confident_matches)}) '{requested_class}' detected without disambiguation rule.",
            )

        # Exactly 1 confident match
        return confident_matches[0], TargetStatus.VALID, f"Unique '{requested_class}' identified."


class TemporalStabilityTracker:
    """
    Sliding-window temporal filter to prevent robot movement on jittery frames.
    Requires multiple consecutive consistent observations before certifying stability.
    """

    def __init__(
        self,
        window_size: int = 4,
        min_samples: int = 3,
        max_std_dev_mm: float = 3.5,
        max_jump_mm: float = 20.0,
    ):
        """
        :param window_size: Number of recent positions to retain.
        :param min_samples: Minimum frames required before certifying stability.
        :param max_std_dev_mm: Maximum allowable standard deviation in millimeters.
        :param max_jump_mm: Maximum single-frame jump before resetting history.
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.max_std_dev_mm = max_std_dev_mm
        self.max_jump_mm = max_jump_mm
        self._history: Dict[str, Deque[Point3D]] = {}

    def reset(self, class_name: Optional[str] = None) -> None:
        """Reset history for a specific class or all classes."""
        if class_name:
            self._history.pop(class_name, None)
        else:
            self._history.clear()

    def update(
        self,
        class_name: str,
        new_point: Point3D,
    ) -> Tuple[Point3D, TargetStatus, float, str]:
        """
        Update tracker with a new candidate position and evaluate stability.

        :param class_name: Object class
        :param new_point: Latest calculated 3D robot coordinates
        :return: (Filtered Point3D, TargetStatus, stability_score [0..1], message)
        """
        if class_name not in self._history:
            self._history[class_name] = deque(maxlen=self.window_size)

        history = self._history[class_name]

        # Check for sudden physical jumps (e.g. detection switched objects or noise)
        if len(history) > 0:
            last_point = history[-1]
            jump = new_point.planar_distance_to(last_point)
            if jump > self.max_jump_mm:
                # Sudden jump: reset history with new point
                history.clear()
                history.append(new_point)
                return (
                    new_point,
                    TargetStatus.UNSTABLE,
                    0.0,
                    f"Position jumped {jump:.1f}mm (> {self.max_jump_mm}mm); resetting stability window.",
                )

        history.append(new_point)

        # If not enough samples yet
        if len(history) < self.min_samples:
            return (
                new_point,
                TargetStatus.UNSTABLE,
                float(len(history) / self.window_size),
                f"Stabilizing target ({len(history)}/{self.min_samples} frames).",
            )

        # Calculate standard deviation
        xs = [p.x for p in history]
        ys = [p.y for p in history]
        zs = [p.z for p in history]

        std_x = float(np.std(xs))
        std_y = float(np.std(ys))
        total_std = float(np.sqrt(std_x ** 2 + std_y ** 2))

        if total_std > self.max_std_dev_mm:
            stability_score = max(0.0, 1.0 - (total_std / (2.0 * self.max_std_dev_mm)))
            return (
                new_point,
                TargetStatus.UNSTABLE,
                round(stability_score, 2),
                f"Position jitter detected: std_dev {total_std:.2f}mm > max {self.max_std_dev_mm}mm.",
            )

        # Stable! Return mean filtered coordinate to suppress sub-pixel noise
        mean_point = Point3D(
            x=round(float(np.mean(xs)), 2),
            y=round(float(np.mean(ys)), 2),
            z=round(float(np.mean(zs)), 2),
        )
        stability_score = round(min(1.0, max(0.0, 1.0 - (total_std / self.max_std_dev_mm))), 2)

        return (
            mean_point,
            TargetStatus.VALID,
            stability_score,
            f"Target stable across {len(history)} frames (std_dev={total_std:.2f}mm).",
        )


class TargetFreshnessChecker:
    """
    Guarantees targets sent to Member 4 are fresh and not outdated.
    """

    def __init__(self, max_age_seconds: float = 1.5):
        self.max_age_seconds = max_age_seconds

    def check(self, target: RobotTarget) -> Tuple[bool, float, Optional[str]]:
        """
        Check if target timestamp is within allowable age limit.
        :return: (is_fresh: bool, age_seconds: float, reason if stale)
        """
        try:
            target_time = datetime.fromisoformat(target.timestamp)
            now = datetime.now(timezone.utc)
            age = (now - target_time).total_seconds()

            if age > self.max_age_seconds:
                return False, age, f"Target is STALE: age {age:.2f}s > max {self.max_age_seconds}s."
            return True, age, None
        except Exception as e:
            return False, 999.0, f"Failed to parse target timestamp: {e}"


def create_invalid_robot_target(
    class_name: str,
    status: TargetStatus,
    message: str,
    position: Optional[Point3D] = None,
    confidence: float = 0.0,
    source_detection_id: Optional[str] = None,
) -> RobotTarget:
    """Helper to instantiate a standardized INVALID RobotTarget for Member 4."""
    return RobotTarget(
        target_id=f"tgt_{uuid.uuid4().hex[:8]}",
        class_name=class_name,
        position=position or Point3D(x=0.0, y=0.0, z=0.0),
        confidence=confidence,
        timestamp=datetime.now(timezone.utc).isoformat(),
        coordinate_frame="dobot_base",
        valid=False,
        status=status,
        stability_score=0.0,
        source_detection_id=source_detection_id,
        message=message,
    )
