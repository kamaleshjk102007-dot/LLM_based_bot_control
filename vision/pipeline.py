"""
Vision pipeline orchestrator for Member 3.
Integrates camera acquisition, detection, target selection, homography calibration,
coordinate transformation, temporal stability, and Member 4 contract dispatch.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import numpy as np

from .calibration import TabletopCalibration
from .camera import BaseCamera, MockCamera
from .coordinate_transform import CoordinateTransformer
from .detector import BaseDetector, ColorShapeDetector
from .models import Detection, Frame, RobotTarget, TargetStatus
from .target import (
    TargetFreshnessChecker,
    TargetSelector,
    TemporalStabilityTracker,
    create_invalid_robot_target,
)
from .visualization import VisionVisualizer


class VisionPipeline:
    """
    Main entry point for Member 3.
    Processes camera frames and delivers a verified RobotTarget to Member 4.
    """

    def __init__(
        self,
        camera: Optional[BaseCamera] = None,
        detector: Optional[BaseDetector] = None,
        calibration: Optional[TabletopCalibration] = None,
        transformer: Optional[CoordinateTransformer] = None,
        min_confidence: float = 0.60,
        stability_samples: int = 3,
        stability_window: int = 4,
        max_std_dev_mm: float = 3.5,
    ):
        # 1. Camera
        self.camera: BaseCamera = camera or MockCamera()

        # 2. Detector
        self.detector: BaseDetector = detector or ColorShapeDetector()

        # 3. Calibration & Coordinate Transformation
        self.calibration: TabletopCalibration = calibration or TabletopCalibration.create_default()
        self.transformer: CoordinateTransformer = transformer or CoordinateTransformer(
            calibration=self.calibration,
            table_z_mm=-50.0,
        )

        # 4. Target selection & verification modules
        self.selector = TargetSelector(min_confidence=min_confidence)
        self.stability_tracker = TemporalStabilityTracker(
            window_size=stability_window,
            min_samples=stability_samples,
            max_std_dev_mm=max_std_dev_mm,
        )
        self.freshness_checker = TargetFreshnessChecker(max_age_seconds=1.5)
        self.visualizer = VisionVisualizer()

        # Internal state
        self.last_target: Optional[RobotTarget] = None
        self.last_detections: List[Detection] = []

    def reset_history(self, class_name: Optional[str] = None) -> None:
        """Reset temporal stability history."""
        self.stability_tracker.reset(class_name)

    def process_frame(
        self,
        frame: Frame,
        requested_class: str,
        visualize: bool = False,
    ) -> Tuple[RobotTarget, Optional[np.ndarray]]:
        """
        Execute full perception pipeline on a single frame.

        :param frame: Captured input frame.
        :param requested_class: Object label requested by task/Member 2 (e.g. 'red_block').
        :param visualize: If True, renders annotated debug image.
        :return: (RobotTarget, Optional annotated BGR image)
        """
        # Step 1: Detect objects in frame
        detections = self.detector.detect(frame)
        self.last_detections = detections

        # Step 2: Object Selection & Ambiguity Check
        candidate_det, sel_status, sel_msg = self.selector.select(detections, requested_class)

        annotated_image: Optional[np.ndarray] = None

        # Handle negative selection outcomes (NOT_FOUND, AMBIGUOUS, LOW_CONFIDENCE)
        if sel_status != TargetStatus.VALID or candidate_det is None:
            target = create_invalid_robot_target(
                class_name=requested_class,
                status=sel_status,
                message=sel_msg,
            )
            self.last_target = target
            if visualize:
                annotated_image = self.visualizer.draw_overlay(frame, detections, target)
            return target, annotated_image

        # Step 3: Calibration & Coordinate Transform (Pixel -> dobot_base X, Y, Z in mm)
        point_3d, coordinate_valid, coordinate_msg = self.transformer.pixel_to_robot_3d(
            pixel=candidate_det.center,
            class_name=requested_class,
        )

        if not coordinate_valid:
            target = create_invalid_robot_target(
                class_name=requested_class,
                status=TargetStatus.OUT_OF_REACH,
                message=coordinate_msg or "Target outside calibrated camera coverage",
                position=point_3d,
                confidence=candidate_det.confidence,
                source_detection_id=candidate_det.detection_id,
            )
            self.last_target = target
            if visualize:
                annotated_image = self.visualizer.draw_overlay(frame, detections, target)
            return target, annotated_image

        # Step 4: Temporal Stability Check
        filtered_point, stab_status, stab_score, stab_msg = self.stability_tracker.update(
            class_name=requested_class,
            new_point=point_3d,
        )

        if stab_status != TargetStatus.VALID:
            target = create_invalid_robot_target(
                class_name=requested_class,
                status=TargetStatus.UNSTABLE,
                message=stab_msg,
                position=filtered_point,
                confidence=candidate_det.confidence,
                source_detection_id=candidate_det.detection_id,
            )
            self.last_target = target
            if visualize:
                annotated_image = self.visualizer.draw_overlay(frame, detections, target)
            return target, annotated_image

        # Step 5: Construct Verified RobotTarget for Member 4
        target = RobotTarget(
            target_id=f"tgt_{uuid.uuid4().hex[:8]}",
            class_name=requested_class,
            position=filtered_point,
            confidence=candidate_det.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            coordinate_frame="dobot_base",
            valid=True,
            status=TargetStatus.VALID,
            stability_score=stab_score,
            source_detection_id=candidate_det.detection_id,
            message=f"Valid target verified at X:{filtered_point.x:.1f}, Y:{filtered_point.y:.1f}, Z:{filtered_point.z:.1f} mm",
        )

        self.last_target = target
        if visualize:
            annotated_image = self.visualizer.draw_overlay(frame, detections, target)

        return target, annotated_image

    def capture_and_process(
        self,
        requested_class: str,
        visualize: bool = False,
    ) -> Tuple[RobotTarget, Optional[np.ndarray]]:
        """
        Capture frame from camera and process end-to-end.
        """
        if not self.camera.is_opened():
            opened = self.camera.open()
            if not opened:
                target = create_invalid_robot_target(
                    class_name=requested_class,
                    status=TargetStatus.INVALID,
                    message=f"Camera source '{self.camera.name}' could not be opened.",
                )
                return target, None

        frame = self.camera.read()
        if frame is None:
            target = create_invalid_robot_target(
                class_name=requested_class,
                status=TargetStatus.INVALID,
                message="Failed to read frame from camera.",
            )
            return target, None

        return self.process_frame(frame, requested_class, visualize=visualize)
