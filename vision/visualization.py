"""
Visualization and debugging overlay module for Member 3.
Renders real-time HUD overlays displaying detections, pixel centers,
transformed robot base coordinates (X, Y, Z in mm), and target validity banners.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .models import Detection, Frame, RobotTarget, TargetStatus


class VisionVisualizer:
    """Renders debug overlays on camera frames for simulation and live monitoring."""

    # Visual palette (BGR)
    COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
        "red_block": (0, 0, 230),
        "blue_block": (230, 80, 0),
        "green_block": (0, 200, 0),
        "yellow_block": (0, 215, 235),
        "default": (180, 180, 180),
    }

    STATUS_BG_COLORS: Dict[TargetStatus, Tuple[int, int, int]] = {
        TargetStatus.VALID: (40, 160, 40),          # Green
        TargetStatus.AMBIGUOUS: (0, 140, 255),       # Orange
        TargetStatus.UNSTABLE: (0, 200, 220),        # Yellow
        TargetStatus.NOT_FOUND: (80, 80, 80),        # Dark Gray
        TargetStatus.LOW_CONFIDENCE: (50, 50, 200),  # Red-Orange
        TargetStatus.OUT_OF_REACH: (0, 0, 200),      # Red
        TargetStatus.STALE: (100, 100, 100),         # Gray
        TargetStatus.INVALID: (0, 0, 200),          # Red
    }

    def __init__(self, font_scale: float = 0.5, line_thickness: int = 2):
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_overlay(
        self,
        frame: Frame,
        detections: List[Detection],
        target: Optional[RobotTarget] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, detection labels, robot coordinates, and status HUD banner.

        :param frame: Source camera frame.
        :param detections: All detections in the frame.
        :param target: Evaluated RobotTarget (if available).
        :return: Annotated BGR image as a NumPy array.
        """
        canvas = frame.image.copy()
        h, w = canvas.shape[:2]

        # 1. Draw individual detections
        for det in detections:
            color = self.COLOR_MAP.get(det.class_name, self.COLOR_MAP["default"])
            bb = det.bbox

            # Draw bounding rectangle
            cv2.rectangle(canvas, (bb.x1, bb.y1), (bb.x2, bb.y2), color, self.line_thickness)

            # Draw centroid circle and crosshair
            cx, cy = int(round(det.center.x)), int(round(det.center.y))
            cv2.circle(canvas, (cx, cy), 4, (0, 255, 255), -1)
            cv2.line(canvas, (cx - 8, cy), (cx + 8, cy), (0, 255, 255), 1)
            cv2.line(canvas, (cx, cy - 8), (cx, cy + 8), (0, 255, 255), 1)

            # Label text: class + confidence
            label = f"{det.class_name} ({det.confidence:.2f})"
            (tw, th), baseline = cv2.getTextSize(label, self.font, self.font_scale, 1)

            # Draw background tag for text
            tag_y1 = max(0, bb.y1 - th - baseline - 4)
            tag_y2 = bb.y1
            cv2.rectangle(canvas, (bb.x1, tag_y1), (bb.x1 + tw + 6, tag_y2), color, -1)
            cv2.putText(
                canvas,
                label,
                (bb.x1 + 3, tag_y2 - baseline - 1),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. If target is active, highlight the target and render coordinates
        if target is not None:
            # Render HUD banner at top of image
            banner_height = 55
            hud_color = self.STATUS_BG_COLORS.get(target.status, (80, 80, 80))
            cv2.rectangle(canvas, (0, 0), (w, banner_height), hud_color, -1)

            status_text = f"TARGET: [{target.status.value}] {target.class_name.upper()}"
            cv2.putText(
                canvas,
                status_text,
                (12, 22),
                self.font,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if target.valid:
                coord_text = (
                    f"dobot_base -> X:{target.position.x:+.1f}mm | "
                    f"Y:{target.position.y:+.1f}mm | Z:{target.position.z:+.1f}mm "
                    f"(Conf:{target.confidence * 100:.0f}%, Stab:{target.stability_score * 100:.0f}%)"
                )
            else:
                coord_text = f"REASON: {target.message or 'Verification failed'}"

            cv2.putText(
                canvas,
                coord_text,
                (12, 44),
                self.font,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

            # If target has a valid source detection, draw a target ring
            if target.valid and target.source_detection_id:
                matching_dets = [d for d in detections if d.detection_id == target.source_detection_id]
                if matching_dets:
                    tgt_det = matching_dets[0]
                    tcx, tcy = int(round(tgt_det.center.x)), int(round(tgt_det.center.y))
                    cv2.circle(canvas, (tcx, tcy), 18, (0, 255, 0), 2)
                    cv2.circle(canvas, (tcx, tcy), 26, (0, 255, 0), 1)
                    coord_tag = f"X:{target.position.x:.1f} Y:{target.position.y:.1f} Z:{target.position.z:.1f}"
                    cv2.putText(
                        canvas,
                        coord_tag,
                        (tcx - 50, tgt_det.bbox.y2 + 18),
                        self.font,
                        0.45,
                        (0, 230, 0),
                        1,
                        cv2.LINE_AA,
                    )

        return canvas
