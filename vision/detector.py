"""
Detector abstraction and implementations for Member 3.
Provides high-accuracy color/shape block detection and pluggable YOLO detector.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import uuid
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .models import BoundingBox, Detection, Frame, Point2D


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        """Process a frame and return detected objects."""
        pass


class ColorShapeDetector(BaseDetector):
    """
    Robust color and shape based object detector for tabletop blocks.
    Identifies colored blocks (red, green, blue, yellow) using HSV thresholding,
    morphological cleaning, and geometric contour verification.
    """

    DEFAULT_COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
        "red_block": [
            # Low-hue red
            ((0, 100, 80), (10, 255, 255)),
            # High-hue red wrap-around
            ((170, 100, 80), (180, 255, 255)),
        ],
        "blue_block": [
            ((100, 100, 60), (135, 255, 255)),
        ],
        "green_block": [
            ((35, 70, 60), (85, 255, 255)),
        ],
        "yellow_block": [
            ((20, 100, 100), (35, 255, 255)),
        ],
    }

    def __init__(
        self,
        color_ranges: Optional[Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]] = None,
        min_area: int = 300,
        max_area: int = 150000,
        min_solidity: float = 0.6,
        aspect_ratio_range: Tuple[float, float] = (0.4, 2.5),
    ):
        self.color_ranges = color_ranges or self.DEFAULT_COLOR_RANGES
        self.min_area = min_area
        self.max_area = max_area
        self.min_solidity = min_solidity
        self.aspect_ratio_range = aspect_ratio_range
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, frame: Frame) -> List[Detection]:
        img = frame.image
        if img is None or img.size == 0:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detections: List[Detection] = []

        for class_name, ranges in self.color_ranges.items():
            # Combine ranges for this color (e.g. red wrapping around 0 and 180)
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Morphological noise removal
            cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self._morph_kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self._morph_kernel)

            # Extract contours
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (self.min_area <= area <= self.max_area):
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if w == 0 or h == 0:
                    continue

                aspect_ratio = float(w) / float(h)
                if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                    continue

                # Solidity: contour area divided by convex hull area
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0.0
                if solidity < self.min_solidity:
                    continue

                # Precise centroid via moments
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                else:
                    cx = float(x + w / 2.0)
                    cy = float(y + h / 2.0)

                # Confidence calculation: function of solidity and bbox fill ratio
                bbox_area = float(w * h)
                fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
                confidence = float(np.clip(0.5 * solidity + 0.5 * fill_ratio, 0.50, 0.99))

                det_id = f"det_{uuid.uuid4().hex[:8]}"
                detections.append(
                    Detection(
                        detection_id=det_id,
                        class_name=class_name,
                        confidence=round(confidence, 3),
                        bbox=BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h),
                        center=Point2D(x=round(cx, 2), y=round(cy, 2)),
                    )
                )

        return detections


class YOLODetector(BaseDetector):
    """
    Pluggable YOLO detector wrapper using Ultralytics.
    Allows seamlessly switching from color detector to deep learning detector.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
    ):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise ImportError(
                "Ultralytics is not installed. Run 'pip install ultralytics' to use YOLODetector."
            )

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes

    def detect(self, frame: Frame) -> List[Detection]:
        if frame.image is None:
            return []

        results = self.model(frame.image, conf=self.confidence_threshold, verbose=False)
        detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.model.names[cls_id]

                if self.target_classes and class_name not in self.target_classes:
                    continue

                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                det_id = f"det_{uuid.uuid4().hex[:8]}"
                detections.append(
                    Detection(
                        detection_id=det_id,
                        class_name=class_name,
                        confidence=round(conf, 3),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        center=Point2D(x=round(cx, 2), y=round(cy, 2)),
                    )
                )

        return detections
