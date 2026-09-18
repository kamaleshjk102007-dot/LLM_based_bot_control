"""
Calibration module for Member 3.
Implements planar homography calibration mapping camera pixel coordinates (u, v)
to physical robot base coordinates (X, Y) with rigorous hold-out validation metrics.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from .models import Point2D


class ValidationMetrics(BaseModel):
    """Holds calibration accuracy metrics evaluated on held-out reference points."""
    mean_x_error_mm: float = Field(..., description="Mean absolute error along X axis (mm)")
    mean_y_error_mm: float = Field(..., description="Mean absolute error along Y axis (mm)")
    mean_euclidean_error_mm: float = Field(..., description="Mean Euclidean error on tabletop plane (mm)")
    max_euclidean_error_mm: float = Field(..., description="Maximum single-point Euclidean error (mm)")
    num_validation_points: int = Field(..., description="Number of held-out validation points evaluated")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TabletopCalibration:
    """
    Computes, validates, and applies planar homography between camera image plane
    and the robot's physical tabletop workspace.
    """

    def __init__(self, homography_matrix: Optional[np.ndarray] = None):
        self.homography: Optional[np.ndarray] = None
        self.inv_homography: Optional[np.ndarray] = None
        self.reference_points: List[Dict[str, Any]] = []
        self.last_validation: Optional[ValidationMetrics] = None

        if homography_matrix is not None:
            self.set_matrix(homography_matrix)

    @property
    def is_calibrated(self) -> bool:
        return self.homography is not None and self.homography.shape == (3, 3)

    def set_matrix(self, matrix: np.ndarray) -> None:
        """Directly set and verify homography matrix."""
        mat = np.array(matrix, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {mat.shape}")
        # Normalize matrix so H[2, 2] == 1.0
        if mat[2, 2] != 0:
            mat = mat / mat[2, 2]
        self.homography = mat
        self.inv_homography = np.linalg.inv(mat)

    def calibrate(
        self,
        point_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        method: int = 0,
    ) -> bool:
        """
        Compute homography from point correspondences.
        :param point_pairs: List of ((pixel_u, pixel_v), (robot_x, robot_y))
        :param method: cv2.RANSAC or 0 (least squares)
        :return: True if calibration succeeded
        """
        if len(point_pairs) < 4:
            raise ValueError(f"Planar homography requires at least 4 point pairs, got {len(point_pairs)}")

        src_pts = np.array([p[0] for p in point_pairs], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([p[1] for p in point_pairs], dtype=np.float32).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, method)
        if H is None:
            return False

        self.set_matrix(H)
        self.reference_points = [
            {"pixel": {"u": float(p[0][0]), "v": float(p[0][1])}, "robot": {"x": float(p[1][0]), "y": float(p[1][1])}}
            for p in point_pairs
        ]
        return True

    def validate(
        self,
        validation_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> ValidationMetrics:
        """
        Evaluate calibration against held-out ground truth test points.
        :param validation_pairs: List of ((pixel_u, pixel_v), (actual_robot_x, actual_robot_y))
        :return: ValidationMetrics summarizing accuracy
        """
        if not self.is_calibrated:
            raise RuntimeError("Cannot validate uncalibrated system. Calibrate or load matrix first.")

        if len(validation_pairs) == 0:
            raise ValueError("Must provide at least 1 validation pair.")

        x_errors = []
        y_errors = []
        euclidean_errors = []

        for (u, v), (true_x, true_y) in validation_pairs:
            pred_x, pred_y = self.pixel_to_robot(u, v)
            dx = abs(pred_x - true_x)
            dy = abs(pred_y - true_y)
            euc = float(np.sqrt(dx ** 2 + dy ** 2))

            x_errors.append(dx)
            y_errors.append(dy)
            euclidean_errors.append(euc)

        metrics = ValidationMetrics(
            mean_x_error_mm=round(float(np.mean(x_errors)), 3),
            mean_y_error_mm=round(float(np.mean(y_errors)), 3),
            mean_euclidean_error_mm=round(float(np.mean(euclidean_errors)), 3),
            max_euclidean_error_mm=round(float(np.max(euclidean_errors)), 3),
            num_validation_points=len(validation_pairs),
        )
        self.last_validation = metrics
        return metrics

    def pixel_to_robot(self, u: float, v: float) -> Tuple[float, float]:
        """
        Map camera pixel (u, v) to physical robot tabletop (X, Y) in mm.
        """
        if not self.is_calibrated:
            raise RuntimeError("Tabletop is not calibrated. Call calibrate() or load() first.")

        vec = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
        res = self.homography @ vec
        w = res[2, 0]
        if abs(w) < 1e-9:
            raise ValueError("Degenerate projection: homogeneous scale w is nearly zero.")

        robot_x = float(res[0, 0] / w)
        robot_y = float(res[1, 0] / w)
        return (robot_x, robot_y)

    def robot_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """
        Map physical robot tabletop (X, Y) back to image pixel (u, v).
        """
        if not self.is_calibrated:
            raise RuntimeError("Tabletop is not calibrated.")

        vec = np.array([x, y, 1.0], dtype=np.float64).reshape(3, 1)
        res = self.inv_homography @ vec
        w = res[2, 0]
        if abs(w) < 1e-9:
            raise ValueError("Degenerate inverse projection.")

        pixel_u = float(res[0, 0] / w)
        pixel_v = float(res[1, 0] / w)
        return (pixel_u, pixel_v)

    def save(self, filepath: str) -> None:
        """Save calibration data and validation results to JSON file."""
        if not self.is_calibrated:
            raise RuntimeError("Cannot save uncalibrated system.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "homography": self.homography.tolist(),
            "reference_points": self.reference_points,
            "validation": self.last_validation.model_dump() if self.last_validation else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load calibration matrix and reference points from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found at: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.set_matrix(np.array(data["homography"], dtype=np.float64))
        self.reference_points = data.get("reference_points", [])
        if data.get("validation"):
            self.last_validation = ValidationMetrics(**data["validation"])

    @classmethod
    def create_default(cls) -> TabletopCalibration:
        """
        Factory method providing a realistic default tabletop calibration.
        Corresponds to a standard overhead camera looking down on a Dobot tabletop:
        Image corners/grid mapped to typical Dobot Magician Lite pickable zone:
        X: 180 to 300 mm, Y: -120 to +120 mm.
        """
        # 4 representative points:
        # Pixel (u, v) -> Robot (X, Y in mm)
        pairs = [
            ((120.0, 100.0), (300.0, -120.0)),
            ((520.0, 100.0), (300.0, 120.0)),
            ((520.0, 400.0), (180.0, 120.0)),
            ((120.0, 400.0), (180.0, -120.0)),
        ]
        calib = cls()
        calib.calibrate(pairs)
        return calib
