"""
Coordinate transformation module for Member 3.
Transforms 2D pixel coordinates into 3D robot coordinates in the 'dobot_base' frame.
Supports two Z-estimation modes:
  1. Fixed-physics Z: table_z_mm + object_height + grasp_offset (always available)
  2. Dynamic Z from bounding box apparent size: uses perspective projection to estimate
     the true physical Z height of the block from how large it appears in the image.
     A bigger bounding box = block is physically higher/closer; smaller = lower/farther.
Validates reachability inside the DOBOT Magician Lite workspace envelope.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

from .calibration import TabletopCalibration
from .models import BoundingBox, Point2D, Point3D


class ZEstimator:
    """
    Estimates the physical Z coordinate of a detected block from its apparent bounding-box
    size in the image using the perspective projection inverse formula.

    Physical model
    --------------
    A standard overhead camera is mounted at a fixed height above the tabletop.
    When a block sits flat on the table it appears at the *reference* bounding-box
    size (``ref_bbox_side_px``).  If the block is elevated (e.g. stacked), its
    bounding box appears *larger* in the image (closer to the lens = bigger projection).

    Perspective projection::

        apparent_px / ref_px  =  (camera_height - table_z) / (camera_height - block_z)

    Rearranged to solve for block_z::

        block_z = camera_height - (camera_height - table_z) * (ref_px / apparent_px)

    Parameters
    ----------
    camera_height_mm:
        Physical height of the camera lens above the robot base origin (mm).
        Positive upward.  For a typical overhead rig this is > 300 mm.
    table_z_mm:
        Z coordinate of the bare tabletop in dobot_base frame (mm).  Usually negative.
    ref_bbox_side_px:
        The bounding-box side length (pixels) that a standard block *on the table* produces
        at the centre of the image.  Calibrate once with a ruler or via
        ``calibrate_ref_size()``.
    use_diagonal:
        If True, uses the diagonal of the bounding box instead of the average side length
        to be more robust against non-square detections.
    alpha:
        Smoothing factor in [0, 1] for exponential moving average on the estimated Z
        across successive frames.  1.0 = no smoothing (raw estimate every frame).
    """

    def __init__(
        self,
        camera_height_mm: float = 500.0,
        table_z_mm: float = -50.0,
        ref_bbox_side_px: float = 60.0,
        use_diagonal: bool = False,
        alpha: float = 0.6,
    ):
        self.camera_height_mm  = camera_height_mm
        self.table_z_mm        = table_z_mm
        self.ref_bbox_side_px  = max(ref_bbox_side_px, 1.0)   # guard against /0
        self.use_diagonal      = use_diagonal
        self.alpha             = float(np.clip(alpha, 0.0, 1.0))
        self._smoothed_z: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the EMA smoother (call when switching to a different object)."""
        self._smoothed_z = None

    def estimate_z(
        self,
        bbox: BoundingBox,
        grasp_offset_mm: float = 0.0,
    ) -> float:
        """
        Estimate the physical Z position of the block top surface.

        :param bbox:            Detected bounding box in pixel coordinates.
        :param grasp_offset_mm: Additional vertical offset added to the estimated block
                                top surface (positive = approach higher above the block).
        :return: Estimated Z in mm (dobot_base frame).
        """
        apparent_px = self._apparent_size(bbox)
        raw_z       = self._perspective_z(apparent_px)
        z_with_offset = raw_z + grasp_offset_mm

        # Exponential moving average smoothing to reduce frame-to-frame noise
        if self._smoothed_z is None:
            self._smoothed_z = z_with_offset
        else:
            self._smoothed_z = self.alpha * z_with_offset + (1.0 - self.alpha) * self._smoothed_z

        return round(self._smoothed_z, 2)

    def calibrate_ref_size(
        self,
        bbox: BoundingBox,
    ) -> None:
        """
        Set the reference bounding-box side length from a block that is known to be
        resting flat on the tabletop at a central position.
        Call this once during calibration with a real measurement.
        """
        self.ref_bbox_side_px = max(self._apparent_size(bbox), 1.0)
        self._smoothed_z = None   # reset smoother after recalibration

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apparent_size(self, bbox: BoundingBox) -> float:
        """Return a single pixel measure that represents how large the detection is."""
        if self.use_diagonal:
            return float(np.sqrt(bbox.width ** 2 + bbox.height ** 2))
        return float((bbox.width + bbox.height) / 2.0)

    def _perspective_z(self, apparent_px: float) -> float:
        """
        Apply the inverse perspective formula to obtain estimated Z.

        ::  block_z = camera_height - (camera_height - table_z) * (ref_px / apparent_px)
        """
        if apparent_px < 1e-3:
            # Degenerate: bbox has zero size — return table level
            return self.table_z_mm
        scale = self.ref_bbox_side_px / apparent_px
        depth = (self.camera_height_mm - self.table_z_mm) * scale
        return self.camera_height_mm - depth


class CoordinateTransformer:
    """
    Translates camera pixel centroids into physical DOBOT Magician Lite coordinates.
    All outputs strictly adhere to the 'dobot_base' frame in millimeters.

    Z Estimation Modes
    ------------------
    * **Dynamic** (preferred): When ``z_estimator`` is provided, the Z coordinate is
      computed from the block's apparent bounding-box size in the image via perspective
      projection.  Larger box → block is physically higher (closer to camera).
    * **Fixed-physics** (fallback): Z = ``table_z_mm`` + object height + ``grasp_offset_mm``.
      Used when no ZEstimator is supplied or when a detection has no bounding box.
    """

    # Default physical block heights in millimeters (e.g. standard wooden/plastic blocks)
    DEFAULT_OBJECT_HEIGHTS: Dict[str, float] = {
        "red_block": 25.0,
        "blue_block": 25.0,
        "green_block": 25.0,
        "yellow_block": 25.0,
        "default": 25.0,
    }

    # DOBOT Magician Lite physical workspace boundaries (mm)
    MIN_RADIUS_MM: float = 140.0
    MAX_RADIUS_MM: float = 330.0
    MIN_Z_MM: float = -70.0
    MAX_Z_MM: float = 160.0

    def __init__(
        self,
        calibration: TabletopCalibration,
        table_z_mm: float = -50.0,
        object_heights: Optional[Dict[str, float]] = None,
        grasp_offset_mm: float = 0.0,
        z_estimator: Optional[ZEstimator] = None,
    ):
        """
        :param calibration:     Active TabletopCalibration instance.
        :param table_z_mm:      Physical elevation of the tabletop in dobot_base frame (mm).
        :param object_heights:  Map of class_name -> physical height in mm (fixed-physics fallback).
        :param grasp_offset_mm: Additional offset above the block top for approach clearance (mm).
        :param z_estimator:     Optional ZEstimator for dynamic Z from bounding-box size.
                                When provided, the fixed-physics model is used only as a fallback.
        """
        self.calibration     = calibration
        self.table_z_mm      = table_z_mm
        self.object_heights  = object_heights or self.DEFAULT_OBJECT_HEIGHTS
        self.grasp_offset_mm = grasp_offset_mm
        self.z_estimator     = z_estimator

    def pixel_to_robot_3d(
        self,
        pixel: Point2D,
        class_name: str,
        bbox: Optional[BoundingBox] = None,
    ) -> Tuple[Point3D, bool, Optional[str], str]:
        """
        Transform 2D image pixel into 3D 'dobot_base' coordinates.

        Z is estimated using one of two modes (priority order):
          1. **Dynamic** — if a ``ZEstimator`` is configured AND a ``bbox`` is provided,
             Z is computed from the block's apparent bounding-box size in the image via
             perspective projection (bigger box → block is physically higher).
          2. **Fixed-physics** — Z = table_z_mm + object_height + grasp_offset_mm.
             Always available as a deterministic fallback.

        :param pixel:      2D pixel centroid coordinates (u, v).
        :param class_name: Object class label (used for fixed-physics object height lookup).
        :param bbox:       Detected bounding box (enables dynamic Z estimation).
        :return: (Point3D in dobot_base, is_reachable, diagnostic_message, z_mode)
                 where z_mode is 'dynamic' or 'fixed'.
        """
        # Step 1: Calculate planar X, Y via calibration homography
        robot_x, robot_y = self.calibration.pixel_to_robot(pixel.x, pixel.y)

        # Step 2: Estimate Z
        z_mode: str
        if self.z_estimator is not None and bbox is not None:
            # --- Dynamic Z: perspective projection from bounding-box apparent size ---
            target_z = self.z_estimator.estimate_z(bbox, grasp_offset_mm=self.grasp_offset_mm)
            z_mode   = "dynamic"
        else:
            # --- Fixed-physics Z: table + object height + approach offset ---
            obj_height = self.object_heights.get(class_name, self.object_heights.get("default", 25.0))
            target_z   = self.table_z_mm + obj_height + self.grasp_offset_mm
            z_mode     = "fixed"

        point_3d = Point3D(
            x=round(robot_x, 2),
            y=round(robot_y, 2),
            z=round(target_z, 2),
        )

        # Step 3: Validate physical reachability for DOBOT Magician Lite
        is_reachable, reason = self.check_reachability(point_3d)

        return point_3d, is_reachable, reason, z_mode

    def check_reachability(self, point: Point3D) -> Tuple[bool, Optional[str]]:
        """
        Verify that coordinates lie within DOBOT Magician Lite kinematic envelope.
        """
        radius = float(np.sqrt(point.x ** 2 + point.y ** 2))

        if radius < self.MIN_RADIUS_MM:
            return False, f"Target too close to robot base: radius {radius:.1f}mm < {self.MIN_RADIUS_MM}mm"
        if radius > self.MAX_RADIUS_MM:
            return False, f"Target exceeds robot arm reach: radius {radius:.1f}mm > {self.MAX_RADIUS_MM}mm"
        if point.x <= 0:
            return False, f"Target is behind robot base plane: X={point.x:.1f}mm <= 0"
        if not (self.MIN_Z_MM <= point.z <= self.MAX_Z_MM):
            return False, f"Target Z={point.z:.1f}mm outside allowable vertical range [{self.MIN_Z_MM}, {self.MAX_Z_MM}]"

        return True, None
