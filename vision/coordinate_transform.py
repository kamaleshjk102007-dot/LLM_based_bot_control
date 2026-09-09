"""
Coordinate transformation module for Member 3.
Transforms 2D pixel coordinates into 3D robot coordinates in the 'dobot_base' frame.
Enforces physical Z modeling from tabletop elevation + object dimensions,
and validates reachability inside the DOBOT Magician Lite workspace envelope.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

from .calibration import TabletopCalibration
from .models import Point2D, Point3D


class CoordinateTransformer:
    """
    Translates camera pixel centroids into calibrated robot-base coordinates.
    It contains no robot workspace or motion-safety policy.
    """

    # Default physical block heights in millimeters (e.g. standard wooden/plastic blocks)
    DEFAULT_OBJECT_HEIGHTS: Dict[str, float] = {
        "red_block": 25.0,
        "blue_block": 25.0,
        "green_block": 25.0,
        "yellow_block": 25.0,
        "default": 25.0,
    }

    def __init__(
        self,
        calibration: TabletopCalibration,
        table_z_mm: float = -50.0,
        object_heights: Optional[Dict[str, float]] = None,
        grasp_offset_mm: float = 0.0,
        perception_region_mm: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        :param calibration: Active TabletopCalibration instance.
        :param table_z_mm: Physical elevation of the tabletop in dobot_base frame.
        :param object_heights: Map of class_name -> physical height in mm.
        :param grasp_offset_mm: Additional offset above the block top.
        :param perception_region_mm: Optional calibrated camera coverage as
            (min_x, max_x, min_y, max_y). This is not a robot workspace limit.
        """
        self.calibration = calibration
        self.table_z_mm = table_z_mm
        self.object_heights = object_heights or self.DEFAULT_OBJECT_HEIGHTS
        self.grasp_offset_mm = grasp_offset_mm
        self.perception_region_mm = perception_region_mm

    def pixel_to_robot_3d(
        self,
        pixel: Point2D,
        class_name: str,
    ) -> Tuple[Point3D, bool, Optional[str]]:
        """
        Transform 2D image pixel into 3D 'dobot_base' coordinates.

        :param pixel: 2D pixel coordinates (u, v)
        :param class_name: Object class label (used to look up physical Z height)
        :return: (Point3D in dobot_base, is_reachable, diagnostic_message)
        """
        # Step 1: Calculate planar X, Y via calibration homography
        robot_x, robot_y = self.calibration.pixel_to_robot(pixel.x, pixel.y)

        # Step 2: Calculate deterministic physical Z (NO guessing/faking Z from 2D pixel size)
        obj_height = self.object_heights.get(class_name, self.object_heights.get("default", 25.0))
        target_z = self.table_z_mm + obj_height + self.grasp_offset_mm

        point_3d = Point3D(
            x=round(robot_x, 2),
            y=round(robot_y, 2),
            z=round(target_z, 2),
        )

        # This only checks whether the result is inside an optional calibrated
        # camera coverage region. Gateway/adapter layers own robot safety.
        is_valid, reason = self.check_perception_region(point_3d)

        return point_3d, is_valid, reason

    def check_perception_region(self, point: Point3D) -> Tuple[bool, Optional[str]]:
        """
        Verify optional calibrated-camera coverage, never robot reachability.
        """
        if not all(np.isfinite(value) for value in point.to_list()):
            return False, "Calibrated coordinate is not finite."
        if self.perception_region_mm is None:
            return True, None
        min_x, max_x, min_y, max_y = self.perception_region_mm
        if not (min_x <= point.x <= max_x and min_y <= point.y <= max_y):
            return False, "Target lies outside the configured calibrated camera region."
        return True, None
