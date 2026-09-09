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
    Translates camera pixel centroids into physical DOBOT Magician Lite coordinates.
    All outputs strictly adhere to the 'dobot_base' frame in millimeters.
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
    ):
        """
        :param calibration: Active TabletopCalibration instance.
        :param table_z_mm: Physical elevation of the tabletop in dobot_base frame.
        :param object_heights: Map of class_name -> physical height in mm.
        :param grasp_offset_mm: Additional offset above the block top (e.g. for approach clearance).
        """
        self.calibration = calibration
        self.table_z_mm = table_z_mm
        self.object_heights = object_heights or self.DEFAULT_OBJECT_HEIGHTS
        self.grasp_offset_mm = grasp_offset_mm

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

        # Step 3: Validate physical reachability for DOBOT Magician Lite
        is_reachable, reason = self.check_reachability(point_3d)

        return point_3d, is_reachable, reason

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
