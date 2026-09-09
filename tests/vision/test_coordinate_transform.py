"""Unit tests for vision.coordinate_transform."""

import pytest

from vision.calibration import TabletopCalibration
from vision.coordinate_transform import CoordinateTransformer
from vision.models import Point2D, Point3D


def test_pixel_to_robot_3d_z_calculation():
    calib = TabletopCalibration.create_default()
    transformer = CoordinateTransformer(
        calibration=calib,
        table_z_mm=-50.0,
        object_heights={"red_block": 25.0},
    )

    pixel = Point2D(x=320.0, y=250.0)
    point_3d, is_reachable, msg = transformer.pixel_to_robot_3d(pixel, "red_block")

    # Z must be table_z + object_height = -50.0 + 25.0 = -25.0 mm
    assert point_3d.z == -25.0
    assert is_reachable is True
    assert msg is None


def test_optional_calibrated_camera_region_checks():
    calib = TabletopCalibration.create_default()
    transformer = CoordinateTransformer(
        calibration=calib,
        perception_region_mm=(180.0, 300.0, -120.0, 120.0),
    )

    # Inside the measured camera coverage region.
    valid_point = Point3D(x=220.0, y=50.0, z=-25.0)
    covered, msg = transformer.check_perception_region(valid_point)
    assert covered is True
    assert msg is None

    # This is camera/calibration coverage only, not a DOBOT safety limit.
    too_far = Point3D(x=350.0, y=100.0, z=-25.0)
    covered, msg = transformer.check_perception_region(too_far)
    assert covered is False
    assert "calibrated camera region" in msg
