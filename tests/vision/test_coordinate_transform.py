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


def test_reachability_checks():
    calib = TabletopCalibration.create_default()
    transformer = CoordinateTransformer(calibration=calib)

    # Within valid reach
    valid_point = Point3D(x=220.0, y=50.0, z=-25.0)
    reachable, msg = transformer.check_reachability(valid_point)
    assert reachable is True
    assert msg is None

    # Too far (radius > 330 mm)
    too_far = Point3D(x=350.0, y=100.0, z=-25.0)
    reachable, msg = transformer.check_reachability(too_far)
    assert reachable is False
    assert "exceeds robot arm reach" in msg

    # Too close (radius < 140 mm)
    too_close = Point3D(x=100.0, y=20.0, z=-25.0)
    reachable, msg = transformer.check_reachability(too_close)
    assert reachable is False
    assert "too close" in msg

    # Behind robot (X <= 0)
    behind = Point3D(x=-150.0, y=50.0, z=-25.0)
    reachable, msg = transformer.check_reachability(behind)
    assert reachable is False
    assert "behind robot" in msg
