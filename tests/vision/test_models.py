"""Unit tests for vision.models."""

import json
import numpy as np
import pytest
from pydantic import ValidationError

from vision.models import (
    BoundingBox,
    Detection,
    Frame,
    Point2D,
    Point3D,
    RobotTarget,
    TargetStatus,
)


def test_point2d_immutability():
    p = Point2D(x=120.5, y=340.2)
    assert p.x == 120.5
    assert p.y == 340.2
    with pytest.raises(ValidationError):
        p.x = 200.0  # Frozen


def test_point3d_distances():
    p1 = Point3D(x=200.0, y=0.0, z=-50.0)
    p2 = Point3D(x=203.0, y=4.0, z=-50.0)
    # Planar distance: sqrt(3^2 + 4^2) = 5.0
    assert pytest.approx(p1.planar_distance_to(p2), 0.001) == 5.0

    p3 = Point3D(x=203.0, y=4.0, z=-38.0)
    # 3D distance: sqrt(3^2 + 4^2 + 12^2) = 13.0
    assert pytest.approx(p1.distance_to(p3), 0.001) == 13.0


def test_bounding_box_metrics():
    bb = BoundingBox(x1=100, y1=150, x2=160, y2=210)
    assert bb.width == 60
    assert bb.height == 60
    assert bb.area == 3600
    assert bb.center.x == 130.0
    assert bb.center.y == 180.0


def test_frame_from_image():
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame.from_image(dummy_img, frame_id="test_01")
    assert frame.width == 640
    assert frame.height == 480
    assert frame.frame_id == "test_01"
    assert frame.timestamp > 0


def test_detection_model():
    bb = BoundingBox(x1=50, y1=50, x2=100, y2=100)
    det = Detection(
        detection_id="det_001",
        class_name="red_block",
        confidence=0.94,
        bbox=bb,
        center=bb.center,
    )
    assert det.class_name == "red_block"
    assert det.confidence == 0.94
    assert det.center.x == 75.0
    assert det.center.y == 75.0

    # Invalid confidence > 1.0
    with pytest.raises(ValidationError):
        Detection(
            detection_id="det_bad",
            class_name="red_block",
            confidence=1.5,
            bbox=bb,
            center=bb.center,
        )


def test_robot_target_member4_contract():
    target = RobotTarget(
        target_id="tgt_001",
        class_name="red_block",
        position=Point3D(x=245.2, y=82.5, z=25.0),
        confidence=0.94,
        coordinate_frame="dobot_base",
        valid=True,
        status=TargetStatus.VALID,
    )

    # Check JSON roundtrip serialization
    json_str = target.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["target_id"] == "tgt_001"
    assert parsed["class_name"] == "red_block"
    assert parsed["position"]["x"] == 245.2
    assert parsed["position"]["y"] == 82.5
    assert parsed["position"]["z"] == 25.0
    assert parsed["coordinate_frame"] == "dobot_base"
    assert parsed["valid"] is True
    assert parsed["status"] == "VALID"
