"""Unit tests for vision.detector."""

import pytest

from vision.camera import MockCamera
from vision.detector import ColorShapeDetector


def test_detect_single_red_block():
    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(300.0, 200.0), size=(60, 60))
    frame = cam.read()
    cam.release()

    detector = ColorShapeDetector()
    detections = detector.detect(frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "red_block"
    assert det.confidence >= 0.70
    assert abs(det.center.x - 300.0) < 5.0
    assert abs(det.center.y - 200.0) < 5.0
    assert det.bbox.width > 45
    assert det.bbox.height > 45


def test_detect_multiple_colors():
    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(200.0, 200.0), size=(50, 50))
    cam.add_object("blue_block", center=(400.0, 200.0), size=(50, 50))
    cam.add_object("green_block", center=(300.0, 350.0), size=(50, 50))
    frame = cam.read()
    cam.release()

    detector = ColorShapeDetector()
    detections = detector.detect(frame)

    classes_found = {d.class_name for d in detections}
    assert "red_block" in classes_found
    assert "blue_block" in classes_found
    assert "green_block" in classes_found
    assert len(detections) == 3


def test_detect_empty_canvas():
    cam = MockCamera(width=640, height=480)
    cam.open()
    # No objects added
    frame = cam.read()
    cam.release()

    detector = ColorShapeDetector()
    detections = detector.detect(frame)
    assert len(detections) == 0


def test_detect_rejects_tiny_noise():
    cam = MockCamera(width=640, height=480)
    cam.open()
    # Block smaller than min_area (300 px) -> 10x10 = 100 px
    cam.add_object("red_block", center=(250.0, 250.0), size=(10, 10))
    frame = cam.read()
    cam.release()

    detector = ColorShapeDetector(min_area=300)
    detections = detector.detect(frame)
    assert len(detections) == 0
