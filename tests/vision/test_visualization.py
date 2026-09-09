"""Unit tests for vision.visualization."""

import numpy as np
from vision.camera import MockCamera
from vision.detector import ColorShapeDetector
from vision.models import Point3D, RobotTarget, TargetStatus
from vision.visualization import VisionVisualizer


def test_visualizer_draw_overlay_with_detections():
    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(320.0, 240.0), size=(60, 60))
    frame = cam.read()
    cam.release()

    detector = ColorShapeDetector()
    detections = detector.detect(frame)
    assert len(detections) >= 1

    visualizer = VisionVisualizer()
    annotated = visualizer.draw_overlay(frame, detections)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (480, 640, 3)
    # Check that canvas was modified from original
    assert not np.array_equal(annotated, frame.image)


def test_visualizer_valid_target_hud():
    cam = MockCamera(width=640, height=480)
    cam.open()
    frame = cam.read()
    cam.release()

    target = RobotTarget(
        target_id="tgt_test",
        class_name="red_block",
        position=Point3D(x=240.0, y=0.0, z=-25.0),
        confidence=0.95,
        coordinate_frame="dobot_base",
        valid=True,
        status=TargetStatus.VALID,
    )

    visualizer = VisionVisualizer()
    annotated = visualizer.draw_overlay(frame, [], target=target)

    assert annotated.shape == (480, 640, 3)
    # Top banner area should be colored green (BGR around (40, 160, 40))
    banner_pixel = annotated[10, 10]
    # Green channel should be dominant in banner
    assert banner_pixel[1] > banner_pixel[0] and banner_pixel[1] > banner_pixel[2]


def test_visualizer_ambiguous_target_hud():
    cam = MockCamera(width=640, height=480)
    cam.open()
    frame = cam.read()
    cam.release()

    target = RobotTarget(
        target_id="tgt_test",
        class_name="red_block",
        position=Point3D(x=0.0, y=0.0, z=0.0),
        confidence=0.0,
        coordinate_frame="dobot_base",
        valid=False,
        status=TargetStatus.AMBIGUOUS,
        message="Multiple (2) 'red_block' detected without disambiguation rule.",
    )

    visualizer = VisionVisualizer()
    annotated = visualizer.draw_overlay(frame, [], target=target)

    assert annotated.shape == (480, 640, 3)
