"""Unit and integration tests for vision.pipeline."""

from vision.camera import MockCamera
from vision.models import TargetStatus
from vision.pipeline import VisionPipeline


def test_pipeline_single_target_stabilization_and_valid_target():
    # Setup mock camera with single red block
    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(320.0, 250.0), size=(50, 50))

    pipeline = VisionPipeline(camera=cam, stability_samples=3)

    # Frame 1 & 2: Stabilizing
    t1, _ = pipeline.capture_and_process("red_block")
    assert t1.valid is False
    assert t1.status == TargetStatus.UNSTABLE

    t2, _ = pipeline.capture_and_process("red_block")
    assert t2.valid is False
    assert t2.status == TargetStatus.UNSTABLE

    # Frame 3: Stabilized and VALID
    t3, debug_img = pipeline.capture_and_process("red_block", visualize=True)
    assert t3.valid is True
    assert t3.status == TargetStatus.VALID
    assert t3.coordinate_frame == "dobot_base"
    assert t3.class_name == "red_block"
    assert t3.confidence >= 0.70
    assert t3.position.x > 150.0  # Reachable workspace
    assert t3.position.z == -25.0 # -50 table + 25 height
    assert debug_img is not None
    assert debug_img.shape == (480, 640, 3)

    cam.release()


def test_pipeline_ambiguous_two_red_blocks():
    cam = MockCamera(width=640, height=480)
    cam.open()
    # Add two distinct red blocks
    cam.add_object("red_block", center=(200.0, 200.0), size=(50, 50))
    cam.add_object("red_block", center=(450.0, 300.0), size=(50, 50))

    pipeline = VisionPipeline(camera=cam)
    target, _ = pipeline.capture_and_process("red_block")

    assert target.valid is False
    assert target.status == TargetStatus.AMBIGUOUS
    assert "Multiple" in target.message

    cam.release()


def test_pipeline_not_found():
    cam = MockCamera(width=640, height=480)
    cam.open()
    # Only blue block on table
    cam.add_object("blue_block", center=(300.0, 200.0), size=(50, 50))

    pipeline = VisionPipeline(camera=cam)
    target, _ = pipeline.capture_and_process("red_block")

    assert target.valid is False
    assert target.status == TargetStatus.NOT_FOUND

    cam.release()
