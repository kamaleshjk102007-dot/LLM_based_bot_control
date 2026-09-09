"""Unit tests for vision.target.TemporalStabilityTracker."""

from vision.models import Point3D, TargetStatus
from vision.target import TemporalStabilityTracker


def test_stability_tracker_stable_sequence():
    tracker = TemporalStabilityTracker(
        window_size=4,
        min_samples=3,
        max_std_dev_mm=2.0,
    )

    # Sequence of very close points (standard deviation < 0.5mm)
    points = [
        Point3D(x=245.1, y=82.4, z=-25.0),
        Point3D(x=245.3, y=82.6, z=-25.0),
        Point3D(x=245.2, y=82.5, z=-25.0),
        Point3D(x=245.2, y=82.5, z=-25.0),
    ]

    # Frame 1: Accumulating (< min_samples)
    _, s1, _, _ = tracker.update("red_block", points[0])
    assert s1 == TargetStatus.UNSTABLE

    # Frame 2: Accumulating (< min_samples)
    _, s2, _, _ = tracker.update("red_block", points[1])
    assert s2 == TargetStatus.UNSTABLE

    # Frame 3: Meets min_samples (3) and std_dev is small -> VALID!
    pt3, s3, score3, _ = tracker.update("red_block", points[2])
    assert s3 == TargetStatus.VALID
    assert score3 > 0.80
    assert abs(pt3.x - 245.2) < 0.1
    assert abs(pt3.y - 82.5) < 0.1

    # Frame 4: Fully stable
    pt4, s4, score4, _ = tracker.update("red_block", points[3])
    assert s4 == TargetStatus.VALID
    assert score4 > 0.90


def test_stability_tracker_unstable_jitter():
    tracker = TemporalStabilityTracker(
        window_size=4,
        min_samples=3,
        max_std_dev_mm=2.0,
    )

    # Frame jitter: X jumps by 20mm back and forth
    points = [
        Point3D(x=240.0, y=80.0, z=-25.0),
        Point3D(x=255.0, y=80.0, z=-25.0),
        Point3D(x=235.0, y=80.0, z=-25.0),
    ]

    tracker.update("red_block", points[0])
    tracker.update("red_block", points[1])
    _, status, score, msg = tracker.update("red_block", points[2])

    assert status == TargetStatus.UNSTABLE
    assert "jitter detected" in msg


def test_stability_tracker_sudden_jump_resets():
    tracker = TemporalStabilityTracker(
        window_size=4,
        min_samples=3,
        max_jump_mm=15.0,
    )

    p1 = Point3D(x=200.0, y=50.0, z=-25.0)
    tracker.update("red_block", p1)

    # Jump of 50mm
    p2 = Point3D(x=250.0, y=50.0, z=-25.0)
    _, status, _, msg = tracker.update("red_block", p2)

    assert status == TargetStatus.UNSTABLE
    assert "jumped" in msg
