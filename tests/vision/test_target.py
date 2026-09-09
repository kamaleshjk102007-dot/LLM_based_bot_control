"""Unit tests for vision.target."""

import time
from datetime import datetime, timezone, timedelta
from vision.models import BoundingBox, Detection, Point2D, Point3D, RobotTarget, TargetStatus
from vision.target import TargetFreshnessChecker, TargetSelector


def make_dummy_detection(det_id: str, class_name: str, conf: float, cx: float = 100.0, cy: float = 100.0) -> Detection:
    bb = BoundingBox(x1=int(cx - 20), y1=int(cy - 20), x2=int(cx + 20), y2=int(cy + 20))
    return Detection(
        detection_id=det_id,
        class_name=class_name,
        confidence=conf,
        bbox=bb,
        center=Point2D(x=cx, y=cy),
    )


def test_target_selector_not_found():
    selector = TargetSelector(min_confidence=0.60)
    dets = [make_dummy_detection("d1", "blue_block", 0.90)]

    det, status, msg = selector.select(dets, "red_block")
    assert det is None
    assert status == TargetStatus.NOT_FOUND
    assert "No object of class 'red_block'" in msg


def test_target_selector_low_confidence():
    selector = TargetSelector(min_confidence=0.75)
    dets = [make_dummy_detection("d1", "red_block", 0.40)]

    det, status, msg = selector.select(dets, "red_block")
    assert det is None
    assert status == TargetStatus.LOW_CONFIDENCE
    assert "confidence" in msg


def test_target_selector_single_match_valid():
    selector = TargetSelector(min_confidence=0.60)
    dets = [
        make_dummy_detection("d1", "red_block", 0.92),
        make_dummy_detection("d2", "blue_block", 0.88),
    ]

    det, status, msg = selector.select(dets, "red_block")
    assert det is not None
    assert det.detection_id == "d1"
    assert status == TargetStatus.VALID


def test_target_selector_multiple_matches_ambiguous():
    """Verify cardinal rule: 2+ objects must return AMBIGUOUS, never silently pick one!"""
    selector = TargetSelector(min_confidence=0.60)
    dets = [
        make_dummy_detection("d1", "red_block", 0.94, cx=150, cy=150),
        make_dummy_detection("d2", "red_block", 0.91, cx=350, cy=250),
    ]

    det, status, msg = selector.select(dets, "red_block")
    assert det is None
    assert status == TargetStatus.AMBIGUOUS
    assert "Multiple" in msg


def test_target_freshness():
    freshness = TargetFreshnessChecker(max_age_seconds=1.0)

    # Fresh target
    fresh_target = RobotTarget(
        target_id="tgt_fresh",
        class_name="red_block",
        position=Point3D(x=200, y=0, z=0),
        confidence=0.9,
        timestamp=datetime.now(timezone.utc).isoformat(),
        coordinate_frame="dobot_base",
        valid=True,
        status=TargetStatus.VALID,
    )
    is_fresh, age, reason = freshness.check(fresh_target)
    assert is_fresh is True
    assert age < 0.5
    assert reason is None

    # Stale target (created 3 seconds ago)
    past_time = (datetime.now(timezone.utc) - timedelta(seconds=3.0)).isoformat()
    stale_target = RobotTarget(
        target_id="tgt_stale",
        class_name="red_block",
        position=Point3D(x=200, y=0, z=0),
        confidence=0.9,
        timestamp=past_time,
        coordinate_frame="dobot_base",
        valid=True,
        status=TargetStatus.VALID,
    )
    is_fresh, age, reason = freshness.check(stale_target)
    assert is_fresh is False
    assert age >= 3.0
    assert "STALE" in reason
