"""Create a measured pixel-to-Webots-coordinate calibration for the demo world.

This is simulation-only. It reads a Webots camera frame, detects four fixed
colored markers, queries their known simulated coordinates, and saves the
existing TabletopCalibration JSON format. It never invokes motion planning or
the gateway.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.adapters.webots import WebotsClient
from vision.calibration import TabletopCalibration
from vision.camera import WebotsTcpCamera
from vision.detector import ColorShapeDetector
import math

MARKER_RANGES = {
    "magenta_marker": [((140, 100, 60), (169, 255, 255))],
    "cyan_marker": [((86, 100, 60), (99, 255, 255))],
    "green_marker": [((35, 70, 60), (85, 255, 255))],
    "yellow_marker": [((20, 100, 100), (35, 255, 255))],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the simulated Webots camera")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = WebotsClient(timeout=10.0)
    references = client.request({"type": "calibration_references"})
    camera = WebotsTcpCamera(client.request)
    if not camera.open():
        print("Calibration failed: Webots camera is not available.")
        return 2
    detector = ColorShapeDetector(color_ranges=MARKER_RANGES)
    markers = references.get("markers", {})
    frame = None
    by_class = {}
    # A Webots controller can expose one stale/blank camera buffer just after a
    # world reload. Require a complete marker frame instead of failing on it.
    for _ in range(10):
        candidate = camera.read()
        if candidate is None:
            time.sleep(0.1)
            continue
        detections = detector.detect(candidate)
        candidate_by_class = {}
        for detection in detections:
            candidate_by_class.setdefault(detection.class_name, []).append(detection)
        if all(len(candidate_by_class.get(name, [])) == 1 for name in markers):
            frame = candidate
            by_class = candidate_by_class
            break
        by_class = candidate_by_class
        time.sleep(0.1)
    camera.release()
    if frame is None:
        marker_counts = {name: len(by_class.get(name, [])) for name in markers}
        print({
            "calibration_failed": "did not observe exactly one of each marker after camera warmup",
            "marker_counts": marker_counts,
        })
        return 2

    pairs = []
    for name, robot_point in markers.items():
        matches = by_class.get(name, [])
        if len(matches) != 1:
            print(f"Calibration failed: expected one {name}, found {len(matches)}.")
            return 2
        pairs.append(((matches[0].center.x, matches[0].center.y), (robot_point["x"], robot_point["y"])))

    calibration = TabletopCalibration()
    calibration.calibrate(pairs)

    red_detections = ColorShapeDetector(
        color_ranges={"red_block": [((0, 100, 180), (12, 255, 255)), ((168, 100, 180), (180, 255, 255))]}
    ).detect(frame)
    red_matches = [item for item in red_detections if item.class_name == "red_block"]
    if len(red_matches) != 1:
        print(f"Calibration failed: expected one red_block validation target, found {len(red_matches)}.")
        return 2

    expected = references["validation_target"]
    measured_x, measured_y = calibration.pixel_to_robot(
        red_matches[0].center.x,
        red_matches[0].center.y,
    )
    validation_error_mm = math.dist(
        (measured_x, measured_y),
        (expected["x"], expected["y"]),
    )
    if validation_error_mm > 2.0:
        print({
            "calibration_failed": "held-out red target error exceeds 2.0 mm",
            "expected_xy_mm": expected,
            "measured_xy_mm": {
                "x": round(measured_x, 3),
                "y": round(measured_y, 3),
            },
            "error_mm": round(validation_error_mm, 3),
        })
        return 2

    calibration.save(str(args.output))
    print({
        "calibration": str(args.output),
        "marker_count": len(pairs),
        "held_out_target_error_mm": round(validation_error_mm, 3),
        "coordinate_frame": references.get("coordinate_frame"),
        "target_plane_z_mm": references.get("target_plane_z_mm"),
        "next_command": (
            "python examples/camera_to_webots_demo.py --calibration "
            f"{args.output} --table-z-mm {references.get('target_plane_z_mm')}"
        ),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
