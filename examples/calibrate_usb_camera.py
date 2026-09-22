"""Calibrate a physical USB camera against a measured tabletop marker layout.

This tool reads one local camera frame, detects four fixed markers, and saves the
project's existing TabletopCalibration JSON. It never contacts Webots or sends
robot commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.calibration import TabletopCalibration
from vision.camera import USBCamera
from vision.detector import ColorShapeDetector


MARKER_POINTS_MM = {
    "red_block": (120.0, -130.0),
    "green_block": (340.0, -130.0),
    "yellow_block": (340.0, 130.0),
    "blue_block": (120.0, 130.0),
}


def marker_order_is_crossed(pixel_points: dict[str, tuple[float, float]]) -> bool:
    """Reject a self-crossing marker perimeter before fitting a homography."""
    red, green, yellow, blue = (
        pixel_points["red_block"],
        pixel_points["green_block"],
        pixel_points["yellow_block"],
        pixel_points["blue_block"],
    )

    def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
        return orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0

    return intersects(red, green, yellow, blue) or intersects(green, yellow, blue, red)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate a USB camera using red, green, yellow, and blue tabletop markers."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--directshow", action="store_true")
    args = parser.parse_args()

    camera = USBCamera(
        device_index=args.camera_index,
        width=640,
        height=480,
        backend=cv2.CAP_DSHOW if args.directshow else None,
    )
    if not camera.open():
        print(f"Calibration failed: could not open camera index {args.camera_index}.")
        return 2

    by_class = {}
    successful_frames = 0
    for _ in range(60):
        frame = camera.read()
        if frame is None:
            continue
        detections = ColorShapeDetector().detect(frame)
        candidates = {}
        for detection in detections:
            candidates.setdefault(detection.class_name, []).append(detection)
        if all(len(candidates.get(name, [])) == 1 for name in MARKER_POINTS_MM):
            by_class = candidates
            successful_frames += 1
            if successful_frames == 3:
                break
        else:
            successful_frames = 0
    camera.release()

    if successful_frames < 3:
        counts = {name: len(by_class.get(name, [])) for name in MARKER_POINTS_MM}
        print({
            "calibration_failed": "did not observe exactly one of each marker for three consecutive frames",
            "last_valid_marker_counts": counts,
        })
        return 2

    pairs = []
    marker_pixels = {}
    marker_points = {}
    for class_name, point_mm in MARKER_POINTS_MM.items():
        matches = by_class.get(class_name, [])
        if len(matches) != 1:
            print(f"Calibration failed: expected one {class_name}, found {len(matches)}.")
            return 2
        center = matches[0].center
        marker_pixels[class_name] = {"x": round(center.x, 1), "y": round(center.y, 1)}
        marker_points[class_name] = (center.x, center.y)
        pairs.append(((center.x, center.y), point_mm))

    if marker_order_is_crossed(marker_points):
        print({
            "calibration_failed": "marker colours form a crossed perimeter in the camera image",
            "expected_camera_order": ["red_block", "green_block", "yellow_block", "blue_block"],
            "marker_pixels": marker_pixels,
        })
        return 2

    calibration = TabletopCalibration()
    if not calibration.calibrate(pairs):
        print("Calibration failed: homography could not be calculated.")
        return 2
    calibration.save(str(args.output))
    print({
        "calibration": str(args.output),
        "camera_index": args.camera_index,
        "coordinate_frame": "webots_demo_table",
        "marker_count": len(pairs),
        "marker_pixels": marker_pixels,
        "note": "Calibration was saved from four markers. Use a separate held-out target before treating this mapping as verified.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())