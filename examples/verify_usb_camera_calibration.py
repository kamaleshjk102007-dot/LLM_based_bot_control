"""Verify a saved USB-camera tabletop calibration with one held-out target.

This reads the local camera and compares a detected target's mapped XY location
with a manually measured expected XY point. It never contacts Webots or sends
robot commands.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.calibration import TabletopCalibration
from vision.camera import USBCamera
from vision.detector import ColorShapeDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a USB-camera calibration with one held-out target.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--directshow", action="store_true")
    parser.add_argument("--class-name", default="red_block")
    parser.add_argument("--expected-x-mm", type=float, required=True)
    parser.add_argument("--expected-y-mm", type=float, required=True)
    parser.add_argument("--tolerance-mm", type=float, default=5.0)
    args = parser.parse_args()

    calibration = TabletopCalibration()
    calibration.load(str(args.calibration))
    camera = USBCamera(
        device_index=args.camera_index,
        backend=cv2.CAP_DSHOW if args.directshow else None,
    )
    if not camera.open():
        print(f"Verification failed: could not open camera index {args.camera_index}.")
        return 2

    matching = None
    for _ in range(60):
        frame = camera.read()
        if frame is None:
            continue
        detections = ColorShapeDetector().detect(frame)
        matches = [item for item in detections if item.class_name == args.class_name]
        if len(matches) == 1:
            matching = matches[0]
            break
    camera.release()

    if matching is None:
        print(f"Verification failed: expected exactly one {args.class_name} target.")
        return 2

    measured_x, measured_y = calibration.pixel_to_robot(matching.center.x, matching.center.y)
    error_mm = math.dist((measured_x, measured_y), (args.expected_x_mm, args.expected_y_mm))
    verified = error_mm <= args.tolerance_mm
    print({
        "verified": verified,
        "class_name": args.class_name,
        "expected_xy_mm": {"x": args.expected_x_mm, "y": args.expected_y_mm},
        "measured_xy_mm": {"x": round(measured_x, 3), "y": round(measured_y, 3)},
        "error_mm": round(error_mm, 3),
        "tolerance_mm": args.tolerance_mm,
    })
    return 0 if verified else 3


if __name__ == "__main__":
    raise SystemExit(main())