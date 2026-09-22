"""Exercise the Member 3 to Member 4 contract from a real camera.

The script detects one stable object, converts its vision RobotTarget to the
canonical motion target, and creates a MotionPlan. It intentionally imports no
gateway, adapter, Webots, or robot-execution code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion.models import RobotPosition
from motion.planner import MotionPlanner, MotionPlanningPolicy
from vision.calibration import TabletopCalibration
from vision.camera import USBCamera
from vision.coordinate_transform import CoordinateTransformer
from vision.pipeline import VisionPipeline
from vision.target import MotionTargetConversionError, to_motion_target


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-camera Member 3 to Member 4 dry run; never executes motion.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--directshow", action="store_true")
    parser.add_argument("--class-name", default="red_block")
    parser.add_argument("--target-z-mm", type=float, default=300.0)
    parser.add_argument("--start-x-mm", type=float, default=230.0)
    parser.add_argument("--start-y-mm", type=float, default=0.0)
    parser.add_argument("--start-z-mm", type=float, default=300.0)
    args = parser.parse_args()

    calibration = TabletopCalibration()
    calibration.load(str(args.calibration))
    camera = USBCamera(
        device_index=args.camera_index,
        backend=cv2.CAP_DSHOW if args.directshow else None,
    )
    pipeline = VisionPipeline(
        camera=camera,
        calibration=calibration,
        transformer=CoordinateTransformer(
            calibration=calibration,
            table_z_mm=args.target_z_mm,
            object_heights={args.class_name: 0.0, "default": 0.0},
        ),
    )

    target = None
    for _ in range(6):
        target, _ = pipeline.capture_and_process(args.class_name)
    camera.release()
    assert target is not None
    if not target.valid:
        print({"member3_status": target.status.value, "message": target.message, "execution": "NOT_REQUESTED"})
        return 2

    try:
        motion_target = to_motion_target(target)
    except MotionTargetConversionError as exc:
        print({"member3_status": "CONVERSION_FAILED", "message": str(exc), "execution": "NOT_REQUESTED"})
        return 2

    planner = MotionPlanner(MotionPlanningPolicy(
        max_translation_step_mm=5.0,
        max_rotation_step_degrees=5.0,
        coordinate_frame="dobot_base",
    ))
    plan = planner.plan(
        RobotPosition(x=args.start_x_mm, y=args.start_y_mm, z=args.start_z_mm),
        motion_target,
        robot_id="webots_001",
    )
    print({
        "member3_target": target.model_dump(mode="json"),
        "member4_motion_plan": plan.model_dump(mode="json"),
        "calibration_state": "UNVERIFIED_DRY_RUN_ONLY",
        "execution": "NOT_REQUESTED",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())