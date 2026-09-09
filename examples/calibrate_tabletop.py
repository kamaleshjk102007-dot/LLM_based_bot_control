"""
Tabletop Calibration Utility for Member 3.
Allows recording (pixel_u, pixel_v) <-> (robot_x, robot_y) correspondences,
calculating the homography matrix, validating accuracy on hold-out points,
and saving the resulting profile to config/calibration.json.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.calibration import TabletopCalibration


def calibrate_and_validate(
    training_points: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    validation_points: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    output_path: str = "config/calibration.json",
):
    print("=" * 60)
    print("TABLETOP HOMOGRAPHY CALIBRATION")
    print("=" * 60)
    print(f"Number of training point pairs:   {len(training_points)}")
    print(f"Number of validation point pairs: {len(validation_points)}")

    calib = TabletopCalibration()
    success = calib.calibrate(training_points)
    if not success:
        print("ERROR: Calibration failed to converge.")
        return False

    print("\nCalibration Matrix (H):")
    print(calib.homography)

    if validation_points:
        metrics = calib.validate(validation_points)
        print("\nValidation Results on Held-Out Points:")
        print(f"  Mean Euclidean Error: {metrics.mean_euclidean_error_mm:.3f} mm")
        print(f"  Max Euclidean Error:  {metrics.max_euclidean_error_mm:.3f} mm")
        print(f"  Mean X Error:         {metrics.mean_x_error_mm:.3f} mm")
        print(f"  Mean Y Error:         {metrics.mean_y_error_mm:.3f} mm")

        if metrics.mean_euclidean_error_mm > 5.0:
            print("WARNING: Mean error > 5mm! Consider re-collecting calibration points.")
        else:
            print("SUCCESS: Calibration accuracy within high-precision tolerances (< 5mm).")

    calib.save(output_path)
    print(f"\nCalibration saved to: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate tabletop homography")
    parser.add_argument("--output", default="config/calibration.json", help="Path to output json")
    args = parser.parse_args()

    # Standard representative tabletop grid points
    # Format: ((pixel_u, pixel_v), (robot_x_mm, robot_y_mm))
    train_pts = [
        ((120.0, 100.0), (300.0, -120.0)),
        ((520.0, 100.0), (300.0, 120.0)),
        ((520.0, 400.0), (180.0, 120.0)),
        ((120.0, 400.0), (180.0, -120.0)),
    ]

    val_pts = [
        ((320.0, 250.0), (240.0, 0.0)),
    ]

    calibrate_and_validate(train_pts, val_pts, args.output)
