"""
Member 3 -- Live Webcam Test
Uses your laptop webcam to detect colored blocks in real time.
Press Q to quit, S to save a snapshot.

HOW TO TEST:
- Hold a RED object (red paper, red pen cap, red tape) in front of camera
- Hold a BLUE object (blue paper, blue pen)
- Hold a GREEN object (green paper, green bottle)
- Watch the detection happen live!
"""

import sys
import os
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.camera import USBCamera
from vision.detector import ColorShapeDetector
from vision.models import Frame
from vision.visualization import VisionVisualizer
from vision.calibration import TabletopCalibration
from vision.coordinate_transform import CoordinateTransformer
from vision.target import TargetSelector, TemporalStabilityTracker, create_invalid_robot_target
from vision.models import TargetStatus, RobotTarget
import uuid
from datetime import datetime, timezone

SEPARATOR = "=" * 60

def print_target_console(target):
    if target.valid:
        print(f"  [OK] VALID  | Class: {target.class_name:12s} | "
              f"X={target.position.x:+7.1f}mm  Y={target.position.y:+7.1f}mm  Z={target.position.z:+6.1f}mm | "
              f"Conf={target.confidence*100:.0f}%  Stab={target.stability_score*100:.0f}%")
    else:
        print(f"  [--] {target.status.value:12s} | {target.message}")


def main():
    print(SEPARATOR)
    print("  MEMBER 3 -- LIVE WEBCAM TEST")
    print(SEPARATOR)
    print()
    print("  What to do:")
    print("  - Hold a RED / BLUE / GREEN / YELLOW object in front of camera")
    print("  - Keep object still for 3 frames to see VALID status")
    print()
    print("  Keyboard shortcuts:")
    print("  [Q]       - Quit")
    print("  [S]       - Save snapshot to output/webcam_snapshot.png")
    print("  [1]       - Look for red_block")
    print("  [2]       - Look for blue_block")
    print("  [3]       - Look for green_block")
    print("  [4]       - Look for yellow_block")
    print("  [R]       - Reset stability tracker")
    print()

    # --- Setup ---
    camera = USBCamera(device_index=0, width=640, height=480)
    opened = camera.open()

    if not opened:
        print("  [ERROR] Could not open webcam (device index 0).")
        print("  Try changing device_index to 1 or 2 if you have multiple cameras.")
        return

    print("  [OK] Webcam opened successfully!")
    print()

    detector  = ColorShapeDetector()
    visualizer = VisionVisualizer()

    # Load calibration
    calib = TabletopCalibration.create_default()
    transformer = CoordinateTransformer(calibration=calib, table_z_mm=-50.0)
    selector  = TargetSelector(min_confidence=0.60)
    tracker   = TemporalStabilityTracker(window_size=4, min_samples=3, max_std_dev_mm=5.0)

    os.makedirs("output", exist_ok=True)

    requested_class = "red_block"
    last_target = None
    frame_count = 0
    fps_time = time.time()

    print(f"  Currently looking for: {requested_class}  (press 1/2/3/4 to change)")
    print(SEPARATOR)

    while True:
        frame = camera.read()
        if frame is None:
            print("  [!!] Failed to read frame from webcam.")
            break

        frame_count += 1

        # --- Detection ---
        detections = detector.detect(frame)

        # --- Target selection pipeline ---
        candidate, sel_status, sel_msg = selector.select(detections, requested_class)

        if sel_status == TargetStatus.VALID and candidate is not None:
            point_3d, is_reachable, reach_msg = transformer.pixel_to_robot_3d(
                candidate.center, requested_class
            )

            if is_reachable:
                filtered_pt, stab_status, stab_score, stab_msg = tracker.update(
                    requested_class, point_3d
                )

                if stab_status == TargetStatus.VALID:
                    last_target = RobotTarget(
                        target_id=f"tgt_{uuid.uuid4().hex[:8]}",
                        class_name=requested_class,
                        position=filtered_pt,
                        confidence=candidate.confidence,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        coordinate_frame="dobot_base",
                        valid=True,
                        status=TargetStatus.VALID,
                        stability_score=stab_score,
                        source_detection_id=candidate.detection_id,
                        message=f"Valid at X:{filtered_pt.x:.1f} Y:{filtered_pt.y:.1f} Z:{filtered_pt.z:.1f} mm",
                    )
                else:
                    last_target = create_invalid_robot_target(
                        requested_class, TargetStatus.UNSTABLE, stab_msg,
                        position=filtered_pt, confidence=candidate.confidence,
                        source_detection_id=candidate.detection_id,
                    )
            else:
                last_target = create_invalid_robot_target(
                    requested_class, TargetStatus.OUT_OF_REACH,
                    reach_msg or "Out of reach", confidence=candidate.confidence,
                )
        else:
            last_target = create_invalid_robot_target(
                requested_class, sel_status, sel_msg
            )

        # --- Draw HUD overlay ---
        annotated = visualizer.draw_overlay(frame, detections, last_target)

        # --- Draw extra info bar at bottom ---
        h, w = annotated.shape[:2]
        fps = frame_count / max(time.time() - fps_time, 0.001)
        info_text = (f"Looking for: {requested_class}   |   "
                     f"Detections: {len(detections)}   |   "
                     f"FPS: {fps:.1f}   |   "
                     f"Press Q=Quit  S=Save  1/2/3/4=Change target  R=Reset")
        cv2.rectangle(annotated, (0, h - 24), (w, h), (40, 40, 40), -1)
        cv2.putText(annotated, info_text, (6, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # --- Show window ---
        cv2.imshow("Member 3 -- Live Webcam Detection (Press Q to quit)", annotated)

        # Print to console every 10 frames
        if frame_count % 10 == 0:
            print_target_console(last_target)

        # --- Keyboard ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:
            print("\n  [Q] Quitting...")
            break

        elif key == ord('s') or key == ord('S'):
            path = "output/webcam_snapshot.png"
            cv2.imwrite(path, annotated)
            print(f"  [S] Snapshot saved to {path}")

        elif key == ord('1'):
            requested_class = "red_block"
            tracker.reset()
            print(f"  [1] Now looking for: red_block")

        elif key == ord('2'):
            requested_class = "blue_block"
            tracker.reset()
            print(f"  [2] Now looking for: blue_block")

        elif key == ord('3'):
            requested_class = "green_block"
            tracker.reset()
            print(f"  [3] Now looking for: green_block")

        elif key == ord('4'):
            requested_class = "yellow_block"
            tracker.reset()
            print(f"  [4] Now looking for: yellow_block")

        elif key == ord('r') or key == ord('R'):
            tracker.reset()
            print("  [R] Stability tracker reset.")

    camera.release()
    cv2.destroyAllWindows()
    print("  Done. Webcam released.")


if __name__ == "__main__":
    main()
