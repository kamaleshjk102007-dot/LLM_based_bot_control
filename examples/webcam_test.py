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
import json
import time

# Suppress OpenCV DSHOW backend warning spam on Windows
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)
elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

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

# --- Global mouse position for HSV debug overlay ---
_mouse_x: int = 0
_mouse_y: int = 0

def _on_mouse(event, x, y, flags, param):
    global _mouse_x, _mouse_y
    _mouse_x, _mouse_y = x, y

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
    print("  [S]       - Save snapshot + save blocks to config/detected_blocks.json")
    print("  [T]       - Toggle HSV debug mode (hover over cube to see H/S/V values)")
    print("  [SPACE]   - Send current VALID target X,Y,Z to Member 2 gateway")
    print("  [1/2/3/4] - Switch color (1: Red, 2: Blue, 3: Green, 4: Yellow)")
    print("  [<- / ->] - Arrow Keys: switch target between cubes (#1, #2, #3, #4)")
    print("  [A / D]   - Also switches between cubes (A: Left, D: Right)")
    print("  [R]       - Reset stability tracker")
    print()
    print("  IMPORTANT: Click ON the camera video popup window to use Arrow Keys and shortcuts!")
    print()

    # --- Camera Selection ---
    cam_idx = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_idx = int(sys.argv[1])
    else:
        # Check if camera 1 is available (using DirectShow on Windows)
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        test_cap = cv2.VideoCapture(1, backend)
        if test_cap.isOpened():
            test_cap.release()
            print("  Detected external / DOBOT camera at index 1!")
            cam_idx = 1
        else:
            cam_idx = 0

    print(f"  Connecting to camera index: {cam_idx}")
    camera = USBCamera(device_index=cam_idx, width=640, height=480)
    opened = camera.open()

    if not opened:
        print(f"  [ERROR] Could not open camera (device index {cam_idx}).")
        print("  Try running with explicit index, e.g.: py examples\\webcam_test.py 0")
        return

    print(f"  [OK] Camera index {cam_idx} opened successfully!")
    print()

    detector  = ColorShapeDetector()
    visualizer = VisionVisualizer()

    # Load calibration
    calib = TabletopCalibration.create_default()
    transformer = CoordinateTransformer(calibration=calib, table_z_mm=-50.0)
    selector  = TargetSelector(min_confidence=0.60, disambiguation_strategy="leftmost")
    tracker   = TemporalStabilityTracker(window_size=4, min_samples=3, max_std_dev_mm=5.0)

    os.makedirs("output", exist_ok=True)

    requested_class = "red_block"
    last_target = None
    frame_count = 0
    fps_time = time.time()
    hsv_debug_mode = False     # toggled by T key
    auto_broadcast = True      # automatically streams detected blocks to Member 2!
    last_broadcast_time = 0.0
    send_count = 0

    print(f"  Currently looking for: {requested_class}  (press 1/2/3/4 to change)")
    print(SEPARATOR)

    try:
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
                point_3d, is_reachable, reach_msg, z_mode = transformer.pixel_to_robot_3d(
                    pixel=candidate.center,
                    class_name=requested_class,
                    bbox=candidate.bbox,
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

            # --- Draw cube index numbers when multiple matching cubes exist ---
            class_dets = [d for d in detections if d.class_name == requested_class and d.confidence >= selector.min_confidence]
            if len(class_dets) > 1:
                sorted_dets = sorted(class_dets, key=lambda d: d.center.x)
                active_idx = selector.target_index % len(sorted_dets)
                for i, d in enumerate(sorted_dets):
                    bx, by = d.bbox.x1, d.bbox.y1
                    is_active = (i == active_idx)
                    tag_label = f"#{i+1} [ACTIVE]" if is_active else f"#{i+1}"
                    tag_color = (0, 255, 0) if is_active else (50, 180, 255)
                    # Draw index badge above bounding box
                    cv2.rectangle(annotated, (bx, max(0, by - 40)), (bx + 85, max(0, by - 20)), (20, 20, 20), -1)
                    cv2.rectangle(annotated, (bx, max(0, by - 40)), (bx + 85, max(0, by - 20)), tag_color, 2)
                    cv2.putText(annotated, tag_label, (bx + 4, max(12, by - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, tag_color, 1, cv2.LINE_AA)

            # --- Auto-stream detected blocks to Member 2 (config/detected_blocks.json) ---
            now = time.time()
            if auto_broadcast and (now - last_broadcast_time > 0.4):
                last_broadcast_time = now
                if detections:
                    live_blocks = []
                    for i, det in enumerate(detections):
                        pt3d, _, _, _ = transformer.pixel_to_robot_3d(
                            pixel=det.center, class_name=det.class_name, bbox=det.bbox
                        )
                        col = det.class_name.split("_")[0]
                        live_blocks.append({
                            "id": f"{col}_block_{i+1}",
                            "color": col,
                            "type": "block",
                            "x": round(float(pt3d.x), 1),
                            "y": round(float(pt3d.y), 1),
                            "z": round(float(pt3d.z), 1),
                            "r": 0.0,
                        })
                    try:
                        os.makedirs("config", exist_ok=True)
                        with open("config/detected_blocks.json", "w", encoding="utf-8") as f:
                            json.dump({"blocks": live_blocks}, f, indent=2)
                    except Exception:
                        pass

            # --- Draw extra info bar at bottom ---
            h, w = annotated.shape[:2]
            fps = frame_count / max(time.time() - fps_time, 0.001)
            total_cubes = len(class_dets)
            curr_cube_num = (selector.target_index % total_cubes) + 1 if total_cubes > 1 else 1
            cube_info = f"Cube [#{curr_cube_num}/{total_cubes}]" if total_cubes > 1 else "1 cube"
            sync_tag = "[M2: LIVE SYNC]" if auto_broadcast else "[M2: MANUAL]"
            info_text = (f"Target: {requested_class} | {cube_info} | {sync_tag} | "
                         f"[SPACE]=Send  [T]=HSV  [B]=Sync  [1-4]=Color  [Q]=Quit")
            cv2.rectangle(annotated, (0, h - 26), (w, h), (30, 30, 30), -1)
            cv2.putText(annotated, info_text, (6, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 255, 220), 1, cv2.LINE_AA)

            # --- HSV Debug Overlay (T key) ---
            if hsv_debug_mode:
                hsv_frame = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
                mx = min(max(_mouse_x, 0), w - 1)
                my = min(max(_mouse_y, 0), h - 1)
                hv, sv, vv = hsv_frame[my, mx]
                # Crosshair at mouse position
                cv2.line(annotated, (mx - 10, my), (mx + 10, my), (0, 255, 255), 1)
                cv2.line(annotated, (mx, my - 10), (mx, my + 10), (0, 255, 255), 1)
                # HSV info box
                dbg_txt = f"H={hv}  S={sv}  V={vv}  (pixel {mx},{my})"
                cv2.rectangle(annotated, (0, 0), (310, 22), (0, 0, 0), -1)
                cv2.putText(annotated, dbg_txt, (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                # HSV mode label top-right
                cv2.rectangle(annotated, (w - 130, 0), (w, 22), (0, 160, 0), -1)
                cv2.putText(annotated, "[T] HSV DEBUG ON", (w - 127, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # --- Show window and bind mouse for HSV debug ---
            win_name = "Member 3 -- Live Webcam Detection (SPACE=Send T=HSV Q=Quit)"
            cv2.imshow(win_name, annotated)
            cv2.setMouseCallback(win_name, _on_mouse)

            # Print to console every 10 frames
            if frame_count % 10 == 0:
                print_target_console(last_target)

            # --- Keyboard (using waitKeyEx to support Windows arrow keys) ---
            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n  [Q] Quitting...")
                break

            elif key == ord(' '):  # SPACE -- send current valid target to Member 2
                if last_target is not None and last_target.valid:
                    pos = last_target.position
                    color = last_target.class_name.split("_")[0]
                    blocks_data = {
                        "blocks": [{
                            "id": f"{color}_block_1",
                            "color": color,
                            "type": "block",
                            "x": round(float(pos.x), 1),
                            "y": round(float(pos.y), 1),
                            "z": round(float(pos.z), 1),
                            "r": 0.0,
                        }]
                    }
                    os.makedirs("config", exist_ok=True)
                    with open("config/detected_blocks.json", "w", encoding="utf-8") as f:
                        json.dump(blocks_data, f, indent=2)
                    send_count += 1
                    print(f"\n  [SPACE] Sent #{send_count}: {last_target.class_name} --> "
                          f"X={pos.x:+.1f}mm  Y={pos.y:+.1f}mm  Z={pos.z:+.1f}mm")
                    print(f"         Saved to config/detected_blocks.json (Member 2 can read this now!)")
                else:
                    status = last_target.status.value if last_target else "NONE"
                    print(f"\n  [SPACE] Cannot send -- target not VALID yet. Status: {status}")

            elif key == ord('b') or key == ord('B'):  # B -- toggle auto sync
                auto_broadcast = not auto_broadcast
                state = "ENABLED" if auto_broadcast else "DISABLED"
                print(f"  [B] Member 2 Live Auto-Sync: {state}")

            elif key == ord('t') or key == ord('T'):  # T -- toggle HSV debug
                hsv_debug_mode = not hsv_debug_mode
                state = "ON" if hsv_debug_mode else "OFF"
                print(f"  [T] HSV debug mode {state}. Hover over a cube in the video window to see H/S/V values.")
                if hsv_debug_mode:
                    print("      Use those values to tune detector.py DEFAULT_COLOR_RANGES!")

            elif key == ord('s') or key == ord('S'):
                path = "output/webcam_snapshot.png"
                cv2.imwrite(path, annotated)
                print(f"  [S] Snapshot saved to {path}")
                # Also save detected blocks for Member 2
                blocks = []
                for i, det in enumerate(detections):
                    pt3d, reachable, _, _ = transformer.pixel_to_robot_3d(
                        pixel=det.center, class_name=det.class_name, bbox=det.bbox
                    )
                    col = det.class_name.split("_")[0]
                    blocks.append({
                        "id": f"{col}_block_{i+1}",
                        "color": col,
                        "type": "block",
                        "x": round(float(pt3d.x), 1),
                        "y": round(float(pt3d.y), 1),
                        "z": round(float(pt3d.z), 1),
                        "r": 0.0,
                    })
                os.makedirs("config", exist_ok=True)
                with open("config/detected_blocks.json", "w", encoding="utf-8") as f:
                    json.dump({"blocks": blocks}, f, indent=2)
                print(f"  [S] Saved {len(blocks)} block(s) to config/detected_blocks.json for Member 2")

            elif key == ord('1'):
                requested_class = "red_block"
                selector.target_index = 0
                tracker.reset()
                print(f"  [1] Now looking for: red_block")

            elif key == ord('2'):
                requested_class = "blue_block"
                selector.target_index = 0
                tracker.reset()
                print(f"  [2] Now looking for: blue_block")

            elif key == ord('3'):
                requested_class = "green_block"
                selector.target_index = 0
                tracker.reset()
                print(f"  [3] Now looking for: green_block")

            elif key == ord('4'):
                requested_class = "yellow_block"
                selector.target_index = 0
                tracker.reset()
                print(f"  [4] Now looking for: yellow_block")

            # --- Right Arrow / Up Arrow / 'D' / '>' / Tab : Next cube ---
            elif key_ex in (2555904, 2490368, 65363, 65362, 39, 38) or key in (ord('d'), ord('D'), ord('.'), ord('>'), 9):
                if len(class_dets) > 1:
                    selector.target_index = (selector.target_index + 1) % len(class_dets)
                    tracker.reset()
                    print(f"  [->] Switched to cube #{selector.target_index + 1} of {len(class_dets)} (left-to-right)")

            # --- Left Arrow / Down Arrow / 'A' / '<' : Previous cube ---
            elif key_ex in (2424832, 2621440, 65361, 65364, 37, 40) or key in (ord('a'), ord('A'), ord(','), ord('<')):
                if len(class_dets) > 1:
                    selector.target_index = (selector.target_index - 1) % len(class_dets)
                    tracker.reset()
                    print(f"  [<-] Switched to cube #{selector.target_index + 1} of {len(class_dets)} (left-to-right)")

            elif key == ord('r') or key == ord('R'):
                tracker.reset()
                print("  [R] Stability tracker reset.")

    except KeyboardInterrupt:
        print("\n  [INFO] Stopped by user (Ctrl+C). Exiting cleanly.")
    finally:
        try:
            camera.release()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            cv2.destroyAllWindows()
        except (Exception, KeyboardInterrupt):
            pass
        print("  Done. Webcam released.")


if __name__ == "__main__":
    main()
