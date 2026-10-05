"""
Member 3 -- DOBOT Physical Verification Test
============================================
Use this to verify that Member 3's coordinates match the real DOBOT workspace.

HOW TO USE:
1. Connect your camera (laptop webcam or DOBOT camera)
2. Place a colored block on the DOBOT table
3. Run this script: py examples\dobot_verify.py
4. Note the X, Y, Z values shown
5. Open DobotLab, manually jog arm to those X, Y, Z values
6. Check if the arm tip points at the block!

If it does -- Member 3 is VERIFIED and complete!
If not -- run calibrate_tabletop.py to fix the calibration.0
"""

import sys
import os
import time

# Suppress OpenCV DSHOW backend warning spam on Windows
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)
elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.camera import USBCamera
from vision.pipeline import VisionPipeline

SEPARATOR = "=" * 55

def main():
    print(SEPARATOR)
    print("  MEMBER 3 -- DOBOT PHYSICAL VERIFICATION TEST")
    print(SEPARATOR)
    print()
    print("  This script detects a colored block and shows")
    print("  the EXACT X, Y, Z coordinates to enter in DobotLab.")
    print()

    # --- Ask which camera ---
    print("  Which camera do you want to use?")
    print("  [0] Laptop webcam (default)")
    print("  [1] DOBOT camera or external USB camera")
    print("  [2] Other (enter manually)")
    cam_input = input("  Enter camera index (default=0): ").strip()
    cam_idx = int(cam_input) if cam_input.isdigit() else 0

    # --- Ask which block to detect ---
    print()
    print("  Which block color is on the table?")
    print("  [1] red_block")
    print("  [2] blue_block")
    print("  [3] green_block")
    print("  [4] yellow_block")
    color_map = {"1": "red_block", "2": "blue_block", "3": "green_block", "4": "yellow_block"}
    color_input = input("  Choose (1/2/3/4): ").strip()
    target_class = color_map.get(color_input, "red_block")

    print()
    print(f"  Using camera index : {cam_idx}")
    print(f"  Looking for        : {target_class}")
    print()

    # --- Open camera ---
    cam = USBCamera(device_index=cam_idx, width=640, height=480)
    if not cam.open():
        print(f"  [ERROR] Cannot open camera at index {cam_idx}.")
        print("  Try a different index.")
        return

    print(f"  [OK] Camera opened. Detecting {target_class}...")
    print("  Hold the block steady in front of the camera.")
    print("  Press Q in the camera window to quit.")
    print(SEPARATOR)

    pipeline = VisionPipeline(camera=cam, stability_samples=3)
    last_valid = None

    try:
        while True:
            target, hud = pipeline.capture_and_process(target_class, visualize=True)

            # Show camera window
            if hud is not None:
                # Draw instruction text on frame
                h, w = hud.shape[:2]
                cv2.rectangle(hud, (0, h - 28), (w, h), (30, 30, 30), -1)
                cv2.putText(hud, f"Looking for: {target_class}  |  Press Q to quit",
                            (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.imshow("Member 3 -- DOBOT Verification (Press Q to quit)", hud)

            # Print only when status changes or valid
            if target.valid:
                if last_valid is None or abs(target.position.x - last_valid.position.x) > 1.0:
                    print()
                    print("  +-----------------------------------------------+")
                    print("  |   OBJECT DETECTED -- COPY THESE TO DOBOTLAB   |")
                    print("  +-----------------------------------------------+")
                    print(f"  |  Class      : {target_class:<31}|")
                    print(f"  |  X (mm)     : {target.position.x:>+10.2f}                       |")
                    print(f"  |  Y (mm)     : {target.position.y:>+10.2f}                       |")
                    print(f"  |  Z (mm)     : {target.position.z:>+10.2f}                       |")
                    print(f"  |  Confidence : {target.confidence * 100:>10.1f}%                      |")
                    print(f"  |  Stability  : {target.stability_score * 100:>10.1f}%                      |")
                    print("  +-----------------------------------------------+")
                    print()
                    print("  --> Open DobotLab --> Jog to these X, Y, Z values")
                    print("  --> Does the arm tip point at the block?")
                    print("      YES = Member 3 VERIFIED! Work complete!")
                    print("      NO  = Run: py examples\\calibrate_tabletop.py")
                    print()
                    last_valid = target
            else:
                if target.status.value in ("NOT_FOUND", "AMBIGUOUS"):
                    print(f"  [{target.status.value}] {target.message}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
    except KeyboardInterrupt:
        print("\n  [INFO] Stopped by user (Ctrl+C).")
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if last_valid:
        print(SEPARATOR)
        print("  FINAL COORDINATES FOR DOBOTLAB:")
        print(f"  X = {last_valid.position.x:+.2f} mm")
        print(f"  Y = {last_valid.position.y:+.2f} mm")
        print(f"  Z = {last_valid.position.z:+.2f} mm")
        print(SEPARATOR)
    else:
        print("  No valid target was detected. Check camera and block placement.")


if __name__ == "__main__":
    main()
