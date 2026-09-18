"""
Member 3 -- Interactive HSV Color Tuner
Use this to find the exact HSV values for any object under your room's lighting.

HOW TO USE:
1. Run: py examples\\tune_color.py
2. Hold your colored object in front of the webcam
3. Adjust the 6 trackbars until ONLY your object is white, and the background is black
4. Press 'P' to print the exact Python code for vision/detector.py
5. Press 'Q' to quit
"""

import sys
import os
import cv2
import numpy as np

def nothing(x):
    pass

def main():
    print("=" * 60)
    print("  MEMBER 3 -- INTERACTIVE HSV COLOR TUNER")
    print("=" * 60)
    print("  Adjust the trackbars to isolate your object's color.")
    print("  Keyboard shortcuts:")
    print("  [P] - Print the Python code for this color")
    print("  [Q] - Quit")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open webcam (index 0).")
        return

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner", 700, 320)

    # Initial values (defaults set for generic vibrant colors)
    cv2.createTrackbar("H Min", "HSV Tuner", 0, 180, nothing)
    cv2.createTrackbar("H Max", "HSV Tuner", 180, 180, nothing)
    cv2.createTrackbar("S Min", "HSV Tuner", 70, 255, nothing)
    cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Tuner", 60, 255, nothing)
    cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, nothing)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for display
        frame = cv2.resize(frame, (480, 360))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get trackbar positions
        h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        # Filter
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Highlight detected object on original frame
        masked_view = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert 1-channel mask to 3-channel for side-by-side display
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Put label text on images
        cv2.putText(frame, "Webcam Live", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(mask_3ch, f"H:[{h_min},{h_max}] S:[{s_min},{s_max}] V:[{v_min},{v_max}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Stack side by side
        combined = np.hstack((frame, mask_3ch))
        cv2.imshow("Detection Preview (Live vs Mask)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('p') or key == ord('P'):
            print("\n  [COPY-PASTE THIS INTO vision/detector.py]:")
            print(f'  "custom_block": [\n      (({h_min}, {s_min}, {v_min}), ({h_max}, {s_max}, {v_max})),\n  ],')
            print("-" * 50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
