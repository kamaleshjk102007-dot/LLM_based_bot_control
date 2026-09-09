# Vision System Documentation — Member 3

## 1. Overview
The `vision` package provides hardware-agnostic image acquisition, color & contour object detection, and visualization overlays for the DOBOT Magician Lite tabletop environment.

## 2. Camera Sources (`vision/camera.py`)
All cameras implement the `BaseCamera` interface (`open()`, `read()`, `release()`, `is_opened()`):

* **`MockCamera`**: Generates synthetic tabletop frames (640x480) with configurable colored blocks and optional coordinate jitter for offline CI testing.
* **`USBCamera`**: OpenCV-based capture for physical USB webcams and laptop cameras (`device_index=0`).
* **`WebotsCamera`**: Adapter for Webots camera devices (`wb_camera_get_image`), converting BGRA simulation streams to standard BGR frames.
* **`FileCamera`**: Plays recorded video files or static test images.

## 3. Object Detector (`vision/detector.py`)
* **`ColorShapeDetector`**: Tabletop block detector using OpenCV HSV color thresholding, morphological noise filtering (`cv2.MORPH_OPEN`, `cv2.MORPH_CLOSE`), contour solidity filtering ($\ge 0.6$), and spatial moments for sub-pixel centroid calculation.
* **`YOLODetector`**: Pluggable deep learning wrapper ready to load Ultralytics YOLO models (`.pt` or ONNX) for general object detection.

### Supported Block Classes & HSV Thresholds
* `red_block`: Wrap-around ranges: `[0, 100, 80]` to `[10, 255, 255]` OR `[170, 100, 80]` to `[180, 255, 255]`.
* `blue_block`: `[100, 100, 60]` to `[135, 255, 255]`.
* `green_block`: `[35, 70, 60]` to `[85, 255, 255]`.
* `yellow_block`: `[20, 100, 100]` to `[35, 255, 255]`.

## 4. Visualizer HUD (`vision/visualization.py`)
Renders real-time debug graphics on camera frames:
* Bounding boxes with class labels and confidence percentages.
* Centroid dots and crosshairs.
* Top HUD status banner:
  * 🟢 **VALID**: Displays physical coordinates in `dobot_base` frame (mm).
  * 🟠 **AMBIGUOUS**: Flags multi-object conflicts.
  * 🟡 **UNSTABLE**: Flags position jitter.
  * 🔴 **NOT_FOUND / LOW_CONFIDENCE / OUT_OF_REACH**: Explains failure.
