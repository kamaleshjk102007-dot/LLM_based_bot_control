# 👁️ Member 3 — Vision + Sensors + Perception System
**Robot 1 — DOBOT Magician Lite**

Member 3 owns the vision and perception pipeline. The primary mission is to convert raw camera imagery into a verified, stable, and calibrated **`RobotTarget`** in the `dobot_base` coordinate frame for **Member 4 (Motion Planning)**.

---

## 🎯 Main Responsibility

> **Camera / Sensor → Detect Object → Select Target → Homography Calibration → Coordinate Transform → Temporal Stability & Freshness Check → Send RobotTarget to Member 4**

### ❌ Member 3 Boundaries (What Member 3 must NOT do)
* **Never control the robot arm**: Do not call `dobot.move_to()` or send motor commands. Member 1 and Member 4 own robot kinematics and motor execution.
* **Never send raw pixel coordinates**: Member 4 must receive metric physical coordinates $(X, Y, Z)$ in millimeters, strictly in the `dobot_base` frame.
* **Never guess between multiple ambiguous objects**: If multiple matching objects are found, return `AMBIGUOUS`.
* **Never fake physical $Z$**: A 2D camera cannot measure metric height directly; $Z$ is computed deterministically from the calibrated tabletop plane $Z_{table}$ and known physical object heights.

---

## 🏗️ Architecture Pipeline

```text
                 USER TASK (Member 2)
                           │
                           ▼
                    Requested Object
                           │
                           ▼
                     CAMERA SOURCE
          (MockCamera / USBCamera / WebotsCamera)
                           │
                           ▼
                         FRAME
                           │
                           ▼
                    OBJECT DETECTOR
             (ColorShapeDetector / YOLODetector)
                           │
                           ▼
                       DETECTIONS
                           │
                           ▼
                    TARGET SELECTOR
           ┌───────────────┴───────────────┐
           │                               │
        Unique                         Ambiguous
           │                               │
           ▼                               ▼
      CALIBRATION                      [REJECT]
   (Planar Homography)            Status: AMBIGUOUS
           │
           ▼
  COORDINATE TRANSFORM
   (dobot_base frame)
  Z = Z_table + h_object
           │
           ▼
  CONFIDENCE & STABILITY
(Sliding window variance)
           │
           ▼
      ROBOT TARGET
    (Status: VALID)
           │
           ▼
   MEMBER 4 CONTRACT
```

---

## 📦 Project Structure

```text
member 3/
├── config/
│   ├── calibration_default.json   # Default tabletop homography matrix
│   └── calibration.json           # User-calibrated matrix
├── vision/
│   ├── __init__.py                # Package exports
│   ├── models.py                  # Pydantic schemas (Point2D, Point3D, BoundingBox, Frame, RobotTarget)
│   ├── camera.py                  # BaseCamera, USBCamera, MockCamera, FileCamera, WebotsCamera
│   ├── detector.py                # BaseDetector, ColorShapeDetector, YOLODetector
│   ├── calibration.py             # TabletopCalibration & ValidationMetrics
│   ├── coordinate_transform.py    # CoordinateTransformer (dobot_base frame & reachability check)
│   ├── target.py                  # TargetSelector, TemporalStabilityTracker, TargetFreshnessChecker
│   ├── visualization.py           # VisionVisualizer (HUD overlay & debug graphics)
│   └── pipeline.py                # VisionPipeline (Full end-to-end orchestrator)
├── tests/
│   ├── test_models.py             # Data model immutability and schema checks
│   ├── test_camera.py             # Camera open/read/release lifecycle
│   ├── test_detector.py           # Color detector accuracy, noise filtering
│   ├── test_calibration.py        # Homography calculation and hold-out validation
│   ├── test_coordinate_transform.py # Millimeter mapping and workspace reachability
│   ├── test_target.py             # Target selection, ambiguity, and freshness
│   ├── test_detection_stability.py  # Temporal variance filter and jump rejection
│   └── test_pipeline.py           # Full integration test
├── examples/
│   ├── run_demo.py                # Standalone simulation demo with HUD output
│   └── calibrate_tabletop.py      # Calibration script with hold-out error report
├── requirements.txt
└── README.md
```

---

## 🤝 Contract with Member 4 (`RobotTarget`)

Downstream Member 4 receives a standard JSON payload:

```json
{
  "target_id": "tgt_b576a545",
  "class_name": "red_block",
  "position": {
    "x": 239.8,
    "y": 0.0,
    "z": -25.0
  },
  "confidence": 0.98,
  "timestamp": "2026-09-09T05:18:23.177784+00:00",
  "coordinate_frame": "dobot_base",
  "valid": true,
  "status": "VALID",
  "stability_score": 0.87,
  "source_detection_id": "det_5154ae60",
  "message": "Valid target verified at X:239.8, Y:0.0, Z:-25.0 mm"
}
```

### Possible Target Statuses
| Status | Meaning | Valid | Member 4 Action |
|---|---|---|---|
| `VALID` | Object uniquely identified, calibrated, and temporally stable | `true` | Proceed with motion planning |
| `AMBIGUOUS` | 2+ matching objects seen without disambiguation rule | `false` | Request clarification from Member 2 |
| `NOT_FOUND` | Target class is not visible in frame | `false` | Halt or initiate camera search sweep |
| `LOW_CONFIDENCE` | Candidate seen but confidence score is below threshold | `false` | Reject |
| `UNSTABLE` | Position jumping across frames (jitter) | `false` | Wait for tracking to settle |
| `OUT_OF_REACH` | Coordinates are outside Dobot Magician Lite reach envelope | `false` | Abort motion |
| `STALE` | Target generated too long ago ($> 1.5$s) | `false` | Request fresh frame |

---

## 🚀 Quickstart & Verification

### 1. Run Unit Tests (100% Passing)
```bash
python -m pytest tests/ -v
```

### 2. Run Live Simulation Demo
```bash
python examples/run_demo.py
```
This will:
- Simulate camera frames with colored blocks.
- Perform multi-frame temporal stabilization.
- Output the verified `RobotTarget` JSON to `output/member4_target.json`.
- Test ambiguity rejection when two red blocks appear.
- Save debug HUD visualizations to `output/vision_hud_demo.png` and `output/vision_hud_ambiguous.png`.

### 3. Calibrate Physical Camera / Webots
```bash
python examples/calibrate_tabletop.py --output config/calibration.json
```
Calculates homography matrix from $N \ge 4$ reference point pairs and reports Mean Squared / Euclidean error on held-out test points.
