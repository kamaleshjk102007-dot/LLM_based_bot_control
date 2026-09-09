# Member 3 — Vision + Sensors + Perception System

**Branch:** `member-3/vision-perception`  
**Robot:** DOBOT Magician Lite (Robot 1)

---

## 🎯 Role & Responsibility

Member 3 answers: **"What does the robot see, and where exactly is the target in the robot's base coordinate system?"**

### Core Pipeline
```text
Task Request (from Member 2 / LLM)
       ↓
 Camera Source (Mock / USB / Webots / File)
       ↓
  Frame Capture (640x480)
       ↓
 Object Detection (HSV Color & Shape / YOLO)
       ↓
 Target Selection (NOT_FOUND / LOW_CONFIDENCE / AMBIGUOUS / VALID)
       ↓
Tabletop Calibration (3x3 Planar Homography)
       ↓
Coordinate Transform (Pixels → dobot_base X, Y, Z in mm)
       ↓
Temporal Stability & Freshness Check (variance filter, age < 1.5s)
       ↓
RobotTarget Contract (Dispatched to Member 4)
```

---

## 🤝 Contract with Member 4 (`RobotTarget`)

Member 4 consumes the standard `RobotTarget` JSON:

```json
{
  "target_id": "tgt_b576a545",
  "class_name": "red_block",
  "position": {
    "x": 239.8,
    "y": 0.0,
    "z": -25.0
  },
  "confidence": 0.984,
  "timestamp": "2026-09-09T05:18:23.177784+00:00",
  "coordinate_frame": "dobot_base",
  "valid": true,
  "status": "VALID",
  "stability_score": 0.87,
  "source_detection_id": "det_5154ae60",
  "message": "Valid target verified at X:239.8, Y:0.0, Z:-25.0 mm"
}
```

### Safety Guarantees Provided by Member 3:
1. **Ambiguity Rejection**: If 2+ matching objects exist without a task selection rule, returns `status: "AMBIGUOUS"` and `valid: false`.
2. **Deterministic Z**: Computes $Z = Z_{table} + h_{object}$ using known physical block dimensions. Never guesses $Z$ from 2D pixel scale.
3. **Temporal Stability**: Requires consistent position across consecutive frames; noisy jitter returns `status: "UNSTABLE"`.
4. **Target Freshness**: Outdated targets ($> 1.5$s) are flagged `status: "STALE"`.
5. **Workspace Verification**: Validates that $(X, Y, Z)$ lie within the DOBOT Magician Lite reach envelope ($R \in [140, 330]$ mm).

---

## 🧪 Testing

Run all Member 3 perception tests:
```bash
python -m pytest tests/vision/ -v
```

Run full repository test suite:
```bash
python -m pytest -m "not live and not hardware"
```
