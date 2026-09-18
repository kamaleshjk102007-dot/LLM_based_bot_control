# Perception & Target Selection Documentation — Member 3

## 1. Safety Policies

### Ambiguity Policy (Zero Random Guessing)
When the task specifies an object class (e.g. `red_block`):
* 0 matches $\to$ `status: "NOT_FOUND"`, `valid: false`
* 1 confident match $\to$ Candidate for validation
* 2+ matching objects $\to$ `status: "AMBIGUOUS"`, `valid: false`

The robot will never guess or randomly choose between multiple objects unless Member 2 provides an explicit disambiguation rule (e.g. "pick nearest").

### Confidence Policy
Detections with confidence $< \tau$ (default $\tau = 0.60$) are rejected with `status: "LOW_CONFIDENCE"`.

### Temporal Stability Filter (`TemporalStabilityTracker`)
To prevent autonomous movement based on a single noisy camera frame:
* Maintains a sliding window (default 4 frames).
* Requires $\ge 3$ consistent frames before certifying stability.
* Rejects position jitter when planar standard deviation $> 3.5$ mm (`status: "UNSTABLE"`).
* Automatically resets if a sudden position jump $> 20.0$ mm is observed.

### Target Freshness Checker (`TargetFreshnessChecker`)
Targets older than $1.5$ seconds are marked `status: "STALE"` and `valid: false`. They cannot cross the explicit Member 3 -> Member 4 conversion boundary.

---

## 2. Member 4 Integration Contract (`RobotTarget`)

The verified JSON contract dispatched to Member 4:

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

### Coordinate Frame Specification:
* Frame ID: strictly `"dobot_base"`
* Unit: Millimeters
* $X$: Forward from DOBOT base center
* $Y$: Left/Right lateral offset
* $Z$: Vertical height relative to base origin
