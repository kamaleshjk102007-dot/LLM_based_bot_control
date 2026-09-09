# Calibration Documentation — Member 3

## 1. Mathematical Principle
Planar homography ($3 \times 3$ matrix $H$) maps points on the camera image plane $(u, v)$ to the physical tabletop plane $(X, Y)$ in DOBOT Magician Lite base millimeters:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad X_{robot} = \frac{x'}{w'}, \quad Y_{robot} = \frac{y'}{w'}$$

## 2. Calibration Procedure
1. Mount camera securely above the tabletop workspace.
2. Mark at least 4 non-collinear physical points on the tabletop.
3. Record robot arm base coordinates $(X_i, Y_i)$ in millimeters using DobotLink / `GetPose`.
4. Record corresponding pixel locations $(u_i, v_i)$ from the camera feed.
5. Compute $H$ using `cv2.findHomography`.

## 3. Hold-Out Validation
Every calibration profile must be evaluated against held-out ground truth points not used during matrix estimation:
* $\Delta X = |X_{predicted} - X_{true}|$
* $\Delta Y = |Y_{predicted} - Y_{true}|$
* $\text{Euclidean Error} = \sqrt{\Delta X^2 + \Delta Y^2}$
* **Accuracy Requirement**: Mean Euclidean Error $< 5.0$ mm across the workspace envelope.

## 4. Physical Z Modeling
Standard 2D cameras cannot measure metric height directly. To avoid faking $Z$:
$$Z_{target} = Z_{table} + h_{object} + \text{offset}$$
* $Z_{table}$: Calibrated table elevation in `dobot_base` (e.g. $-50.0$ mm).
* $h_{object}$: Known physical height of the object class (e.g. $25.0$ mm for blocks).

## 5. Storage
Calibration profiles are saved to and loaded from JSON files (`config/calibration.json`).
Run the calibration tool via:
```bash
python examples/calibrate_tabletop.py --output config/calibration.json
```
