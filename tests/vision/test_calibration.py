"""Unit tests for vision.calibration."""

import os
import tempfile
import pytest

from vision.calibration import TabletopCalibration


def test_calibration_minimum_points_enforced():
    calib = TabletopCalibration()
    with pytest.raises(ValueError, match="at least 4 point pairs"):
        calib.calibrate([((10.0, 10.0), (100.0, 100.0))])


def test_calibration_forward_and_inverse_mapping():
    # 4 known corner points on tabletop
    pairs = [
        ((100.0, 100.0), (300.0, -100.0)),
        ((500.0, 100.0), (300.0, 100.0)),
        ((500.0, 400.0), (180.0, 100.0)),
        ((100.0, 400.0), (180.0, -100.0)),
    ]

    calib = TabletopCalibration()
    success = calib.calibrate(pairs)
    assert success is True
    assert calib.is_calibrated

    # Check forward mapping on training points
    for (u, v), (exp_x, exp_y) in pairs:
        rx, ry = calib.pixel_to_robot(u, v)
        assert pytest.approx(rx, abs=0.5) == exp_x
        assert pytest.approx(ry, abs=0.5) == exp_y

        # Check inverse mapping
        pu, pv = calib.robot_to_pixel(rx, ry)
        assert pytest.approx(pu, abs=0.5) == u
        assert pytest.approx(pv, abs=0.5) == v


def test_calibration_held_out_validation():
    # Training points
    train_pairs = [
        ((100.0, 100.0), (300.0, -100.0)),
        ((500.0, 100.0), (300.0, 100.0)),
        ((500.0, 400.0), (180.0, 100.0)),
        ((100.0, 400.0), (180.0, -100.0)),
    ]

    calib = TabletopCalibration()
    calib.calibrate(train_pairs)

    # Held-out center point: halfway in pixels (300, 250) -> halfway in robot (240, 0)
    val_pairs = [
        ((300.0, 250.0), (240.0, 0.0)),
    ]

    metrics = calib.validate(val_pairs)
    assert metrics.num_validation_points == 1
    assert metrics.mean_euclidean_error_mm < 1.0  # Linear homography should be exact here
    assert metrics.mean_x_error_mm < 1.0
    assert metrics.mean_y_error_mm < 1.0


def test_calibration_save_and_load():
    calib = TabletopCalibration.create_default()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_calib.json")
        calib.save(filepath)
        assert os.path.exists(filepath)

        loaded_calib = TabletopCalibration()
        loaded_calib.load(filepath)
        assert loaded_calib.is_calibrated

        # Ensure mappings are identical
        rx1, ry1 = calib.pixel_to_robot(320.0, 240.0)
        rx2, ry2 = loaded_calib.pixel_to_robot(320.0, 240.0)
        assert pytest.approx(rx1, abs=0.001) == rx2
        assert pytest.approx(ry1, abs=0.001) == ry2
