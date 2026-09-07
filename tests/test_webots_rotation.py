import ast
import math
from pathlib import Path

import pytest

from simulation.webots.controllers.llm_robot_controller.rotation_motion import (
    angular_report,
    requested_r_radians,
)


def test_positive_and_negative_r_are_preserved():
    assert requested_r_radians({
        "action": "ROTATE", "direction": "R", "angle": 5, "unit": "degrees"
    }) == pytest.approx(math.radians(5))
    assert requested_r_radians({
        "action": "ROTATE", "direction": "-R", "angle": 5, "unit": "degrees"
    }) == pytest.approx(math.radians(-5))


def test_non_r_rotation_keeps_legacy_path():
    assert requested_r_radians({
        "action": "ROTATE", "direction": "left", "angle": 5, "unit": "degrees"
    }) is None


def test_actual_wrist_rotation_is_measured():
    report = angular_report(
        math.radians(10), math.radians(15.1), math.radians(5)
    )
    assert report["actual_r_deg"] == pytest.approx(5.1)
    assert report["error_deg"] == pytest.approx(0.1)
    assert report["verified"] is True


def test_rotation_outside_tolerance_fails():
    report = angular_report(0, math.radians(4.5), math.radians(5))
    assert report["verified"] is False


def test_world_has_wrist_sensor_and_controller_verification():
    world = Path("simulation/webots/worlds/magician_lite.wbt").read_text(
        encoding="utf-8"
    )
    controller = Path(
        "simulation/webots/controllers/llm_robot_controller/"
        "llm_robot_controller.py"
    ).read_text(encoding="utf-8")
    ast.parse(controller)
    assert 'PositionSensor { name "wrist_sensor" }' in world
    assert 'getDevice("wrist_sensor")' in controller
    assert "def rotate_wrist" in controller
    assert "angular_report(" in controller
    assert "translation_drift_mm" in controller
    assert "ROTATE R verified" in controller
