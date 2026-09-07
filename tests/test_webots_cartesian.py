import ast
from pathlib import Path

import pytest

from simulation.webots.controllers.llm_robot_controller.cartesian_motion import (
    damped_xz_step,
    displacement_report,
    requested_x_metres,
)


def test_universal_positive_x_five_mm_is_preserved():
    task = {
        "action": "MOVE",
        "direction": "X",
        "distance": 5.0,
        "unit": "mm",
    }
    assert requested_x_metres(task) == pytest.approx(0.005)


def test_x_sign_aliases_and_unit_guard():
    assert requested_x_metres({
        "direction": "-X", "distance": 5, "unit": "millimeters"
    }) == pytest.approx(-0.005)
    assert requested_x_metres({
        "direction": "forward", "distance": 5, "unit": "mm"
    }) is None
    with pytest.raises(ValueError, match="millimeters"):
        requested_x_metres({
            "direction": "X", "distance": 5, "unit": "centimeters"
        })


def test_damped_solver_returns_bounded_joint_correction():
    dq1, dq2 = damped_xz_step(
        ((0.25, 0.0), (0.0, 0.20)),
        (0.005, 0.0),
    )
    assert 0 < dq1 <= 0.04
    assert abs(dq2) < 1e-9


def test_measured_five_mm_displacement_is_verified():
    report = displacement_report(
        (0.100, 0.0, 0.200),
        (0.105, 0.0, 0.200),
        0.005,
    )
    assert report["requested_x_mm"] == pytest.approx(5.0)
    assert report["actual_x_mm"] == pytest.approx(5.0)
    assert report["error_mm"] == pytest.approx(0.0)
    assert report["verified"] is True


def test_wrong_measured_displacement_fails_verification():
    report = displacement_report(
        (0.100, 0.0, 0.200),
        (0.102, 0.0, 0.200),
        0.005,
    )
    assert report["actual_x_mm"] == pytest.approx(2.0)
    assert report["verified"] is False


def test_world_exposes_supervisor_measurement_nodes():
    world = open(
        "simulation/webots/worlds/magician_lite.wbt",
        encoding="utf-8",
    ).read()
    assert "supervisor TRUE" in world
    for name in ("ARM_BASE", "SHOULDER_LINK", "ELBOW_LINK", "END_EFFECTOR"):
        assert f"DEF {name} Solid" in world
    assert "DEF MOVE_START_MARKER Transform" in world
    assert "DEF MOVE_END_MARKER Transform" in world
    assert "DEF MOVE_TRAJECTORY_COORD Coordinate" in world


def test_webots_controller_is_valid_python():
    source = Path(
        "simulation/webots/controllers/llm_robot_controller/"
        "llm_robot_controller.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)
    assert "requested_x_metres(task)" in source
    assert "displacement_report(" in source
    assert "show_motion_indicator(before, after, report)" in source
    assert "setSFVec3f(start)" in source
    assert "setSFVec3f(end)" in source
    assert "self.robot.setLabel(" in source
