"""Behavioral regression tests against the controller with a lagging arm model."""
import ast
import math
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def controller(monkeypatch):
    directory = Path('simulation/webots/controllers/llm_robot_controller').resolve()
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setitem(sys.modules, 'controller', types.SimpleNamespace(Supervisor=object))
    tree = ast.parse((directory / 'llm_robot_controller.py').read_text())
    # Load declarations without starting a simulator or binding its TCP server.
    declarations = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'sim' for target in node.targets
        ):
            break
        if not isinstance(node, ast.If):
            declarations.append(node)
    scope = {'__name__': 'cartesian_controller_test'}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), '<controller>', 'exec'), scope)
    return scope


def arm(controller, base_angle=0.0, lag=0.4):
    sim = controller['Simulator'].__new__(controller['Simulator'])
    sim.targets = dict(controller['HOME'], base_motor=base_angle)
    actual = dict(sim.targets)

    def geometry():
        base, shoulder, elbow = (actual[name] for name in
                                ('base_motor', 'shoulder_motor', 'elbow_motor'))
        radial = .15 * math.sin(shoulder) + .15 * math.sin(shoulder + elbow)
        return {
            'ARM_BASE': (0., 0., .08),
            'SHOULDER_LINK': (0., 0., .13),
            'ELBOW_LINK': (.15 * math.sin(shoulder) * math.cos(base),
                           .15 * math.sin(shoulder) * math.sin(base),
                           .13 + .15 * math.cos(shoulder)),
            'END_EFFECTOR': (radial * math.cos(base), radial * math.sin(base),
                             .13 + .15 * math.cos(shoulder)
                             + .15 * math.cos(shoulder + elbow)),
        }

    class Node:
        def __init__(self, name):
            self.name = name

        def getPosition(self):
            return geometry()[self.name]

        def getOrientation(self):
            angle = actual['base_motor']
            c, s = math.cos(angle), math.sin(angle)
            return (c, -s, 0., s, c, 0., 0., 0., 1.)

    sim.nodes = {name: Node(name) for name in geometry()}

    def advance(steps):
        for _ in range(steps):
            for name in actual:
                delta = (sim.targets[name] - actual[name]) * lag
                actual[name] += max(-.7 * .032, min(.7 * .032, delta))

    def set_targets():
        for name, value in sim.targets.items():
            sim.targets[name] = sim.clamp(name, value)

    sim.advance = advance
    sim.set_targets = set_targets
    sim.show_motion_indicator = lambda *args: None
    sim.stopped = False
    return sim


@pytest.mark.parametrize('angle', [0, 30, 90, -60, 120])
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
@pytest.mark.parametrize('distance', [-.005, .005])
def test_world_axis_moves_hold_other_axes(controller, angle, axis, distance):
    sim = arm(controller, math.radians(angle))
    report = getattr(sim, f'move_cartesian_{axis}')(distance)
    assert report['verified']
    assert abs(report['error_mm']) < .25
    for other in set('xyz') - {axis}:
        assert abs(report[f'{other}_drift_mm']) < .25


def test_slow_motors_and_repeated_moves_do_not_accumulate_drift(controller):
    sim = arm(controller, math.radians(45), lag=.06)
    start = sim.end_effector_position()
    for _ in range(5):
        for distance in [.005, -.005]:
            assert sim.move_cartesian_x(distance)['verified']
    sim.advance(300)
    assert math.dist(start, sim.end_effector_position()) < .00075


def test_solver_failure_restores_targets(controller, monkeypatch):
    sim = arm(controller)
    original = dict(sim.targets)

    def fail(*args):
        sim.targets['base_motor'] += .1
        raise ValueError('solver failed')

    monkeypatch.setitem(controller, 'damped_xyz_step', fail)
    with pytest.raises(ValueError, match='solver failed'):
        sim.move_cartesian_x(.005)
    assert sim.targets == original


def test_joint_limit_is_rejected_before_applying_correction(controller, monkeypatch):
    sim = arm(controller, controller['LIMITS']['base_motor'][1])
    original = dict(sim.targets)
    monkeypatch.setitem(controller, 'damped_xyz_step', lambda *args: (.01, 0., 0.))

    with pytest.raises(ValueError, match='joint limit'):
        sim.move_cartesian_x(.005)

    assert sim.targets == original
