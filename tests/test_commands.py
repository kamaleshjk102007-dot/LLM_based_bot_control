"""Valid command model tests."""

import pytest

from app.commands.models import Action, UniversalCommand


@pytest.mark.parametrize(
    ("task", "action"),
    [
        ({"action": "PICK", "object": {"type": "cube", "color": "red"}}, Action.PICK),
        ({"action": "PLACE", "target": {"type": "box", "id": "A"}}, Action.PLACE),
        (
            {"action": "MOVE", "direction": "forward", "distance": 20, "unit": "cm"},
            Action.MOVE,
        ),
        ({"action": "ROTATE", "angle": 90, "unit": "degrees"}, Action.ROTATE),
        ({"action": "STOP"}, Action.STOP),
        ({"action": "HOME"}, Action.HOME),
        ({"action": "GET_STATUS"}, Action.GET_STATUS),
    ],
)
def test_valid_commands(task, action):
    command = UniversalCommand.model_validate(
        {"version": "1.0", "robot_id": None, "tasks": [task]}
    )
    assert command.tasks[0].action is action


def test_multistep_command():
    command = UniversalCommand.model_validate(
        {
            "version": "1.0",
            "robot_id": None,
            "tasks": [
                {"action": "PICK", "object": {"type": "cube", "color": "red"}},
                {"action": "PLACE", "target": {"type": "box", "id": "A"}},
                {"action": "MOVE", "direction": "forward", "distance": 20, "unit": "cm"},
            ],
        }
    )
    assert [task.action.value for task in command.tasks] == ["PICK", "PLACE", "MOVE"]
