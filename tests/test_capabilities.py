import pytest

from app.gateway.capability_manager import CapabilityError, CapabilityManager
from conftest import command, robot


def test_supported_and_unsupported_actions():
    manager = CapabilityManager()
    arm = robot()
    assert manager.has_capability(arm, "PICK")
    assert not manager.has_capability(arm, "NAVIGATE")


def test_multitask_validation_is_all_or_nothing():
    manager = CapabilityManager()
    arm = robot(capabilities=("PICK", "PLACE"))
    supported = command(
        {"action": "PICK", "object": {"type": "cube"}},
        {"action": "PLACE", "target": {"type": "box"}},
    )
    manager.validate_command_against_capabilities(arm, supported)

    unsupported = command(
        {"action": "PICK", "object": {"type": "cube"}},
        {"action": "MOVE", "direction": "forward"},
    )
    with pytest.raises(CapabilityError, match="MOVE"):
        manager.validate_command_against_capabilities(arm, unsupported)
