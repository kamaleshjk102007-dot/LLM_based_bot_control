import pytest

from app.adapters.mock import MockRobotAdapter
from app.gateway.adapter_manager import (
    AdapterManager,
    AdapterManagerError,
    AdapterNotFoundError,
)
from conftest import command, robot


def test_mock_adapter_registration_lookup_and_simulation():
    manager = AdapterManager()
    adapter = manager.get(robot())
    assert isinstance(adapter, MockRobotAdapter)
    cmd = command({"action": "STOP"})
    assert adapter.validate(cmd)
    assert adapter.prepare(cmd) == [{"sequence": 1, "action": "STOP"}]
    assert adapter.execute(cmd) == ["[MOCK] STOP accepted"]
    assert adapter.get_status() == "SIMULATED"


def test_duplicate_and_unknown_adapters_are_rejected():
    manager = AdapterManager()
    with pytest.raises(AdapterManagerError):
        manager.register("mock", MockRobotAdapter)
    with pytest.raises(AdapterNotFoundError):
        manager.get(robot(adapter_type="future_adapter"))
