"""Architecture guard: hardware integration stays optional and adapter-isolated."""

from pathlib import Path


FORBIDDEN_DEPENDENCIES = {
    "dobot", "dobotlink", "rclpy", "ros2",
    "pyserial", "python-can", "pyusb",
}
FORBIDDEN_IMPORT_FRAGMENTS = (
    "import serial", "from serial", "import can", "from can",
    "import rclpy", "from rclpy", "import usb", "from usb",
    "import dobotrpc", "from dobotrpc",
)


def test_base_requirements_have_no_vendor_hardware_dependency():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert not any(name in requirements for name in FORBIDDEN_DEPENDENCIES)


def test_vendor_transport_imports_are_isolated_to_dobot_adapter():
    allowed_root = Path("app/adapters/dobot")
    violations = []
    for path in Path("app").rglob("*.py"):
        if path.is_relative_to(allowed_root):
            continue
        source = path.read_text(encoding="utf-8").lower()
        if any(fragment in source for fragment in FORBIDDEN_IMPORT_FRAGMENTS):
            violations.append(str(path))
    assert violations == []


def test_no_incompatible_legacy_dobot_sdk_import():
    source = Path("app/adapters/dobot/client.py").read_text(encoding="utf-8")
    assert "DobotRPC" not in source
    assert "websockets.sync.client" in source
