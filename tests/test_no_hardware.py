"""Architecture guard: hardware integration stays optional and adapter-isolated."""

from pathlib import Path


FORBIDDEN_DEPENDENCIES = {
    "dobot", "dobotlink", "rclpy", "ros2",
    "pyserial", "python-can", "pyusb",
}
FORBIDDEN_IMPORT_FRAGMENTS = (
    "import serial", "from serial", "import can", "from can",
    "import rclpy", "from rclpy", "import usb", "from usb",
    "dobotrpc", "dobotlink",
)


def test_base_requirements_have_no_hardware_dependency():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert not any(name in requirements for name in FORBIDDEN_DEPENDENCIES)


def test_vendor_transport_is_isolated_to_dobot_adapter():
    allowed_root = Path("app/adapters/dobot")
    violations = []
    for path in Path("app").rglob("*.py"):
        if path.is_relative_to(allowed_root):
            continue
        source = path.read_text(encoding="utf-8").lower()
        if any(fragment in source for fragment in FORBIDDEN_IMPORT_FRAGMENTS):
            violations.append(str(path))
    assert violations == []


def test_dobot_sdk_is_lazy_loaded():
    source = Path("app/adapters/dobot/client.py").read_text(encoding="utf-8")
    assert "importlib.import_module(\"DobotRPC\")" in source
    assert "import DobotRPC" not in source
