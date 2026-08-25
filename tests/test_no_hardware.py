"""Architectural guard: Phase 2 must remain free of hardware integrations."""

from pathlib import Path


FORBIDDEN_DEPENDENCIES = {
    "dobot",
    "dobotlink",
    "rclpy",
    "ros2",
    "pyserial",
    "python-can",
    "pyusb",
}
FORBIDDEN_IMPORT_FRAGMENTS = (
    "import serial",
    "from serial",
    "import can",
    "from can",
    "import rclpy",
    "from rclpy",
    "import usb",
    "from usb",
    "dobotlink",
)


def test_no_hardware_dependencies_or_imports():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert not any(name in requirements for name in FORBIDDEN_DEPENDENCIES)

    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in Path("app").rglob("*.py")
    )
    assert not any(fragment in source for fragment in FORBIDDEN_IMPORT_FRAGMENTS)
