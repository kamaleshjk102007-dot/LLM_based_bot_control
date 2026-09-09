"""Backward-compatible adapter base and vendor-neutral adapter error."""

from __future__ import annotations

from robots.interface import RobotInterface


class RobotAdapterError(RuntimeError):
    """A vendor-neutral adapter preparation or execution failure."""


class RobotAdapter(RobotInterface):
    """Compatibility base for existing adapters.

    The common abstract contract now lives in robots.interface.RobotInterface.
    Existing adapter class names and inheritance remain unchanged.
    """
