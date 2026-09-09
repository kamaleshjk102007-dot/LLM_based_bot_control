"""
Camera abstraction layer for Member 3.
Provides unified interface across physical USB cameras, simulation (Webots),
recorded files, and synthetic mock cameras.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .models import Frame


class BaseCamera(ABC):
    """Abstract base class for all camera sources."""

    def __init__(self, name: str = "camera"):
        self.name = name
        self._frame_count: int = 0

    @abstractmethod
    def open(self) -> bool:
        """Open camera stream or resource. Returns True if successful."""
        pass

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read next frame from camera. Returns None if frame read fails."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resource."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is currently opened."""
        pass

    def __enter__(self) -> BaseCamera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class USBCamera(BaseCamera):
    """Physical USB / webcam device wrapper using OpenCV."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        name: str = "usb_camera",
    ):
        super().__init__(name=name)
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True

        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read(self) -> Optional[Frame]:
        if not self.is_opened():
            return None

        ret, img = self._cap.read()
        if not ret or img is None:
            return None

        self._frame_count += 1
        return Frame(
            frame_id=f"{self.name}_{self._frame_count:06d}",
            timestamp=time.time(),
            width=img.shape[1],
            height=img.shape[0],
            image=img,
        )

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FileCamera(BaseCamera):
    """Camera source loading from a recorded video or static image file."""

    def __init__(self, file_path: str, loop: bool = True, name: str = "file_camera"):
        super().__init__(name=name)
        self.file_path = file_path
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._static_image: Optional[np.ndarray] = None
        self._is_static: bool = False

    def open(self) -> bool:
        # Check if file is a static image
        img = cv2.imread(self.file_path)
        if img is not None:
            self._static_image = img
            self._is_static = True
            return True

        # Otherwise attempt video capture
        self._cap = cv2.VideoCapture(self.file_path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._is_static = False
        return True

    def read(self) -> Optional[Frame]:
        if not self.is_opened():
            return None

        self._frame_count += 1
        now = time.time()

        if self._is_static:
            return Frame(
                frame_id=f"{self.name}_{self._frame_count:06d}",
                timestamp=now,
                width=self._static_image.shape[1],
                height=self._static_image.shape[0],
                image=self._static_image.copy(),
            )

        ret, img = self._cap.read()
        if not ret or img is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, img = self._cap.read()
                if not ret or img is None:
                    return None
            else:
                return None

        return Frame(
            frame_id=f"{self.name}_{self._frame_count:06d}",
            timestamp=now,
            width=img.shape[1],
            height=img.shape[0],
            image=img,
        )

    def is_opened(self) -> bool:
        return self._is_static or (self._cap is not None and self._cap.isOpened())

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._static_image = None
        self._is_static = False


class MockCamera(BaseCamera):
    """
    Synthetic camera generating simulated tabletop scenes for testing.
    Can render red, green, blue, and yellow blocks at configurable positions,
    with optional noise and coordinate jitter.
    """

    # BGR color definitions
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "red_block": (30, 30, 220),      # BGR: strong red
        "blue_block": (220, 60, 30),     # BGR: strong blue
        "green_block": (40, 200, 40),    # BGR: bright green
        "yellow_block": (30, 210, 220),  # BGR: bright yellow
    }

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        name: str = "mock_camera",
        jitter_std: float = 0.0,
    ):
        super().__init__(name=name)
        self.width = width
        self.height = height
        self.jitter_std = jitter_std
        self._opened = False
        self._objects: List[Dict[str, Any]] = []

    def set_objects(self, objects: List[Dict[str, Any]]) -> None:
        """
        Configure synthetic objects to render.
        Each item: {"class_name": str, "center": (u, v), "size": (w, h)}
        """
        self._objects = objects

    def add_object(self, class_name: str, center: Tuple[float, float], size: Tuple[int, int] = (60, 60)) -> None:
        self._objects.append({"class_name": class_name, "center": center, "size": size})

    def clear_objects(self) -> None:
        self._objects.clear()

    def open(self) -> bool:
        self._opened = True
        return True

    def is_opened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self._opened = False

    def read(self) -> Optional[Frame]:
        if not self.is_opened():
            return None

        self._frame_count += 1
        now = time.time()

        # Create neutral tabletop canvas (light wood/gray tone)
        canvas = np.full((self.height, self.width, 3), (220, 225, 230), dtype=np.uint8)

        # Draw subtle grid lines to resemble a calibration mat / workbench
        for x in range(0, self.width, 40):
            cv2.line(canvas, (x, 0), (x, self.height), (205, 210, 215), 1)
        for y in range(0, self.height, 40):
            cv2.line(canvas, (0, y), (self.width, y), (205, 210, 215), 1)

        # Render objects
        for obj in self._objects:
            cname = obj["class_name"]
            color = self.COLORS.get(cname, (100, 100, 100))
            cx, cy = obj["center"]

            # Apply random jitter if specified
            if self.jitter_std > 0:
                cx += np.random.normal(0, self.jitter_std)
                cy += np.random.normal(0, self.jitter_std)

            w, h = obj.get("size", (60, 60))
            x1 = int(round(cx - w / 2))
            y1 = int(round(cy - h / 2))
            x2 = int(round(cx + w / 2))
            y2 = int(round(cy + h / 2))

            # Clamp to canvas
            x1 = max(0, min(self.width - 1, x1))
            y1 = max(0, min(self.height - 1, y1))
            x2 = max(0, min(self.width - 1, x2))
            y2 = max(0, min(self.height - 1, y2))

            # Draw filled block
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            # Draw darker border
            border_color = tuple(max(0, c - 40) for c in color)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2)

        return Frame(
            frame_id=f"{self.name}_{self._frame_count:06d}",
            timestamp=now,
            width=self.width,
            height=self.height,
            image=canvas,
        )


class WebotsCamera(BaseCamera):
    """
    Adapter for Webots Robot Simulator camera device.
    Converts Webots BGRA buffer into standard OpenCV BGR image array.
    """

    def __init__(self, camera_device: Any, name: str = "webots_camera"):
        """
        :param camera_device: Webots Camera device handle (e.g. robot.getDevice('camera'))
        """
        super().__init__(name=name)
        self.camera_device = camera_device
        self._opened = False

    def open(self) -> bool:
        if self.camera_device is None:
            return False
        # Webots camera requires enabling with a sampling period (e.g. 32ms)
        try:
            if hasattr(self.camera_device, "enable"):
                self.camera_device.enable(32)
            self._opened = True
            return True
        except Exception:
            self._opened = False
            return False

    def read(self) -> Optional[Frame]:
        if not self.is_opened():
            return None

        try:
            image_data = self.camera_device.getImage()
            if image_data is None:
                return None

            width = self.camera_device.getWidth()
            height = self.camera_device.getHeight()

            # Webots returns BGRA raw buffer
            img_arr = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
            # Convert BGRA to BGR
            bgr_img = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)

            self._frame_count += 1
            return Frame(
                frame_id=f"{self.name}_{self._frame_count:06d}",
                timestamp=time.time(),
                width=width,
                height=height,
                image=bgr_img,
            )
        except Exception:
            return None

    def is_opened(self) -> bool:
        return self._opened

    def release(self) -> None:
        if self._opened and self.camera_device is not None:
            try:
                if hasattr(self.camera_device, "disable"):
                    self.camera_device.disable()
            except Exception:
                pass
        self._opened = False
