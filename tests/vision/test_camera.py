"""Unit tests for vision.camera."""

from vision.camera import FileCamera, MockCamera, USBCamera


def test_mock_camera_lifecycle():
    cam = MockCamera(width=640, height=480)
    assert not cam.is_opened()

    assert cam.open() is True
    assert cam.is_opened()

    frame = cam.read()
    assert frame is not None
    assert frame.width == 640
    assert frame.height == 480
    assert frame.image.shape == (480, 640, 3)

    cam.release()
    assert not cam.is_opened()
    assert cam.read() is None


def test_mock_camera_context_manager():
    with MockCamera(width=320, height=240) as cam:
        assert cam.is_opened()
        frame = cam.read()
        assert frame is not None
        assert frame.width == 320
    assert not cam.is_opened()


def test_mock_camera_renders_objects():
    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(320.0, 240.0), size=(50, 50))
    frame = cam.read()
    assert frame is not None

    # Check that pixel at center (240, 320) is red-ish (B, G, R where R is high)
    b, g, r = frame.image[240, 320]
    assert r > 180
    assert b < 100
    cam.release()


def test_usb_camera_unavailable():
    # Attempt opening invalid device index
    cam = USBCamera(device_index=999)
    assert cam.open() is False
    assert not cam.is_opened()
    assert cam.read() is None
    cam.release()


def test_file_camera_nonexistent():
    cam = FileCamera(file_path="nonexistent_video_12345.mp4")
    assert cam.open() is False
    assert not cam.is_opened()
    assert cam.read() is None
    cam.release()
