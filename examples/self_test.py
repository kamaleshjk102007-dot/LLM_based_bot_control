"""
Member 3 -- Interactive Self-Test Tool
Run this and test every scenario yourself by choosing from the menu.
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision import (
    MockCamera,
    VisionPipeline,
    ColorShapeDetector,
    TabletopCalibration,
    CoordinateTransformer,
    ZEstimator,
    TargetSelector,
    TemporalStabilityTracker,
)
from vision.models import BoundingBox, Detection, Point2D, Point3D

SEPARATOR = "=" * 60
LINE      = "-" * 60

def pause():
    input("\n  [Press ENTER to continue...]\n")

def print_target(target):
    print()
    if target.valid:
        print(f"  [OK]  STATUS      : {target.status.value}")
        print(f"  [OK]  CLASS       : {target.class_name}")
        print(f"  [OK]  X (forward) : {target.position.x:.1f} mm")
        print(f"  [OK]  Y (sideways): {target.position.y:.1f} mm")
        print(f"  [OK]  Z (height)  : {target.position.z:.1f} mm")
        print(f"  [OK]  CONFIDENCE  : {target.confidence * 100:.1f}%")
        print(f"  [OK]  STABILITY   : {target.stability_score * 100:.1f}%")
        print(f"  [OK]  FRAME       : dobot_base")
    else:
        print(f"  [!!]  STATUS  : {target.status.value}")
        print(f"  [!!]  REASON  : {target.message}")


def test_1_single_block():
    print(SEPARATOR)
    print("  TEST 1 -- Single Red Block (should become VALID after 3 frames)")
    print(SEPARATOR)

    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(320.0, 250.0), size=(60, 60))

    pipeline = VisionPipeline(camera=cam, stability_samples=3)

    for i in range(1, 5):
        target, _ = pipeline.capture_and_process("red_block")
        print(f"\n  Frame #{i}:")
        print_target(target)

    cam.release()
    pause()


def test_2_choose_color():
    print(SEPARATOR)
    print("  TEST 2 -- You Pick The Block Color")
    print(SEPARATOR)
    print()
    print("  Available blocks you can add to the scene:")
    print("    red_block")
    print("    blue_block")
    print("    green_block")
    print("    yellow_block")
    print()

    choice = input("  Which block do you want ON the table? (e.g. red_block): ").strip()
    query  = input("  Which block should the robot LOOK FOR? (e.g. red_block): ").strip()

    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object(choice, center=(300.0, 220.0), size=(60, 60))

    pipeline = VisionPipeline(camera=cam, stability_samples=3)

    print(f"\n  Table has   : {choice}")
    print(f"  Looking for : {query}")

    for i in range(1, 5):
        target, _ = pipeline.capture_and_process(query)
        print(f"\n  Frame #{i}:")
        print_target(target)

    cam.release()
    pause()


def test_3_ambiguous():
    print(SEPARATOR)
    print("  TEST 3 -- Ambiguity: 2 Red Blocks (should REJECT)")
    print(SEPARATOR)
    print("  Rule: If 2+ matching objects exist --> AMBIGUOUS (never guesses!)")
    print()

    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(200.0, 200.0), size=(60, 60))
    cam.add_object("red_block", center=(440.0, 300.0), size=(60, 60))

    pipeline = VisionPipeline(camera=cam)
    target, _ = pipeline.capture_and_process("red_block")
    print_target(target)

    cam.release()
    pause()


def test_4_not_found():
    print(SEPARATOR)
    print("  TEST 4 -- Object Not Found (only blue on table, asking for red)")
    print(SEPARATOR)
    print()

    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("blue_block", center=(320.0, 240.0), size=(60, 60))

    pipeline = VisionPipeline(camera=cam)
    target, _ = pipeline.capture_and_process("red_block")
    print_target(target)

    cam.release()
    pause()


def test_5_low_confidence():
    print(SEPARATOR)
    print("  TEST 5 -- Low Confidence (block too tiny -- rejected)")
    print(SEPARATOR)
    print("  A 12x12 pixel blob is too small -- area < min_area threshold --> rejected")
    print()

    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block", center=(320.0, 240.0), size=(12, 12))  # too tiny

    pipeline = VisionPipeline(camera=cam, min_confidence=0.90)
    target, _ = pipeline.capture_and_process("red_block")
    print_target(target)

    cam.release()
    pause()


def test_6_stability_jitter():
    print(SEPARATOR)
    print("  TEST 6 -- Jitter / Unstable Detection")
    print(SEPARATOR)
    print("  Using large pixel jitter --> position jumps every frame --> UNSTABLE")
    print()

    cam = MockCamera(width=640, height=480, jitter_std=25.0)
    cam.open()
    cam.add_object("red_block", center=(320.0, 250.0), size=(60, 60))

    pipeline = VisionPipeline(camera=cam, stability_samples=3)

    for i in range(1, 5):
        target, _ = pipeline.capture_and_process("red_block")
        print(f"  Frame #{i}: {target.status.value} -- {target.message}")

    cam.release()
    pause()


def test_7_calibration():
    print(SEPARATOR)
    print("  TEST 7 -- Calibration: You Enter Pixel, See Robot mm Coordinates")
    print(SEPARATOR)
    print("  Enter any pixel coordinate and see where it maps in robot space.")
    print("  Image size: 640 x 480 pixels.")
    print()

    calib = TabletopCalibration.create_default()
    transformer = CoordinateTransformer(calibration=calib, table_z_mm=-50.0)

    while True:
        u_str = input("  Enter pixel U (horizontal, 0-640) or 'q' to quit: ").strip()
        if u_str.lower() == 'q':
            break
        v_str = input("  Enter pixel V (vertical,   0-480): ").strip()
        try:
            u, v = float(u_str), float(v_str)
            rx, ry = calib.pixel_to_robot(u, v)
            rz = -50.0 + 25.0  # table_z + red_block height

            is_reachable, reason = transformer.check_reachability(
                Point3D(x=round(rx, 2), y=round(ry, 2), z=rz)
            )

            print()
            print(f"  Pixel ({u:.0f}, {v:.0f})")
            print(f"  --> Robot  X = {rx:.1f} mm  (forward)")
            print(f"  --> Robot  Y = {ry:.1f} mm  (sideways)")
            print(f"  --> Robot  Z = {rz:.1f} mm  (height = table + block)")

            if is_reachable:
                print("  --> [OK] This position is REACHABLE by the DOBOT arm!")
            else:
                print(f"  --> [!!] OUT OF REACH: {reason}")
            print()

        except ValueError:
            print("  Please enter valid numbers.")

    pause()


def test_8_multi_scene():
    print(SEPARATOR)
    print("  TEST 8 -- Build Your Own Scene and Test It")
    print(SEPARATOR)
    print("  Add any blocks to the table, then ask the robot to find one.")
    print("  Block names: red_block, blue_block, green_block, yellow_block")
    print()

    cam = MockCamera(width=640, height=480)
    cam.open()

    while True:
        block = input("  Add a block (type name) or type DONE to finish: ").strip()
        if block.upper() == "DONE":
            break
        try:
            cx = float(input(f"    Center X pixel (0-640) for {block}: "))
            cy = float(input(f"    Center Y pixel (0-480) for {block}: "))
            cam.add_object(block, center=(cx, cy), size=(60, 60))
            print(f"    [OK] Added {block} at pixel ({cx}, {cy})")
        except ValueError:
            print("    Invalid input, skipping.")

    query = input("\n  Which block should the robot pick? (e.g. red_block): ").strip()

    pipeline = VisionPipeline(camera=cam, stability_samples=3)
    print(f"\n  Running 4 frames to stabilize...\n")

    for i in range(1, 5):
        target, _ = pipeline.capture_and_process(query)
        print(f"  Frame #{i}: {target.status.value}")

    print()
    print("  FINAL RESULT:")
    print_target(target)

    if target.valid:
        print()
        print("  JSON Contract sent to Member 4:")
        print(json.dumps(target.model_dump(), indent=4))

    cam.release()
    pause()


def test_9_dynamic_z():
    print(SEPARATOR)
    print("  TEST 9 -- Dynamic Z from Bounding Box Apparent Size")
    print(SEPARATOR)
    print("  The ZEstimator uses perspective projection:")
    print("    block_z = cam_height - (cam_height - table_z) * (ref_px / apparent_px)")
    print("  Bigger bounding box  => block is CLOSER to camera => HIGHER Z")
    print("  Smaller bounding box => block is FARTHER away      => LOWER Z")
    print()

    # Camera at 500mm above base, table at -50mm, reference block = 60px side on table
    z_est = ZEstimator(
        camera_height_mm=500.0,
        table_z_mm=-50.0,
        ref_bbox_side_px=60.0,   # block flat on table produces ~60x60 px bounding box
        alpha=1.0,                # no smoothing for this demo
    )

    calib = TabletopCalibration.create_default()
    transformer = CoordinateTransformer(
        calibration=calib,
        table_z_mm=-50.0,
        z_estimator=z_est,
    )

    scenarios = [
        ("Block on table       (60x60 px)", BoundingBox(x1=290, y1=220, x2=350, y2=280)),
        ("Block elevated ~25mm (70x70 px)", BoundingBox(x1=285, y1=215, x2=355, y2=285)),
        ("Block elevated ~50mm (80x80 px)", BoundingBox(x1=280, y1=210, x2=360, y2=290)),
        ("Block far/small      (45x45 px)", BoundingBox(x1=298, y1=228, x2=343, y2=273)),
    ]

    print(f"  {'Scenario':<38} {'BBox px':>8}  {'Z (mm)':>9}  {'Z-mode':>8}")
    print("  " + "-" * 70)
    for label, bbox in scenarios:
        z_est.reset()   # fresh smoother each scenario for clean comparison
        pixel = Point2D(x=float((bbox.x1 + bbox.x2) / 2), y=float((bbox.y1 + bbox.y2) / 2))
        point_3d, reachable, reason, z_mode = transformer.pixel_to_robot_3d(
            pixel=pixel,
            class_name="red_block",
            bbox=bbox,
        )
        avg_side = (bbox.width + bbox.height) / 2.0
        reach_str = "[OK]" if reachable else "[!!]"
        print(f"  {label:<38} {avg_side:>8.1f}  {point_3d.z:>+9.2f}  {z_mode:>8}  {reach_str}")

    print()
    print("  Notice Z increases as bounding box grows (block moves up toward camera).")
    print("  The fixed-physics fallback always gives Z = table_z + block_height = -25.0 mm.")
    pause()


def main_menu():
    tests = {
        "1": ("Single red block -- should go VALID after 3 frames",     test_1_single_block),
        "2": ("You choose which block is on table",                      test_2_choose_color),
        "3": ("Two red blocks -- AMBIGUOUS rejection",                   test_3_ambiguous),
        "4": ("Wrong object requested -- NOT FOUND",                     test_4_not_found),
        "5": ("Too small block -- LOW CONFIDENCE / NOT DETECTED",        test_5_low_confidence),
        "6": ("Jitter noise -- UNSTABLE result",                         test_6_stability_jitter),
        "7": ("Enter pixel coordinate, see robot mm output",             test_7_calibration),
        "8": ("Build your own scene and test it",                        test_8_multi_scene),
        "9": ("Dynamic Z from bounding box size (perspective projection)", test_9_dynamic_z),
        "q": ("Quit",                                                    None),
    }

    while True:
        print()
        print(SEPARATOR)
        print("  MEMBER 3 -- SELF-TEST MENU")
        print(SEPARATOR)
        for key, (label, _) in tests.items():
            print(f"  [{key}]  {label}")
        print(SEPARATOR)
        choice = input("  Your choice: ").strip().lower()

        if choice == "q":
            print("  Goodbye!")
            break
        elif choice in tests and tests[choice][1] is not None:
            tests[choice][1]()
        else:
            print("  [!] Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
