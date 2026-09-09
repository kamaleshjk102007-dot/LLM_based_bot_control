"""
Member 3 Perception Pipeline Demonstration.
Simulates camera frames, performs detection, calibration, stability checking,
and produces a validated RobotTarget for Member 4.
"""

import json
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision import (
    MockCamera,
    VisionPipeline,
)


def run():
    print("=" * 70)
    print("MEMBER 3: VISION + SENSORS + PERCEPTION SYSTEM")
    print("Robot 1 — DOBOT Magician Lite")
    print("=" * 70)

    # 1. Initialize Mock Camera with synthetic tabletop scene
    camera = MockCamera(width=640, height=480, jitter_std=0.5)
    camera.open()

    # Scene setup: Red block, Blue block, Green block
    camera.add_object("red_block", center=(320.0, 250.0), size=(60, 60))
    camera.add_object("blue_block", center=(180.0, 180.0), size=(60, 60))
    camera.add_object("green_block", center=(460.0, 320.0), size=(60, 60))

    # 2. Instantiate Vision Pipeline
    pipeline = VisionPipeline(camera=camera, stability_samples=3)

    query = "red_block"
    print(f"\n[Task Request from LLM / Member 2]: 'Pick the {query}'")

    os.makedirs("output", exist_ok=True)

    # Simulate multi-frame acquisition to demonstrate temporal stabilization
    target = None
    debug_frame = None

    for frame_idx in range(1, 5):
        target, debug_frame = pipeline.capture_and_process(requested_class=query, visualize=True)
        print(f"\n--- Frame #{frame_idx} ---")
        print(f"Status:          {target.status.value}")
        print(f"Valid:           {target.valid}")
        print(f"Stability Score: {target.stability_score * 100:.1f}%")
        print(f"Message:         {target.message}")

    if target and target.valid:
        print("\n" + "=" * 70)
        print("[CONTRACT DISPATCH TO MEMBER 4]: Valid Target Ready for Motion Planning")
        print("=" * 70)
        target_dict = target.model_dump()
        print(json.dumps(target_dict, indent=2))

        # Save contract json
        with open("output/member4_target.json", "w") as f:
            json.dump(target_dict, f, indent=2)
        print("\nSaved contract output to: output/member4_target.json")

    if debug_frame is not None:
        cv2.imwrite("output/vision_hud_demo.png", debug_frame)
        print("Saved debug HUD visualization to: output/vision_hud_demo.png")

    # Demonstrate Ambiguity Rejection (Prompt: Pick red block, but 2 red blocks exist)
    print("\n" + "=" * 70)
    print("[AMBIGUITY TEST]: Two red blocks in camera view")
    print("=" * 70)
    camera.add_object("red_block", center=(400.0, 160.0), size=(60, 60))
    pipeline.reset_history()

    ambig_target, ambig_frame = pipeline.capture_and_process(requested_class=query, visualize=True)
    print(f"Status:  {ambig_target.status.value}")
    print(f"Valid:   {ambig_target.valid}")
    print(f"Message: {ambig_target.message}")

    if ambig_frame is not None:
        cv2.imwrite("output/vision_hud_ambiguous.png", ambig_frame)
        print("Saved ambiguity debug HUD to: output/vision_hud_ambiguous.png")

    camera.release()
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run()
