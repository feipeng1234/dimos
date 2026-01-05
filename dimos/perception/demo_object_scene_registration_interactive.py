#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive CLI for ObjectSceneRegistration with hosted service control.

This demo allows you to:
- List detected objects in real-time
- Trigger full pipeline (/process: SAM3 -> SAM3D -> FoundationPose) with box or text prompts
- Trigger fast pipeline (/grasp: SAM3 -> GraspGen) for quick grasp generation
- Toggle auto mesh/pose enhancement on/off

Usage:
    python -m dimos.perception.demo_object_scene_registration_interactive \\
        --service-url http://localhost:8080 \\
        [--auto]  # Enable auto enhancement (slow, processes all detections)

Commands:
    list                          - Show current detections
    auto on|off                   - Toggle auto mesh/pose enhancement
    process <idx|id:N> [prompt]   - Run full pipeline (mesh+pose)
    grasp <idx|id:N> [prompt]     - Run fast pipeline (grasps only)
    quit                          - Exit
"""

import argparse
import shlex

from dimos.core.blueprints import autoconnect
from dimos.core.global_config import GlobalConfig
from dimos.perception.object_scene_registration import (
    ObjectSceneRegistrationModule,
    object_scene_registration_module,
)
from dimos.protocol import pubsub
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _parse_target(tok: str) -> dict:
    """Parse target selection token (index or track_id)."""
    if tok.startswith("id:"):
        return {"track_id": int(tok[3:])}
    return {"index": int(tok)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive ObjectSceneRegistration with hosted service control"
    )
    ap.add_argument(
        "--service-url",
        default="http://localhost:8080",
        help="Hosted mesh/pose service URL (default: http://localhost:8080)",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Enable auto /process enhancement for all detections (slow)",
    )
    ap.add_argument(
        "--image-topic",
        default="/camera/color/image_raw/compressed",
        help="ROS compressed color image topic",
    )
    ap.add_argument(
        "--depth-topic",
        default="/camera/aligned_depth_to_color/image_raw/compressedDepth",
        help="ROS compressed depth image topic",
    )
    args = ap.parse_args()

    logger.info("Starting interactive ObjectSceneRegistration demo...")
    logger.info(f"Hosted service: {args.service_url}")
    logger.info(f"Auto enhancement: {'enabled' if args.auto else 'disabled'}")

    pubsub.lcm.autoconf()

    blueprint = autoconnect(
        object_scene_registration_module(
            image_topic=args.image_topic,
            depth_topic=args.depth_topic,
            mesh_pose_service_url=args.service_url,
            auto_mesh_pose=args.auto,
            mesh_pose_use_box_prompt=True,  # Default: ignore YOLO-E labels
        )
    )

    coordinator = blueprint.build(global_config=GlobalConfig(n_dask_workers=1))

    try:
        osr = coordinator.get_instance(ObjectSceneRegistrationModule)

        print("\n" + "=" * 70)
        print("Interactive ObjectSceneRegistration")
        print("=" * 70)
        print(osr.io())
        print("\n" + "=" * 70)
        print("Commands:")
        print("  list                          - Show current detections")
        print("  auto on|off                   - Toggle auto mesh/pose enhancement")
        print("  process <idx|id:N> [prompt]   - Run full pipeline (mesh+pose)")
        print("  grasp <idx|id:N> [prompt]     - Run fast pipeline (grasps only)")
        print("  quit                          - Exit")
        print("=" * 70)
        print("\nExamples:")
        print("  list")
        print("  process 0                     # Box-only for detection at index 0")
        print("  process id:5 red coffee mug   # Text prompt for track_id 5")
        print("  grasp 0                       # Fast grasps for detection 0")
        print("  auto off                      # Disable auto enhancement")
        print("=" * 70 + "\n")

        while True:
            try:
                line = input("osr> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not line:
                continue

            try:
                parts = shlex.split(line)
            except ValueError as e:
                print(f"Parse error: {e}")
                continue

            cmd = parts[0].lower()

            if cmd in {"quit", "exit", "q"}:
                break

            if cmd == "list":
                try:
                    dets = osr.get_latest_detections()
                    if not dets:
                        print("No detections available yet")
                    else:
                        print(f"\nFound {len(dets)} detection(s):")
                        for d in dets:
                            print(
                                f"  [{d['index']}] track_id={d['track_id']:3d} "
                                f"conf={d['confidence']:.2f} "
                                f"name={d['name']:15s} "
                                f"bbox={[int(x) for x in d['bbox']]}"
                            )
                        print()
                except Exception as e:
                    print(f"Error: {e}")
                continue

            if cmd == "auto" and len(parts) == 2:
                try:
                    enabled = parts[1].lower() == "on"
                    osr.set_auto_mesh_pose(enabled)
                    print(f"Auto mesh/pose enhancement: {'enabled' if enabled else 'disabled'}")
                except Exception as e:
                    print(f"Error: {e}")
                continue

            if cmd == "process" and len(parts) >= 2:
                try:
                    target = _parse_target(parts[1])
                    prompt = " ".join(parts[2:]) if len(parts) > 2 else None
                    
                    print(f"Running full pipeline (mesh+pose)...")
                    if prompt:
                        print(f"  Using text prompt: '{prompt}'")
                    else:
                        print(f"  Using box-only prompting (no label)")
                    
                    res = osr.run_hosted_pipeline("process", prompt=prompt, **target)
                    
                    print("\nResult:")
                    print(f"  Mesh: {res.get('mesh_obj_bytes', 0)} bytes")
                    if res.get("mesh_dimensions"):
                        dims = res["mesh_dimensions"]
                        print(f"  Dimensions: {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m")
                    if res.get("fp_position"):
                        pos = res["fp_position"]
                        print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    if res.get("fp_orientation"):
                        ori = res["fp_orientation"]
                        print(f"  Orientation (xyzw): [{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]")
                    print()
                except Exception as e:
                    print(f"Error: {e}")
                continue

            if cmd == "grasp" and len(parts) >= 2:
                try:
                    target = _parse_target(parts[1])
                    prompt = " ".join(parts[2:]) if len(parts) > 2 else None
                    
                    print(f"Running fast pipeline (grasps only)...")
                    if prompt:
                        print(f"  Using text prompt: '{prompt}'")
                    else:
                        print(f"  Using box-only prompting (no label)")
                    
                    res = osr.run_hosted_pipeline("grasp", prompt=prompt, **target)
                    
                    grasps = res.get("grasps", [])
                    print(f"\nGenerated {len(grasps)} grasp(s)")
                    if grasps:
                        best_score = max((g.get("score", 0.0) for g in grasps), default=0.0)
                        collision_free = sum(1 for g in grasps if g.get("collision_free", False))
                        print(f"  Best score: {best_score:.3f}")
                        print(f"  Collision-free: {collision_free}/{len(grasps)}")
                        print(f"  Inference time: {res.get('inference_time_ms', 0):.0f} ms")
                    print()
                except Exception as e:
                    print(f"Error: {e}")
                continue

            print("Unknown command. Type 'quit' to exit or see help above.")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.exception("Error in interactive demo")
        print(f"Fatal error: {e}")
    finally:
        logger.info("Stopping coordinator...")
        coordinator.stop()
        logger.info("Demo completed")


if __name__ == "__main__":
    main()

