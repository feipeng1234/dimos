#!/usr/bin/env python3
"""Final ZED camera test - use this in your code."""

import pyzed.sl as sl
import numpy as np

def test_zed():
    print("Opening ZED Mini...")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # or ULTRA, QUALITY
    init.coordinate_units = sl.UNIT.METER

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {err}")
        return False

    print(f"✓ Camera opened: {zed.get_camera_information().camera_model}")

    # Capture frames
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    print("\nCapturing 10 frames...")
    for i in range(10):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Get RGB image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            img_data = image.get_data()[:, :, :3]  # RGB

            # Get depth map
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_data = depth.get_data()

            # Filter out invalid depth values
            valid_depth = depth_data[~np.isnan(depth_data) & ~np.isinf(depth_data)]
            if len(valid_depth) > 0:
                print(f"  Frame {i+1}: RGB {img_data.shape}, Depth range: {valid_depth.min():.2f}-{valid_depth.max():.2f}m")
            else:
                print(f"  Frame {i+1}: RGB {img_data.shape}, Depth: initializing...")
        else:
            print(f"  Frame {i+1}: Failed to grab")

    zed.close()
    print("\n✓✓✓ ZED Mini test complete!")
    return True

if __name__ == "__main__":
    test_zed()
