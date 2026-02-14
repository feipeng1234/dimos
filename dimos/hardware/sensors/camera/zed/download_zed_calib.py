#!/usr/bin/env python3
"""Download ZED camera calibration file."""

import subprocess
import sys

serial_number = "15054972"

print(f"Downloading calibration for ZED camera S/N: {serial_number}")
print("This requires internet connection...")

# Try using wget to download calibration
url = f"https://calib.stereolabs.com/?SN={serial_number}"
print(f"\nCalibration URL: {url}")
print("\nOption 1: Download via browser and save to:")
print(f"  /usr/local/zed/settings/SN{serial_number}.conf")

print("\nOption 2: Skip calibration and use default (less accurate):")
print("  Set disable_self_calib=True in ZEDCameraConfig")

print("\nOption 3: Run ZED Depth Viewer if available:")
print(f"  ZED_Depth_Viewer --dc {serial_number}")

sys.exit(0)
