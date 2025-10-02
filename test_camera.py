"""
Quick test script to verify Hikvision camera RTSP connection.
Run this before starting the full application.
"""

import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.camera import CameraHandler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "=" * 60)
    print("Hikvision Camera Connection Test")
    print("=" * 60 + "\n")

    # Test main stream
    print("Testing MAIN stream (high quality)...")
    camera = CameraHandler(use_main_stream=True)
    result = camera.test_connection()

    print("\n📊 Test Results:")
    print(f"  Camera IP: {result['camera_ip']}")
    print(f"  Stream URL: {result['stream_url']}")
    print(f"  Connected: {'✓ YES' if result['connected'] else '✗ NO'}")
    print(f"  Frame Readable: {'✓ YES' if result['frame_readable'] else '✗ NO'}")
    print(f"  Resolution: {result['resolution'] or 'N/A'}")
    print(f"  FPS: {result['fps'] or 'N/A'}")

    if result['error']:
        print(f"  ❌ Error: {result['error']}")

    camera.disconnect()

    # Test sub stream
    print("\n" + "-" * 60)
    print("Testing SUB stream (lower quality)...")
    camera_sub = CameraHandler(use_main_stream=False)
    result_sub = camera_sub.test_connection()

    print("\n📊 Test Results:")
    print(f"  Camera IP: {result_sub['camera_ip']}")
    print(f"  Stream URL: {result_sub['stream_url']}")
    print(f"  Connected: {'✓ YES' if result_sub['connected'] else '✗ NO'}")
    print(f"  Frame Readable: {'✓ YES' if result_sub['frame_readable'] else '✗ NO'}")
    print(f"  Resolution: {result_sub['resolution'] or 'N/A'}")
    print(f"  FPS: {result_sub['fps'] or 'N/A'}")

    if result_sub['error']:
        print(f"  ❌ Error: {result_sub['error']}")

    camera_sub.disconnect()

    print("\n" + "=" * 60)
    if result['connected'] or result_sub['connected']:
        print("✓ Camera connection test PASSED!")
        print("=" * 60 + "\n")
        return 0
    else:
        print("✗ Camera connection test FAILED!")
        print("=" * 60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
