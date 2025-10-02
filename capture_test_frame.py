"""
Visual camera test - captures frames from both streams and saves them.
This proves the camera is actually working and accessible.
"""

import sys
import os
import cv2
from datetime import datetime

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.camera import CameraHandler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def capture_frames():
    """Capture and save frames from both camera streams"""

    print("\n" + "=" * 70)
    print("VISUAL CAMERA TEST - Capturing Actual Frames")
    print("=" * 70 + "\n")

    # Create output directory
    output_dir = "data/test_captures"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Test 1: Main Stream (High Quality)
    print("📹 Capturing from MAIN STREAM (High Quality - 2560x1440)...")
    camera_main = CameraHandler(use_main_stream=True)

    if camera_main.connect():
        print("   ✓ Connected to main stream")

        # Capture 5 frames
        for i in range(5):
            ret, frame = camera_main.read_frame()
            if ret and frame is not None:
                if i == 2:  # Save the 3rd frame (middle one)
                    filename = f"{output_dir}/main_stream_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    height, width = frame.shape[:2]
                    print(f"   ✓ Captured frame {i+1}/5 - Resolution: {width}x{height}")
                    print(f"   💾 Saved to: {filename}")
            else:
                print(f"   ✗ Failed to capture frame {i+1}")

        camera_main.disconnect()
    else:
        print("   ✗ Failed to connect to main stream")

    print()

    # Test 2: Sub Stream (Lower Quality)
    print("📹 Capturing from SUB STREAM (Lower Quality - 704x576)...")
    camera_sub = CameraHandler(use_main_stream=False)

    if camera_sub.connect():
        print("   ✓ Connected to sub stream")

        # Capture 5 frames
        for i in range(5):
            ret, frame = camera_sub.read_frame()
            if ret and frame is not None:
                if i == 2:  # Save the 3rd frame
                    filename = f"{output_dir}/sub_stream_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    height, width = frame.shape[:2]
                    print(f"   ✓ Captured frame {i+1}/5 - Resolution: {width}x{height}")
                    print(f"   💾 Saved to: {filename}")
            else:
                print(f"   ✗ Failed to capture frame {i+1}")

        camera_sub.disconnect()
    else:
        print("   ✗ Failed to connect to sub stream")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE - Check the captured images:")
    print(f"   📁 Location: {output_dir}/")
    print(f"   📷 Main stream: main_stream_{timestamp}.jpg")
    print(f"   📷 Sub stream: sub_stream_{timestamp}.jpg")
    print("=" * 70 + "\n")

    # List all files in output directory
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if files:
            print("📂 All captured images:")
            for f in sorted(files):
                filepath = os.path.join(output_dir, f)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"   - {f} ({size_kb:.1f} KB)")
        print()

if __name__ == "__main__":
    capture_frames()
