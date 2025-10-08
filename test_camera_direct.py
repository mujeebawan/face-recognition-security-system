#!/usr/bin/env python3
"""
Direct camera connection test
"""
import cv2
import sys

# Test RTSP URL with URL-encoded password
rtsp_url = "rtsp://admin:Mujeeb%40321@192.168.1.64:554/Streaming/Channels/102"

print(f"Testing camera connection...")
print(f"URL: {rtsp_url}")

try:
    # Try to connect to camera
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if cap.isOpened():
        print("✓ Camera opened successfully")

        # Try to read a frame
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✓ Successfully read frame - Resolution: {w}x{h}")

            # Save test frame
            cv2.imwrite("/tmp/test_camera_frame.jpg", frame)
            print(f"✓ Test frame saved to /tmp/test_camera_frame.jpg")

            cap.release()
            sys.exit(0)
        else:
            print("✗ Camera opened but cannot read frames")
            cap.release()
            sys.exit(1)
    else:
        print("✗ Failed to open camera stream")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
