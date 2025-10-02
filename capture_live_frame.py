"""
Capture a live frame from camera and test face detection in real-time.
This script will capture multiple frames and save the best one with faces.
"""

import sys
import os
import cv2
from datetime import datetime

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.camera import CameraHandler
from app.core.detector import FaceDetector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def capture_with_detection():
    print("\n" + "=" * 70)
    print("LIVE FACE DETECTION TEST")
    print("=" * 70 + "\n")
    print("📹 This will capture frames from the camera and detect faces")
    print("   Stand in front of the camera!")
    print()

    # Initialize
    detector = FaceDetector(min_detection_confidence=0.5)
    camera = CameraHandler(use_main_stream=False)  # Use sub-stream for faster processing

    output_dir = "data/live_detection"
    os.makedirs(output_dir, exist_ok=True)

    if not camera.connect():
        print("❌ Failed to connect to camera")
        return

    print("✅ Camera connected")
    print("🔍 Capturing frames and detecting faces...\n")

    best_frame = None
    best_detections = []
    max_faces = 0

    # Capture 20 frames and keep the best one
    for i in range(20):
        ret, frame = camera.read_frame()
        if ret and frame is not None:
            # Detect faces
            detections = detector.detect_faces(frame)
            num_faces = len(detections)

            print(f"Frame {i+1}/20: {num_faces} face(s) detected", end='\r')

            # Keep frame with most faces
            if num_faces > max_faces:
                max_faces = num_faces
                best_frame = frame.copy()
                best_detections = detections

    print("\n")

    camera.disconnect()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if best_frame is not None:
        # Save original frame
        orig_path = os.path.join(output_dir, f"live_original_{timestamp}.jpg")
        cv2.imwrite(orig_path, best_frame)
        print(f"💾 Saved original: {orig_path}")

        # Draw and save detected frame
        if best_detections:
            detected_frame = detector.draw_detections(best_frame, best_detections, draw_landmarks=True)
            det_path = os.path.join(output_dir, f"live_detected_{timestamp}.jpg")
            cv2.imwrite(det_path, detected_frame)
            print(f"💾 Saved with detections: {det_path}")

            print(f"\n✅ DETECTED {len(best_detections)} FACE(S)!")

            for i, det in enumerate(best_detections, 1):
                x, y, w, h = det.bbox
                print(f"   Face {i}:")
                print(f"      Bounding box: ({x}, {y}, {w}, {h})")
                print(f"      Confidence: {det.confidence:.3f}")
                print(f"      Landmarks: {len(det.landmarks) if det.landmarks else 0}")

                # Save cropped face
                face_crop = detector.crop_face(best_frame, det, padding=0.2)
                if face_crop is not None:
                    crop_path = os.path.join(output_dir, f"face_crop_{i}_{timestamp}.jpg")
                    cv2.imwrite(crop_path, face_crop)
                    print(f"      Saved crop: face_crop_{i}_{timestamp}.jpg")
        else:
            print("⚠️  No faces detected in any frame")
            print("   Make sure you're in front of the camera!")

    print("\n" + "=" * 70)
    print(f"📂 Output directory: {output_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    capture_with_detection()
