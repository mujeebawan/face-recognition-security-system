"""
Test face detection on captured camera images.
This will detect faces and save annotated images.
"""

import sys
import os
import cv2

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.detector import FaceDetector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_detection():
    print("\n" + "=" * 70)
    print("FACE DETECTION TEST")
    print("=" * 70 + "\n")

    # Initialize detector
    logger.info("Initializing face detector...")
    detector = FaceDetector(min_detection_confidence=0.5)

    # Test images directory
    test_dir = "data/test_captures"
    output_dir = "data/test_detections"
    os.makedirs(output_dir, exist_ok=True)

    # Get all test images
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        print("   Run capture_test_frame.py first to capture images")
        return

    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]

    if not images:
        print(f"❌ No test images found in {test_dir}")
        return

    print(f"📁 Found {len(images)} test image(s)\n")

    # Process each image
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        print(f"🔍 Processing: {img_name}")

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"   ❌ Failed to read image")
            continue

        height, width = image.shape[:2]
        print(f"   📐 Image size: {width}x{height}")

        # Detect faces
        detections = detector.detect_faces(image)
        print(f"   👤 Detected {len(detections)} face(s)")

        # Print detection details
        for i, det in enumerate(detections, 1):
            x, y, w, h = det.bbox
            print(f"      Face {i}: bbox=({x}, {y}, {w}, {h}), confidence={det.confidence:.3f}")

        # Draw detections
        output_image = detector.draw_detections(image, detections, draw_landmarks=True)

        # Save annotated image
        output_path = os.path.join(output_dir, f"detected_{img_name}")
        cv2.imwrite(output_path, output_image)
        print(f"   💾 Saved to: {output_path}")

        # Also save cropped faces if any
        if detections:
            for i, det in enumerate(detections, 1):
                face_crop = detector.crop_face(image, det, padding=0.2)
                if face_crop is not None:
                    crop_path = os.path.join(output_dir, f"face_{i}_{img_name}")
                    cv2.imwrite(crop_path, face_crop)
                    print(f"   ✂️  Cropped face {i} saved")

        print()

    print("=" * 70)
    print("✅ DETECTION TEST COMPLETE")
    print(f"   📂 Output directory: {output_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_detection()
