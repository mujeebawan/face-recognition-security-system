"""
Test face recognition model initialization and embedding extraction.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.recognizer import FaceRecognizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_recognizer():
    print("\n" + "=" * 70)
    print("FACE RECOGNITION MODEL TEST")
    print("=" * 70 + "\n")

    # Initialize recognizer (will download model if needed)
    print("📥 Initializing face recognizer (this may download models)...")
    recognizer = FaceRecognizer()
    print("✓ Model loaded successfully\n")

    # Test on a captured image
    test_dir = "data/test_captures"
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if images:
            test_img = os.path.join(test_dir, images[0])
            print(f"🔍 Testing on: {images[0]}")

            image = cv2.imread(test_img)
            if image is not None:
                result = recognizer.extract_embedding(image)

                if result:
                    print(f"✅ Embedding extracted!")
                    print(f"   Embedding shape: {result.embedding.shape}")
                    print(f"   Embedding dimension: {len(result.embedding)}")
                    print(f"   Detection confidence: {result.confidence:.3f}")
                    print(f"   Bounding box: {result.bbox}")
                    print(f"   Embedding norm: {np.linalg.norm(result.embedding):.3f}")
                else:
                    print("⚠️  No face detected in image")

    print("\n" + "=" * 70)
    print("✓ RECOGNIZER TEST COMPLETE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    import numpy as np
    test_recognizer()
