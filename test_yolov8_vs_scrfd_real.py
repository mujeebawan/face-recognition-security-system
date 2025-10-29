"""
YOLOv8 + ArcFace vs SCRFD + ArcFace - Real Image Test

Tests both pipelines with real face images for accurate comparison
"""

import asyncio
import cv2
import numpy as np
import time
import os
from typing import List, Dict, Optional
import logging

# Import models
from app.core.recognizer import FaceRecognizer
from app.core.multi_agent.models.yolov8_detector import YOLOv8FaceDetector

logging.basicConfig(level=logging.WARNING)  # Reduce log verbosity
logger = logging.getLogger(__name__)


class YOLOv8ArcFacePipeline:
    """Two-stage pipeline: YOLOv8 detection → ArcFace recognition"""

    def __init__(self):
        self.yolo_detector = None
        self.arcface_recognizer = None

    async def initialize(self):
        """Initialize both models"""
        print("Initializing YOLOv8 + ArcFace pipeline...")
        start = time.time()

        self.yolo_detector = YOLOv8FaceDetector(stream_id=0)
        await self.yolo_detector.initialize()

        self.arcface_recognizer = FaceRecognizer()

        print(f"✓ Initialized in {time.time() - start:.2f}s")

    async def detect_and_recognize(self, image: np.ndarray) -> Dict:
        """Run full pipeline"""
        start = time.time()

        # Stage 1: YOLOv8 Detection
        detect_start = time.time()
        yolo_result = await self.yolo_detector.infer(image)
        detect_time = (time.time() - detect_start) * 1000

        if yolo_result.bbox is None:
            return {
                'bbox': None,
                'detect_time': detect_time,
                'recognize_time': 0.0,
                'total_time': (time.time() - start) * 1000,
                'pipeline': 'YOLOv8+ArcFace',
                'detected': False
            }

        # Stage 2: Crop face region with padding
        x, y, w, h = yolo_result.bbox
        padding = int(max(w, h) * 0.15)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        face_crop = image[y1:y2, x1:x2]

        # Stage 3: ArcFace Recognition
        recognize_start = time.time()
        arcface_result = self.arcface_recognizer.extract_embedding(face_crop)
        recognize_time = (time.time() - recognize_start) * 1000

        if arcface_result is None:
            return {
                'bbox': yolo_result.bbox,
                'detect_time': detect_time,
                'recognize_time': recognize_time,
                'total_time': (time.time() - start) * 1000,
                'pipeline': 'YOLOv8+ArcFace',
                'detected': True,
                'recognized': False
            }

        return {
            'bbox': yolo_result.bbox,
            'embedding': arcface_result.embedding,
            'detect_time': detect_time,
            'recognize_time': recognize_time,
            'total_time': (time.time() - start) * 1000,
            'pipeline': 'YOLOv8+ArcFace',
            'detected': True,
            'recognized': True,
            'yolo_conf': yolo_result.confidence,
            'arcface_conf': arcface_result.confidence
        }


class SCRFDArcFacePipeline:
    """Single-stage pipeline: SCRFD + ArcFace (bundled)"""

    def __init__(self):
        self.recognizer = None

    async def initialize(self):
        """Initialize InsightFace"""
        print("Initializing SCRFD + ArcFace pipeline...")
        start = time.time()
        self.recognizer = FaceRecognizer()
        print(f"✓ Initialized in {time.time() - start:.2f}s")

    async def detect_and_recognize(self, image: np.ndarray) -> Dict:
        """Run full pipeline"""
        start = time.time()

        result = self.recognizer.extract_embedding(image)

        if result is None:
            return {
                'bbox': None,
                'total_time': (time.time() - start) * 1000,
                'pipeline': 'SCRFD+ArcFace',
                'detected': False,
                'recognized': False
            }

        return {
            'bbox': result.bbox,
            'embedding': result.embedding,
            'total_time': (time.time() - start) * 1000,
            'pipeline': 'SCRFD+ArcFace',
            'detected': True,
            'recognized': True,
            'det_confidence': result.confidence
        }


async def test_with_real_image(image_path: str, num_iterations: int = 20):
    """Test both pipelines with a real face image"""

    print("=" * 80)
    print("YOLOV8 vs SCRFD - REAL IMAGE COMPARISON")
    print("=" * 80)

    # Load image
    print(f"\n1. Loading image: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    print(f"   ✓ Image loaded: {image.shape}")

    # Initialize pipelines
    print("\n2. Initializing pipelines...")
    p1 = YOLOv8ArcFacePipeline()
    await p1.initialize()

    p2 = SCRFDArcFacePipeline()
    await p2.initialize()

    # Warm-up
    print("\n3. Running warm-up...")
    _ = await p1.detect_and_recognize(image)
    _ = await p2.detect_and_recognize(image)
    print("   ✓ Warm-up complete")

    # Benchmark Pipeline 1
    print(f"\n4. Benchmarking YOLOv8 + ArcFace ({num_iterations} iterations)...")
    p1_results = []
    for i in range(num_iterations):
        result = await p1.detect_and_recognize(image)
        p1_results.append(result)
        if i % 5 == 0:
            status = "✓ Detected" if result.get('detected') else "✗ No face"
            print(f"   Iteration {i+1}: {result['total_time']:.1f}ms - {status}")

    # Benchmark Pipeline 2
    print(f"\n5. Benchmarking SCRFD + ArcFace ({num_iterations} iterations)...")
    p2_results = []
    for i in range(num_iterations):
        result = await p2.detect_and_recognize(image)
        p2_results.append(result)
        if i % 5 == 0:
            status = "✓ Detected" if result.get('detected') else "✗ No face"
            print(f"   Iteration {i+1}: {result['total_time']:.1f}ms - {status}")

    # Calculate statistics
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE RESULTS")
    print("=" * 80)

    # Pipeline 1
    p1_times = [r['total_time'] for r in p1_results]
    p1_detected = sum(1 for r in p1_results if r.get('detected', False))
    p1_recognized = sum(1 for r in p1_results if r.get('recognized', False))

    print("\n🔹 YOLOv8 + ArcFace (Two-Stage)")
    print(f"  Total Time:     {np.mean(p1_times):.2f}ms ± {np.std(p1_times):.2f}ms")
    print(f"  Min/Max:        {np.min(p1_times):.2f}ms / {np.max(p1_times):.2f}ms")
    print(f"  Detection Rate: {p1_detected}/{num_iterations} ({p1_detected/num_iterations*100:.1f}%)")
    print(f"  Recognition:    {p1_recognized}/{num_iterations} ({p1_recognized/num_iterations*100:.1f}%)")

    if p1_detected > 0:
        sample = next(r for r in p1_results if r.get('detected'))
        if 'detect_time' in sample:
            detect_times = [r.get('detect_time', 0) for r in p1_results if r.get('detected')]
            recognize_times = [r.get('recognize_time', 0) for r in p1_results if r.get('detected')]
            print(f"  Breakdown:")
            print(f"    - Detection:   {np.mean(detect_times):.2f}ms ± {np.std(detect_times):.2f}ms")
            print(f"    - Recognition: {np.mean(recognize_times):.2f}ms ± {np.std(recognize_times):.2f}ms")

    # Pipeline 2
    p2_times = [r['total_time'] for r in p2_results]
    p2_detected = sum(1 for r in p2_results if r.get('detected', False))
    p2_recognized = sum(1 for r in p2_results if r.get('recognized', False))

    print(f"\n🔹 SCRFD + ArcFace (Bundled)")
    print(f"  Total Time:     {np.mean(p2_times):.2f}ms ± {np.std(p2_times):.2f}ms")
    print(f"  Min/Max:        {np.min(p2_times):.2f}ms / {np.max(p2_times):.2f}ms")
    print(f"  Detection Rate: {p2_detected}/{num_iterations} ({p2_detected/num_iterations*100:.1f}%)")
    print(f"  Recognition:    {p2_recognized}/{num_iterations} ({p2_recognized/num_iterations*100:.1f}%)")

    # Comparison
    print("\n" + "=" * 80)
    print("⚖️  HEAD-TO-HEAD COMPARISON")
    print("=" * 80)

    faster = "SCRFD+ArcFace" if np.mean(p2_times) < np.mean(p1_times) else "YOLOv8+ArcFace"
    speedup = max(np.mean(p1_times), np.mean(p2_times)) / min(np.mean(p1_times), np.mean(p2_times))
    time_diff = abs(np.mean(p1_times) - np.mean(p2_times))

    print(f"\n⏱️  Speed:")
    print(f"  Winner:     {faster}")
    print(f"  Speedup:    {speedup:.2f}x faster")
    print(f"  Difference: {time_diff:.2f}ms")

    print(f"\n🎯 Detection Accuracy:")
    print(f"  YOLOv8: {p1_detected}/{num_iterations} ({p1_detected/num_iterations*100:.1f}%)")
    print(f"  SCRFD:  {p2_detected}/{num_iterations} ({p2_detected/num_iterations*100:.1f}%)")

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    if speedup >= 1.5 and faster == "SCRFD+ArcFace":
        print("\n✅ **Use SCRFD + ArcFace for Production**")
        print(f"   - {speedup:.1f}x faster than YOLOv8")
        print("   - Optimized end-to-end pipeline")
        print("   - TensorRT acceleration")
        print("   - Battle-tested for face recognition")
    elif speedup >= 1.5 and faster == "YOLOv8+ArcFace":
        print("\n✅ **YOLOv8 is surprisingly faster!**")
        print(f"   - {speedup:.1f}x faster than SCRFD")
        print("   - More flexible pipeline")
        print("   - Good for experimentation")
    else:
        print("\n≈ **Similar Performance**")
        print("   - Both pipelines perform comparably")
        print("   - Choose based on requirements:")
        print("     • SCRFD: Production-ready, integrated")
        print("     • YOLOv8: Flexible, experimental")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

    return {
        'p1_avg': np.mean(p1_times),
        'p2_avg': np.mean(p2_times),
        'speedup': speedup,
        'winner': faster
    }


async def main():
    """Main test runner"""

    # Find test images
    test_images = [
        "data/images/16202-3214687-5_094b3fd2-5c28-4f5e-83d8-46dcc6dfce2f.jpeg",
        "data/images/37101-1321636-7_23454 B.jpg",
        "data/alert_snapshots/alert_235_20251015_162619.jpg"
    ]

    # Test with first available image
    for img_path in test_images:
        if os.path.exists(img_path):
            await test_with_real_image(img_path, num_iterations=30)
            break
    else:
        print("❌ No test images found!")


if __name__ == "__main__":
    asyncio.run(main())
