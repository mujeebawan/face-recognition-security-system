"""
YOLOv8 + ArcFace vs SCRFD + ArcFace Comparison Test

Compares two face recognition pipelines:
1. SCRFD detection + ArcFace recognition (bundled in InsightFace)
2. YOLOv8 detection + ArcFace recognition (separate pipeline)
"""

import asyncio
import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
import logging

# Import models
from app.core.recognizer import FaceRecognizer
from app.core.multi_agent.models.yolov8_detector import YOLOv8FaceDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv8ArcFacePipeline:
    """
    Two-stage pipeline: YOLOv8 detection → ArcFace recognition
    """

    def __init__(self):
        self.yolo_detector = None
        self.arcface_recognizer = None

    async def initialize(self):
        """Initialize both models"""
        logger.info("Initializing YOLOv8 + ArcFace pipeline...")
        start = time.time()

        # Initialize YOLOv8 detector
        self.yolo_detector = YOLOv8FaceDetector(stream_id=0)
        await self.yolo_detector.initialize()

        # Initialize ArcFace recognizer
        self.arcface_recognizer = FaceRecognizer()

        logger.info(f"✓ Pipeline initialized in {time.time() - start:.2f}s")

    async def detect_and_recognize(
        self,
        image: np.ndarray,
        database_embeddings: Optional[List[np.ndarray]] = None,
        database_persons: Optional[List[Dict]] = None,
        threshold: float = 0.6
    ) -> Dict:
        """
        Run full pipeline: YOLOv8 detection → ArcFace recognition

        Returns:
            Dict with detection + recognition results and timing
        """
        start = time.time()

        # Stage 1: YOLOv8 Detection
        detect_start = time.time()
        yolo_result = await self.yolo_detector.infer(image)
        detect_time = (time.time() - detect_start) * 1000

        if yolo_result.bbox is None:
            # No face detected
            return {
                'person_id': None,
                'person_name': None,
                'confidence': 0.0,
                'embedding': None,
                'bbox': None,
                'detect_time': detect_time,
                'recognize_time': 0.0,
                'total_time': (time.time() - start) * 1000,
                'num_faces': 0,
                'pipeline': 'YOLOv8+ArcFace'
            }

        # Stage 2: Crop face region
        x, y, w, h = yolo_result.bbox
        # Add padding (10%)
        padding = int(max(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        face_crop = image[y1:y2, x1:x2]

        # Stage 3: ArcFace Recognition on cropped face
        recognize_start = time.time()
        arcface_result = self.arcface_recognizer.extract_embedding(face_crop)

        if arcface_result is None:
            recognize_time = (time.time() - recognize_start) * 1000
            return {
                'person_id': None,
                'person_name': None,
                'confidence': 0.0,
                'embedding': None,
                'bbox': yolo_result.bbox,
                'detect_time': detect_time,
                'recognize_time': recognize_time,
                'total_time': (time.time() - start) * 1000,
                'num_faces': 1,
                'pipeline': 'YOLOv8+ArcFace',
                'note': 'YOLOv8 detected, but ArcFace failed on crop'
            }

        # Match against database
        person_id = None
        person_name = None
        similarity = 0.0

        if database_embeddings and database_persons:
            match_idx, similarity = self.arcface_recognizer.match_face(
                arcface_result.embedding,
                database_embeddings,
                threshold=threshold
            )

            if match_idx >= 0:
                person_id = database_persons[match_idx]['id']
                person_name = database_persons[match_idx]['name']

        recognize_time = (time.time() - recognize_start) * 1000
        total_time = (time.time() - start) * 1000

        return {
            'person_id': person_id,
            'person_name': person_name,
            'confidence': float(similarity),
            'embedding': arcface_result.embedding,
            'bbox': yolo_result.bbox,
            'yolo_confidence': yolo_result.confidence,
            'arcface_confidence': arcface_result.confidence,
            'detect_time': detect_time,
            'recognize_time': recognize_time,
            'total_time': total_time,
            'num_faces': yolo_result.metadata.get('num_faces', 1),
            'pipeline': 'YOLOv8+ArcFace'
        }


class SCRFDArcFacePipeline:
    """
    Single-stage pipeline: SCRFD + ArcFace (bundled in InsightFace)
    """

    def __init__(self):
        self.recognizer = None

    async def initialize(self):
        """Initialize InsightFace (SCRFD + ArcFace)"""
        logger.info("Initializing SCRFD + ArcFace pipeline...")
        start = time.time()

        self.recognizer = FaceRecognizer()

        logger.info(f"✓ Pipeline initialized in {time.time() - start:.2f}s")

    async def detect_and_recognize(
        self,
        image: np.ndarray,
        database_embeddings: Optional[List[np.ndarray]] = None,
        database_persons: Optional[List[Dict]] = None,
        threshold: float = 0.6
    ) -> Dict:
        """
        Run full pipeline: SCRFD detection + ArcFace recognition (bundled)

        Returns:
            Dict with detection + recognition results and timing
        """
        start = time.time()

        # InsightFace does both detection and recognition in one call
        result = self.recognizer.extract_embedding(image)

        if result is None:
            return {
                'person_id': None,
                'person_name': None,
                'confidence': 0.0,
                'embedding': None,
                'bbox': None,
                'detect_time': 0.0,  # Bundled, can't separate
                'recognize_time': 0.0,
                'total_time': (time.time() - start) * 1000,
                'num_faces': 0,
                'pipeline': 'SCRFD+ArcFace'
            }

        # Match against database
        person_id = None
        person_name = None
        similarity = 0.0

        if database_embeddings and database_persons:
            match_idx, similarity = self.recognizer.match_face(
                result.embedding,
                database_embeddings,
                threshold=threshold
            )

            if match_idx >= 0:
                person_id = database_persons[match_idx]['id']
                person_name = database_persons[match_idx]['name']

        total_time = (time.time() - start) * 1000

        return {
            'person_id': person_id,
            'person_name': person_name,
            'confidence': float(similarity),
            'embedding': result.embedding,
            'bbox': result.bbox,
            'det_confidence': result.confidence,
            'detect_time': 0.0,  # Can't separate from bundled call
            'recognize_time': 0.0,
            'total_time': total_time,
            'num_faces': 1,
            'pipeline': 'SCRFD+ArcFace'
        }


def create_test_face_image() -> np.ndarray:
    """Create a synthetic face image for testing"""
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200

    center_x, center_y = 320, 240

    # Face (skin tone)
    cv2.ellipse(test_image, (center_x, center_y), (100, 130), 0, 0, 360, (180, 160, 140), -1)

    # Eyes
    cv2.circle(test_image, (center_x - 40, center_y - 30), 20, (255, 255, 255), -1)
    cv2.circle(test_image, (center_x + 40, center_y - 30), 20, (255, 255, 255), -1)
    cv2.circle(test_image, (center_x - 40, center_y - 30), 10, (100, 80, 60), -1)
    cv2.circle(test_image, (center_x + 40, center_y - 30), 10, (100, 80, 60), -1)
    cv2.circle(test_image, (center_x - 40, center_y - 30), 5, (0, 0, 0), -1)
    cv2.circle(test_image, (center_x + 40, center_y - 30), 5, (0, 0, 0), -1)

    # Nose
    pts = np.array([[center_x, center_y], [center_x - 10, center_y + 30], [center_x + 10, center_y + 30]], np.int32)
    cv2.fillPoly(test_image, [pts], (160, 140, 120))

    # Mouth
    cv2.ellipse(test_image, (center_x, center_y + 60), (40, 20), 0, 0, 180, (120, 60, 60), 3)

    # Hair
    cv2.ellipse(test_image, (center_x, center_y - 80), (110, 60), 0, 180, 360, (60, 40, 20), -1)

    return test_image


async def run_comparison_test(num_iterations: int = 20):
    """
    Run comprehensive comparison test between two pipelines
    """
    print("=" * 80)
    print("YOLOV8 + ARCFACE vs SCRFD + ARCFACE COMPARISON TEST")
    print("=" * 80)

    # Create test image
    print("\n1. Creating test image...")
    test_image = create_test_face_image()
    print(f"   ✓ Created synthetic face image: {test_image.shape}")

    # Initialize Pipeline 1: YOLOv8 + ArcFace
    print("\n2. Initializing Pipeline 1: YOLOv8 + ArcFace...")
    pipeline1 = YOLOv8ArcFacePipeline()
    await pipeline1.initialize()

    # Initialize Pipeline 2: SCRFD + ArcFace
    print("\n3. Initializing Pipeline 2: SCRFD + ArcFace...")
    pipeline2 = SCRFDArcFacePipeline()
    await pipeline2.initialize()

    # Warm-up runs
    print("\n4. Running warm-up iterations...")
    _ = await pipeline1.detect_and_recognize(test_image)
    _ = await pipeline2.detect_and_recognize(test_image)
    print("   ✓ Warm-up complete")

    # Benchmark Pipeline 1: YOLOv8 + ArcFace
    print(f"\n5. Benchmarking Pipeline 1: YOLOv8 + ArcFace ({num_iterations} iterations)...")
    p1_results = []

    for i in range(num_iterations):
        result = await pipeline1.detect_and_recognize(test_image)
        p1_results.append(result)

        if i % 5 == 0:
            print(f"   Iteration {i+1}: {result['total_time']:.1f}ms "
                  f"(detect: {result['detect_time']:.1f}ms, "
                  f"recognize: {result['recognize_time']:.1f}ms)")

    # Benchmark Pipeline 2: SCRFD + ArcFace
    print(f"\n6. Benchmarking Pipeline 2: SCRFD + ArcFace ({num_iterations} iterations)...")
    p2_results = []

    for i in range(num_iterations):
        result = await pipeline2.detect_and_recognize(test_image)
        p2_results.append(result)

        if i % 5 == 0:
            print(f"   Iteration {i+1}: {result['total_time']:.1f}ms")

    # Calculate statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)

    # Pipeline 1 stats
    p1_times = [r['total_time'] for r in p1_results]
    p1_detect_times = [r['detect_time'] for r in p1_results]
    p1_recognize_times = [r['recognize_time'] for r in p1_results]

    print("\n📊 Pipeline 1: YOLOv8 + ArcFace (Two-Stage)")
    print("-" * 80)
    print(f"Total Time:")
    print(f"  Average: {np.mean(p1_times):.2f}ms ± {np.std(p1_times):.2f}ms")
    print(f"  Min: {np.min(p1_times):.2f}ms")
    print(f"  Max: {np.max(p1_times):.2f}ms")
    print(f"  Median: {np.median(p1_times):.2f}ms")
    print(f"\nBreakdown:")
    print(f"  YOLOv8 Detection: {np.mean(p1_detect_times):.2f}ms ± {np.std(p1_detect_times):.2f}ms")
    print(f"  ArcFace Recognition: {np.mean(p1_recognize_times):.2f}ms ± {np.std(p1_recognize_times):.2f}ms")

    # Pipeline 2 stats
    p2_times = [r['total_time'] for r in p2_results]

    print(f"\n📊 Pipeline 2: SCRFD + ArcFace (Bundled - InsightFace)")
    print("-" * 80)
    print(f"Total Time:")
    print(f"  Average: {np.mean(p2_times):.2f}ms ± {np.std(p2_times):.2f}ms")
    print(f"  Min: {np.min(p2_times):.2f}ms")
    print(f"  Max: {np.max(p2_times):.2f}ms")
    print(f"  Median: {np.median(p2_times):.2f}ms")
    print(f"\nNote: Detection and recognition times are bundled in InsightFace")

    # Comparison
    print("\n" + "=" * 80)
    print("⚖️  COMPARISON")
    print("=" * 80)

    speedup = np.mean(p2_times) / np.mean(p1_times)
    faster_pipeline = "YOLOv8+ArcFace" if speedup > 1 else "SCRFD+ArcFace"
    speed_diff = abs(np.mean(p1_times) - np.mean(p2_times))

    print(f"\nSpeed:")
    print(f"  YOLOv8+ArcFace:   {np.mean(p1_times):.2f}ms average")
    print(f"  SCRFD+ArcFace:    {np.mean(p2_times):.2f}ms average")
    print(f"  Difference:       {speed_diff:.2f}ms")
    print(f"  Faster Pipeline:  {faster_pipeline} ({abs(speedup - 1) * 100:.1f}% faster)")

    # Detection comparison
    print(f"\nDetection:")
    p1_detected = sum(1 for r in p1_results if r['bbox'] is not None)
    p2_detected = sum(1 for r in p2_results if r['bbox'] is not None)

    print(f"  YOLOv8 Detection Rate:  {p1_detected}/{num_iterations} ({p1_detected/num_iterations*100:.1f}%)")
    print(f"  SCRFD Detection Rate:   {p2_detected}/{num_iterations} ({p2_detected/num_iterations*100:.1f}%)")

    # Architecture comparison
    print("\n" + "=" * 80)
    print("🏗️  ARCHITECTURE COMPARISON")
    print("=" * 80)

    print("\nPipeline 1: YOLOv8 + ArcFace (Two-Stage)")
    print("  ✓ YOLOv8 (Ultralytics) for face detection")
    print("  ✓ ArcFace (InsightFace) for face recognition")
    print("  ✓ Separate models, more flexibility")
    print("  ✓ Can swap detection model easily")
    print("  ✓ Explicit pipeline control")
    print(f"  ⏱  Average: {np.mean(p1_times):.2f}ms")

    print("\nPipeline 2: SCRFD + ArcFace (Bundled)")
    print("  ✓ SCRFD detector (via InsightFace buffalo_l)")
    print("  ✓ ArcFace recognition (via InsightFace buffalo_l)")
    print("  ✓ Integrated, optimized for end-to-end")
    print("  ✓ TensorRT acceleration")
    print("  ✓ Simpler API, less code")
    print(f"  ⏱  Average: {np.mean(p2_times):.2f}ms")

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    if speedup > 1.1:
        print("\n✓ YOLOv8 + ArcFace is FASTER")
        print("  - Better for real-time applications")
        print("  - Good detection-recognition separation")
    elif speedup < 0.9:
        print("\n✓ SCRFD + ArcFace is FASTER")
        print("  - Better optimized end-to-end pipeline")
        print("  - TensorRT acceleration fully utilized")
        print("  - Recommended for production")
    else:
        print("\n≈ Both pipelines have SIMILAR performance")
        print("  - Choose based on your requirements:")
        print("    • YOLOv8: More flexibility, separate control")
        print("    • SCRFD: Simpler, integrated, production-ready")

    print("\n" + "=" * 80)
    print("✅ COMPARISON TEST COMPLETE")
    print("=" * 80)

    return {
        'pipeline1': {
            'name': 'YOLOv8+ArcFace',
            'avg_time': np.mean(p1_times),
            'std_time': np.std(p1_times),
            'detection_rate': p1_detected / num_iterations
        },
        'pipeline2': {
            'name': 'SCRFD+ArcFace',
            'avg_time': np.mean(p2_times),
            'std_time': np.std(p2_times),
            'detection_rate': p2_detected / num_iterations
        },
        'speedup': speedup
    }


if __name__ == "__main__":
    asyncio.run(run_comparison_test(num_iterations=20))
