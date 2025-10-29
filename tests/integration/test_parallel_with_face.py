"""
Multi-Model Test with Real Face Image

Tests parallel inference with a real face image using OpenCV's built-in sample
"""

import asyncio
import cv2
import numpy as np
import time
import os
from app.core.multi_agent.engine import ParallelInferenceEngine
from app.core.multi_agent.models.arcface_model import ArcFaceModel
from app.core.multi_agent.models.yolov8_detector import YOLOv8FaceDetector
from app.core.multi_agent.models.adaface_model import AdaFaceModel


async def main():
    print("=" * 70)
    print("PARALLEL INFERENCE TEST WITH REAL FACE")
    print("=" * 70)

    # Create test image with face
    print("\n1. Creating test image with face...")

    # Create synthetic face-like image (more realistic)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light background

    # Draw realistic face
    center_x, center_y = 320, 240

    # Face (skin tone)
    cv2.ellipse(test_image, (center_x, center_y), (100, 130), 0, 0, 360, (180, 160, 140), -1)

    # Eyes
    cv2.circle(test_image, (center_x - 40, center_y - 30), 20, (255, 255, 255), -1)  # Left eye white
    cv2.circle(test_image, (center_x + 40, center_y - 30), 20, (255, 255, 255), -1)  # Right eye white
    cv2.circle(test_image, (center_x - 40, center_y - 30), 10, (100, 80, 60), -1)  # Left iris
    cv2.circle(test_image, (center_x + 40, center_y - 30), 10, (100, 80, 60), -1)  # Right iris
    cv2.circle(test_image, (center_x - 40, center_y - 30), 5, (0, 0, 0), -1)  # Left pupil
    cv2.circle(test_image, (center_x + 40, center_y - 30), 5, (0, 0, 0), -1)  # Right pupil

    # Nose
    pts = np.array([[center_x, center_y], [center_x - 10, center_y + 30], [center_x + 10, center_y + 30]], np.int32)
    cv2.fillPoly(test_image, [pts], (160, 140, 120))

    # Mouth
    cv2.ellipse(test_image, (center_x, center_y + 60), (40, 20), 0, 0, 180, (120, 60, 60), 3)

    # Hair
    cv2.ellipse(test_image, (center_x, center_y - 80), (110, 60), 0, 180, 360, (60, 40, 20), -1)

    print("   ✓ Created synthetic face image (480x640)")
    print(f"   Image shape: {test_image.shape}, dtype: {test_image.dtype}")

    # Initialize engine
    print("\n2. Initializing Parallel Inference Engine...")
    engine = ParallelInferenceEngine()

    # Register models (only those that work without torch)
    print("\n3. Registering models...")
    print("   - ArcFace (Stream 1) - TensorRT GPU")
    arcface = ArcFaceModel(stream_id=1)
    engine.register_model(arcface)

    print("   - YOLOv8-Face (Stream 0) - Detection")
    yolo = YOLOv8FaceDetector(stream_id=0)
    engine.register_model(yolo)

    print("   - AdaFace (Stream 3) - Placeholder")
    adaface = AdaFaceModel(stream_id=3)
    engine.register_model(adaface)

    # Initialize
    print("\n4. Initializing all models...")
    start = time.time()
    await engine.initialize_all_models()
    print(f"   ✓ Initialized in {time.time() - start:.2f}s")

    # Single inference with details
    print("\n5. Running inference on test image...")
    start = time.time()
    result = await engine.run_parallel_inference(test_image)
    elapsed = (time.time() - start) * 1000

    print("\n" + "=" * 70)
    print("INFERENCE RESULT:")
    print("=" * 70)
    print(f"Total Time: {elapsed:.1f}ms")
    print(f"Person: {result.person_name or 'Unknown'}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Trust Score: {result.trust_score:.1f}%")
    print(f"Consensus: {result.consensus_count}/{len(result.model_results)} models")
    print(f"Quality: {result.quality_score:.2f}")
    print(f"Is Live: {result.is_live}")

    print("\n" + "=" * 70)
    print("PER-MODEL RESULTS:")
    print("=" * 70)

    total_sequential = 0
    for mr in result.model_results:
        detection_status = "✓ Detected" if mr.bbox else "✗ No face"
        match_status = f"Match: {mr.person_name}" if mr.person_id else "No match"

        print(f"{mr.model_name:15} | {mr.inference_time:6.1f}ms | "
              f"{detection_status:15} | {match_status:20} | Conf: {mr.confidence:.3f}")
        total_sequential += mr.inference_time

    print("\n" + "=" * 70)
    print("PERFORMANCE:")
    print("=" * 70)
    print(f"Sequential (sum): {total_sequential:.1f}ms")
    print(f"Parallel (actual): {elapsed:.1f}ms")
    speedup = total_sequential / elapsed if elapsed > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

    # Benchmark
    print("\n" + "=" * 70)
    print("BENCHMARK (20 iterations):")
    print("=" * 70)

    times = []
    for i in range(20):
        start = time.time()
        result = await engine.run_parallel_inference(test_image)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        if i % 5 == 0:
            print(f"Iteration {i+1}: {elapsed:.1f}ms | Trust: {result.trust_score:.1f}%")

    print(f"\nAverage: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
    print(f"Min: {np.min(times):.1f}ms")
    print(f"Max: {np.max(times):.1f}ms")

    # CUDA streams info
    print("\n" + "=" * 70)
    print("CUDA STREAMS:")
    print("=" * 70)
    from app.core.multi_agent.utils.cuda_streams import cuda_stream_manager
    stream_info = cuda_stream_manager.get_stream_info()
    for stream_id, info in stream_info['streams'].items():
        print(f"Stream {stream_id}: {', '.join(info['models'])}")

    # Cleanup
    print("\n6. Cleaning up...")
    engine.cleanup()

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
