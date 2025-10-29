"""
Test script for Multi-Agent Parallel Inference Engine

Tests the parallel inference engine with ArcFace model
"""

import asyncio
import cv2
import numpy as np
import time
from app.core.multi_agent.engine import ParallelInferenceEngine
from app.core.multi_agent.models.arcface_model import ArcFaceModel


async def main():
    print("=" * 70)
    print("Multi-Agent Parallel Inference Engine Test")
    print("=" * 70)

    # Initialize engine
    print("\n1. Initializing Parallel Inference Engine...")
    engine = ParallelInferenceEngine()

    # Register ArcFace model
    print("\n2. Registering ArcFace model...")
    arcface = ArcFaceModel(stream_id=1)
    engine.register_model(arcface)

    # Initialize all models
    print("\n3. Initializing all models in parallel...")
    start = time.time()
    await engine.initialize_all_models()
    init_time = time.time() - start
    print(f"   ✓ Initialization time: {init_time:.2f}s")

    # Create test image
    print("\n4. Creating test image...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warm-up run
    print("\n5. Warm-up run...")
    _ = await engine.run_parallel_inference(test_image)

    # Benchmark
    print("\n6. Running benchmark (10 iterations)...")
    times = []
    for i in range(10):
        start = time.time()
        result = await engine.run_parallel_inference(test_image)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed:.1f}ms (trust: {result.trust_score:.1f}%)")

    # Statistics
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Average time: {np.mean(times):.1f}ms")
    print(f"Min time: {np.min(times):.1f}ms")
    print(f"Max time: {np.max(times):.1f}ms")
    print(f"Std dev: {np.std(times):.1f}ms")

    # Engine stats
    print("\n" + "=" * 70)
    print("ENGINE STATS:")
    print("=" * 70)
    stats = engine.get_stats()
    print(f"Total inferences: {stats['total_inferences']}")
    print(f"Average latency: {stats['avg_latency']:.1f}ms")
    print(f"Average trust score: {stats['avg_trust_score']:.1f}%")
    print(f"Number of models: {stats['num_models']}")
    print(f"Models: {', '.join(stats['models'])}")

    # Test with real image (if available)
    print("\n" + "=" * 70)
    print("REAL IMAGE TEST (if camera available):")
    print("=" * 70)
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            print("✓ Camera frame captured")
            start = time.time()
            result = await engine.run_parallel_inference(frame)
            elapsed = (time.time() - start) * 1000

            print(f"\nResult:")
            print(f"  Person: {result.person_name or 'Unknown'}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Trust Score: {result.trust_score:.1f}%")
            print(f"  Consensus: {result.consensus_count}/{len(result.model_results)} models")
            print(f"  Inference Time: {elapsed:.1f}ms")
            print(f"  Is Live: {result.is_live}")

            print(f"\nPer-Model Results:")
            for model_result in result.model_results:
                print(f"  - {model_result.model_name}: "
                      f"{model_result.person_name or 'None'} "
                      f"(conf: {model_result.confidence:.3f}, "
                      f"{model_result.inference_time:.1f}ms)")
        else:
            print("⚠️  No camera available, skipping real image test")

    except Exception as e:
        print(f"⚠️  Camera test failed: {e}")

    # Cleanup
    print("\n7. Cleaning up...")
    engine.cleanup()

    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
