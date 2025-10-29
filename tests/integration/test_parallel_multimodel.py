"""
Multi-Model Parallel Inference Test

Tests the parallel inference engine with multiple models running simultaneously
"""

import asyncio
import cv2
import numpy as np
import time
from app.core.multi_agent.engine import ParallelInferenceEngine
from app.core.multi_agent.models.arcface_model import ArcFaceModel

# Try to import optional models
try:
    from app.core.multi_agent.models.yolov8_detector import YOLOv8FaceDetector
    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"YOLOv8 not available: {e}")

try:
    from app.core.multi_agent.models.facenet_model import FaceNetModel
    FACENET_AVAILABLE = True
except ImportError as e:
    FACENET_AVAILABLE = False
    print(f"FaceNet not available: {e}")

try:
    from app.core.multi_agent.models.adaface_model import AdaFaceModel
    ADAFACE_AVAILABLE = True
except ImportError as e:
    ADAFACE_AVAILABLE = False
    print(f"AdaFace not available: {e}")

try:
    from app.core.multi_agent.models.clip_model import CLIPModel
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False
    print(f"CLIP not available: {e}")


async def main():
    print("=" * 70)
    print("MULTI-MODEL PARALLEL INFERENCE ENGINE TEST")
    print("=" * 70)

    # Initialize engine
    print("\n1. Initializing Parallel Inference Engine...")
    engine = ParallelInferenceEngine()

    # Register all models
    print("\n2. Registering models...")
    models_to_test = []

    # Always include ArcFace (working)
    print("   - ArcFace (Stream 1)")
    arcface = ArcFaceModel(stream_id=1)
    engine.register_model(arcface)
    models_to_test.append("ArcFace")

    # Try to add YOLOv8
    if YOLO_AVAILABLE:
        try:
            print("   - YOLOv8-Face (Stream 0)")
            yolo = YOLOv8FaceDetector(stream_id=0)
            engine.register_model(yolo)
            models_to_test.append("YOLOv8")
        except Exception as e:
            print(f"   ⚠️  YOLOv8 failed to load: {e}")
    else:
        print("   ⚠️  YOLOv8 skipped (dependencies missing)")

    # Try to add FaceNet
    if FACENET_AVAILABLE:
        try:
            print("   - FaceNet (Stream 2)")
            facenet = FaceNetModel(stream_id=2)
            engine.register_model(facenet)
            models_to_test.append("FaceNet")
        except Exception as e:
            print(f"   ⚠️  FaceNet failed to load: {e}")
    else:
        print("   ⚠️  FaceNet skipped (torch not installed)")

    # Try to add AdaFace
    if ADAFACE_AVAILABLE:
        try:
            print("   - AdaFace (Stream 3)")
            adaface = AdaFaceModel(stream_id=3)
            engine.register_model(adaface)
            models_to_test.append("AdaFace")
        except Exception as e:
            print(f"   ⚠️  AdaFace failed to load: {e}")
    else:
        print("   ⚠️  AdaFace skipped (dependencies missing)")

    # Try to add CLIP
    if CLIP_AVAILABLE:
        try:
            print("   - CLIP-ViT (Stream 4)")
            clip = CLIPModel(stream_id=4)
            engine.register_model(clip)
            models_to_test.append("CLIP")
        except Exception as e:
            print(f"   ⚠️  CLIP failed to load: {e}")
    else:
        print("   ⚠️  CLIP skipped (transformers not installed)")

    print(f"\n   ✓ Registered {len(models_to_test)} models: {', '.join(models_to_test)}")

    # Initialize all models in parallel
    print("\n3. Initializing all models in parallel...")
    start = time.time()
    await engine.initialize_all_models()
    init_time = time.time() - start
    print(f"   ✓ All models initialized in {init_time:.2f}s")

    # Create test image
    print("\n4. Creating test image...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warm-up run
    print("\n5. Warm-up run...")
    _ = await engine.run_parallel_inference(test_image)

    # Benchmark parallel execution
    print("\n6. Running PARALLEL benchmark (10 iterations)...")
    print("   All models run simultaneously on different CUDA streams")
    parallel_times = []

    for i in range(10):
        start = time.time()
        result = await engine.run_parallel_inference(test_image)
        elapsed = (time.time() - start) * 1000
        parallel_times.append(elapsed)

        print(f"   Iteration {i+1}: {elapsed:.1f}ms | "
              f"Consensus: {result.consensus_count}/{len(result.model_results)} | "
              f"Trust: {result.trust_score:.1f}%")

    # Statistics
    print("\n" + "=" * 70)
    print("PARALLEL EXECUTION RESULTS:")
    print("=" * 70)
    print(f"Models running: {len(models_to_test)}")
    print(f"Average parallel time: {np.mean(parallel_times):.1f}ms")
    print(f"Min time: {np.min(parallel_times):.1f}ms")
    print(f"Max time: {np.max(parallel_times):.1f}ms")
    print(f"Std dev: {np.std(parallel_times):.1f}ms")

    # Show per-model breakdown from last run
    print("\n" + "=" * 70)
    print("PER-MODEL BREAKDOWN (Last Iteration):")
    print("=" * 70)

    total_sequential = 0
    for model_result in result.model_results:
        print(f"{model_result.model_name:15} | "
              f"{model_result.inference_time:6.1f}ms | "
              f"Conf: {model_result.confidence:.3f} | "
              f"Match: {model_result.person_name or 'None'}")
        total_sequential += model_result.inference_time

    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    print("=" * 70)
    print(f"Sequential (sum of all models): {total_sequential:.1f}ms")
    print(f"Parallel (actual execution):    {np.mean(parallel_times):.1f}ms")
    speedup = total_sequential / np.mean(parallel_times) if np.mean(parallel_times) > 0 else 0
    print(f"Speedup: {speedup:.2f}x faster!")

    # Engine stats
    print("\n" + "=" * 70)
    print("ENGINE STATISTICS:")
    print("=" * 70)
    stats = engine.get_stats()
    print(f"Total inferences: {stats['total_inferences']}")
    print(f"Average latency: {stats['avg_latency']:.1f}ms")
    print(f"Average trust score: {stats['avg_trust_score']:.1f}%")
    print(f"Active models: {stats['num_models']}")

    # CUDA stream info
    print("\n" + "=" * 70)
    print("CUDA STREAMS:")
    print("=" * 70)
    from app.core.multi_agent.utils.cuda_streams import cuda_stream_manager
    stream_info = cuda_stream_manager.get_stream_info()
    print(f"Total streams: {stream_info['total_streams']}")
    for stream_id, info in stream_info['streams'].items():
        models = ', '.join(info['models'])
        print(f"  Stream {stream_id}: {info['num_models']} model(s) - {models}")

    # Test with real image if camera available
    print("\n" + "=" * 70)
    print("REAL IMAGE TEST:")
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

            print(f"\nFinal Result:")
            print(f"  Person: {result.person_name or 'Unknown'}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Trust Score: {result.trust_score:.1f}%")
            print(f"  Consensus: {result.consensus_count}/{len(result.model_results)} models agree")
            print(f"  Quality: {result.quality_score:.2f}")
            print(f"  Liveness: {result.liveness_score:.2f} (Live: {result.is_live})")
            print(f"  Inference Time: {elapsed:.1f}ms")

            print(f"\nPer-Model Results:")
            for mr in result.model_results:
                status = "✓" if mr.person_id == result.person_id else "✗"
                print(f"  {status} {mr.model_name:15} | "
                      f"{mr.person_name or 'None':20} | "
                      f"Conf: {mr.confidence:.3f} | "
                      f"{mr.inference_time:.1f}ms")
        else:
            print("⚠️  No camera available")

    except Exception as e:
        print(f"⚠️  Camera test failed: {e}")

    # Cleanup
    print("\n7. Cleaning up...")
    engine.cleanup()

    print("\n" + "=" * 70)
    print("✅ MULTI-MODEL PARALLEL TEST COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - Models tested: {len(models_to_test)}")
    print(f"  - Parallel speedup: {speedup:.2f}x")
    print(f"  - Average latency: {np.mean(parallel_times):.1f}ms")
    print(f"  - Trust score: {stats['avg_trust_score']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
