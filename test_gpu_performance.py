"""
Test GPU performance - verify TensorRT is being used
"""
import time
import cv2
import numpy as np
from app.core.recognizer import FaceRecognizer

print("=" * 60)
print("GPU Performance Test")
print("=" * 60)

# Initialize recognizer
print("\n1. Initializing Face Recognizer...")
start = time.time()
recognizer = FaceRecognizer()
init_time = time.time() - start
print(f"   Initialization time: {init_time:.2f}s")

# Check which provider is being used
try:
    provider = recognizer.app.models['recognition'].session.get_providers()[0]
    print(f"   ✓ Active Provider: {provider}")
    if 'TensorRT' in provider:
        print("   ✅ GPU acceleration ENABLED (TensorRT)")
    elif 'CUDA' in provider:
        print("   ✅ GPU acceleration ENABLED (CUDA)")
    else:
        print("   ⚠️  CPU-only mode")
except Exception as e:
    print(f"   Could not determine provider: {e}")

# Create test image (640x640 with a dummy face)
print("\n2. Creating test image...")
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warm-up run
print("\n3. Warm-up run...")
_ = recognizer.extract_embedding(test_image)

# Benchmark
print("\n4. Running benchmark (10 iterations)...")
times = []
for i in range(10):
    start = time.time()
    result = recognizer.extract_embedding(test_image)
    elapsed = (time.time() - start) * 1000  # Convert to milliseconds
    times.append(elapsed)
    print(f"   Iteration {i+1}: {elapsed:.1f}ms")

# Statistics
print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Average time: {np.mean(times):.1f}ms")
print(f"Min time: {np.min(times):.1f}ms")
print(f"Max time: {np.max(times):.1f}ms")
print(f"Std dev: {np.std(times):.1f}ms")

# Interpretation
avg_time = np.mean(times)
if avg_time < 100:
    print("\n✅ EXCELLENT! Likely using GPU acceleration")
    print(f"   Expected GPU performance: 40-100ms ✓")
elif avg_time < 200:
    print("\n⚠️  GOOD: Moderate performance")
    print(f"   May be using GPU or optimized CPU")
else:
    print("\n❌ SLOW: Likely CPU-only")
    print(f"   Expected CPU performance: 300-400ms")
    print(f"   Current performance: {avg_time:.1f}ms")

print("=" * 60)
