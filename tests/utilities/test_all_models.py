#!/usr/bin/env python3
"""
Comprehensive Model Analysis Script
Tests all models and reports their execution providers, precision, and performance
"""

import sys
import os
sys.path.insert(0, '/home/mujeeb/Downloads/face-recognition-security-system')

import time
import numpy as np
import cv2
from app.core.detector import FaceDetector
from app.core.recognizer import FaceRecognizer

print("=" * 80)
print("COMPREHENSIVE MODEL ANALYSIS")
print("=" * 80)

# Initialize detector and recognizer
print("\n1. INITIALIZING MODELS...")
print("-" * 80)

start_time = time.time()
detector = FaceDetector(min_detection_confidence=0.5)
detector_init_time = time.time() - start_time
print(f"✓ Face Detector initialized in {detector_init_time:.2f}s")

start_time = time.time()
recognizer = FaceRecognizer()
recognizer_init_time = time.time() - start_time
print(f"✓ Face Recognizer initialized in {recognizer_init_time:.2f}s")

# Analyze all models in the detector
print("\n2. DETECTOR MODELS ANALYSIS")
print("-" * 80)

detector_models = []
if hasattr(detector.app, 'models'):
    for model_name, model in detector.app.models.items():
        if hasattr(model, 'session'):
            providers = model.session.get_providers()

            # Get model details
            model_info = {
                'name': model_name,
                'primary_provider': providers[0] if providers else 'Unknown',
                'all_providers': providers,
                'model_path': getattr(model, 'model_file', 'Unknown')
            }

            # Check for FP16 in provider options
            provider_options = model.session.get_provider_options()
            if 'TensorrtExecutionProvider' in provider_options:
                trt_opts = provider_options['TensorrtExecutionProvider']
                model_info['fp16_enabled'] = trt_opts.get('trt_fp16_enable', 'N/A')
                model_info['engine_cache'] = trt_opts.get('trt_engine_cache_enable', 'N/A')

            detector_models.append(model_info)

            print(f"\nModel: {model_name}")
            print(f"  Primary Provider: {providers[0]}")
            print(f"  FP16 Enabled: {model_info.get('fp16_enabled', 'N/A')}")
            print(f"  Model File: {model_info['model_path']}")

# Analyze all models in the recognizer
print("\n3. RECOGNIZER MODELS ANALYSIS")
print("-" * 80)

recognizer_models = []
if hasattr(recognizer.app, 'models'):
    for model_name, model in recognizer.app.models.items():
        if hasattr(model, 'session'):
            providers = model.session.get_providers()

            model_info = {
                'name': model_name,
                'primary_provider': providers[0] if providers else 'Unknown',
                'all_providers': providers,
                'model_path': getattr(model, 'model_file', 'Unknown')
            }

            # Check for FP16 in provider options
            provider_options = model.session.get_provider_options()
            if 'TensorrtExecutionProvider' in provider_options:
                trt_opts = provider_options['TensorrtExecutionProvider']
                model_info['fp16_enabled'] = trt_opts.get('trt_fp16_enable', 'N/A')
                model_info['engine_cache'] = trt_opts.get('trt_engine_cache_enable', 'N/A')

            recognizer_models.append(model_info)

            print(f"\nModel: {model_name}")
            print(f"  Primary Provider: {providers[0]}")
            print(f"  FP16 Enabled: {model_info.get('fp16_enabled', 'N/A')}")
            print(f"  Model File: {model_info['model_path']}")

# Performance benchmark
print("\n4. PERFORMANCE BENCHMARK")
print("-" * 80)

# Create test image with a synthetic face-like pattern
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
# Add some structure to make it more likely to detect something (optional)
cv2.circle(test_image, (320, 320), 100, (255, 200, 150), -1)  # Face-like circle
cv2.circle(test_image, (290, 300), 15, (50, 50, 50), -1)  # Left eye
cv2.circle(test_image, (350, 300), 15, (50, 50, 50), -1)  # Right eye
cv2.ellipse(test_image, (320, 350), (30, 20), 0, 0, 180, (50, 50, 50), 2)  # Mouth

print("\nTesting Face Detector...")
# Warmup
for _ in range(3):
    _ = detector.detect_faces(test_image)

# Benchmark
detection_times = []
for i in range(10):
    start = time.time()
    detections = detector.detect_faces(test_image)
    elapsed = (time.time() - start) * 1000
    detection_times.append(elapsed)

print(f"  Detections found: {len(detections)}")
print(f"  Average time: {np.mean(detection_times):.2f}ms")
print(f"  Min/Max: {np.min(detection_times):.2f}ms / {np.max(detection_times):.2f}ms")
print(f"  FPS: {1000/np.mean(detection_times):.1f}")

print("\nTesting Face Recognizer...")
# Warmup
for _ in range(3):
    _ = recognizer.extract_embedding(test_image)

# Benchmark
recognition_times = []
for i in range(10):
    start = time.time()
    result = recognizer.extract_embedding(test_image)
    elapsed = (time.time() - start) * 1000
    recognition_times.append(elapsed)

print(f"  Embedding extracted: {result is not None}")
print(f"  Average time: {np.mean(recognition_times):.2f}ms")
print(f"  Min/Max: {np.min(recognition_times):.2f}ms / {np.max(recognition_times):.2f}ms")
if result:
    print(f"  Embedding size: {result.embedding.shape}")

# Summary
print("\n5. SUMMARY")
print("=" * 80)

all_models = detector_models + recognizer_models
print(f"\nTotal Models Loaded: {len(all_models)}")

tensorrt_count = sum(1 for m in all_models if 'Tensorrt' in m['primary_provider'])
cuda_count = sum(1 for m in all_models if 'CUDA' in m['primary_provider'] and 'Tensorrt' not in m['primary_provider'])
cpu_count = sum(1 for m in all_models if 'CPU' in m['primary_provider'])

print(f"  TensorRT Provider: {tensorrt_count} models")
print(f"  CUDA Provider: {cuda_count} models")
print(f"  CPU Provider: {cpu_count} models")

fp16_count = sum(1 for m in all_models if m.get('fp16_enabled') == '1')
print(f"\nFP16 Precision: {fp16_count} models")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
