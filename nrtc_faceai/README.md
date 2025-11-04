# NRTC Face AI - Proprietary Face Recognition Library

## Overview

NRTC Face AI is a proprietary, license-protected face recognition library optimized for NVIDIA Jetson devices. This library provides GPU-accelerated face detection, recognition, and matching capabilities with hardware binding and license enforcement.

## Features

- **GPU-Accelerated Face Detection** using SCRFD (InsightFace)
- **Face Recognition** using ArcFace embeddings
- **Traditional Augmentation** techniques for improved accuracy
- **Hardware Binding** to Jetson devices
- **License Enforcement** with expiration and feature checks
- **Optimized for Jetson AGX Orin** with CUDA support

## Requirements

- Python >= 3.8
- NVIDIA Jetson Device (AGX Orin, Xavier, Nano, etc.)
- CUDA 11.4+ with cuDNN
- Valid NRTC Face AI License

## Installation

```bash
pip install nrtc_faceai-1.0.0-py3-none-any.whl
```

## License Setup

Before using the library, you must obtain a license file from NRTC:

1. Contact NRTC at info@nrtc.com.pk
2. Provide your Jetson device hardware ID
3. Receive your `nrtc_license.json` file
4. Place the license file in one of these locations:
   - Current working directory: `./nrtc_license.json`
   - User home directory: `~/.nrtc/license.json`
   - System directory: `/etc/nrtc/license.json`

### Get Hardware ID

```python
from nrtc_faceai.license import HardwareIdentifier

hw_id = HardwareIdentifier.generate_hardware_id()
print(f"Hardware ID: {hw_id}")
```

## Usage

### Face Detection

```python
from nrtc_faceai import FaceDetector
import cv2

# Initialize detector (license validation happens here)
detector = FaceDetector(min_detection_confidence=0.5)

# Load image
image = cv2.imread('photo.jpg')

# Detect faces
faces = detector.detect_faces(image)

for face in faces:
    x, y, w, h = face.bbox
    print(f"Face detected at ({x}, {y}) with confidence {face.confidence:.2f}")
```

### Face Recognition

```python
from nrtc_faceai import FaceRecognizer
import cv2

# Initialize recognizer (license validation happens here)
recognizer = FaceRecognizer()

# Extract embedding
image = cv2.imread('person.jpg')
result = recognizer.extract_embedding(image)

if result:
    print(f"Embedding shape: {result.embedding.shape}")
    print(f"Confidence: {result.confidence:.2f}")
```

### Face Matching

```python
# Compare two faces
similarity = recognizer.compare_embeddings(embedding1, embedding2)
print(f"Similarity: {similarity:.3f}")

# Match against database
database_embeddings = [emb1, emb2, emb3]
match_idx, similarity = recognizer.match_face(
    query_embedding,
    database_embeddings,
    threshold=0.4
)

if match_idx >= 0:
    print(f"Match found at index {match_idx} with similarity {similarity:.3f}")
```

### Augmentation

```python
from nrtc_faceai import FaceAugmentation
import cv2

# Initialize augmentation (license validation happens here)
augmentor = FaceAugmentation()

# Generate variations
face_image = cv2.imread('face.jpg')
variations = augmentor.generate_variations(face_image, num_variations=10)

print(f"Generated {len(variations)} augmented images")
```

## License Validation

The library performs license validation on first use of each module:

- ✓ **Signature Verification** - Ensures license hasn't been tampered
- ✓ **Expiration Check** - Verifies license is still valid
- ✓ **Hardware Binding** - Confirms license is for this device
- ✓ **Feature Check** - Validates requested features are enabled

License validation errors will raise exceptions:
- `LicenseError` - General license error
- `LicenseExpiredError` - License has expired
- `InvalidLicenseError` - License signature invalid
- `HardwareBindingError` - License not valid for this device

## Supported Features

License can enable/disable features:
- `face_detection` - Face detection capabilities
- `face_recognition` - Face recognition and matching
- `augmentation` - Image augmentation techniques
- `*` - All features (full license)

## Performance

Optimized for Jetson AGX Orin:
- **Face Detection**: ~30 FPS @ 640x640
- **Face Recognition**: ~50 FPS embedding extraction
- **GPU Memory**: ~2GB for detection + recognition

## Support

For technical support, license inquiries, or issues:

**NRTC (National Radio & Telecommunication Corporation)**
- Website: https://nrtc.com.pk
- Email: info@nrtc.com.pk
- Phone: +92-51-111-678-200

## Copyright

Copyright (c) 2025 NRTC. All rights reserved.

This software is protected by copyright and license agreement. Unauthorized use, distribution, reverse engineering, or modification is strictly prohibited and may result in legal action.

## Version

Current version: 1.0.0
