# NRTC Face AI - Proprietary Library Documentation

## Overview

This document describes the proprietary NRTC Face AI library created to protect the intellectual property of the face recognition system. The library implements hardware-bound licensing and provides a professional, distributable package for commercial deployment.

## What Was Built

### 1. Proprietary Library Package (`nrtc_faceai/`)

A complete Python package with license enforcement and hardware binding:

```
nrtc_faceai/
├── LICENSE                          # Proprietary license agreement
├── README.md                        # Library documentation
├── setup.py                         # Build configuration
├── generate_license.py              # License generation tool
├── dist/
│   └── nrtc_faceai-1.0.0-py3-none-any.whl   # Wheel distribution
└── nrtc_faceai/
    ├── __init__.py                 # Package exports
    ├── core/                       # Face recognition core
    │   ├── detector.py            # Licensed face detector
    │   └── recognizer.py          # Licensed face recognizer
    ├── augmentation/              # Image augmentation
    │   └── traditional.py         # Traditional augmentation
    ├── license/                   # License management
    │   ├── validator.py          # License validation
    │   └── hardware.py           # Hardware ID extraction
    └── utils/                     # Utilities
        └── exceptions.py          # Custom exceptions
```

### 2. License Protection Features

#### Hardware Binding
- Extracts Jetson serial number, model, and MAC address
- Generates unique SHA256 hardware ID
- Validates license against current device
- Prevents unauthorized hardware usage

#### License Validation
- **Signature Verification**: HMAC-SHA256 signatures prevent tampering
- **Expiration Checking**: Time-based license expiration
- **Feature Gating**: Enable/disable specific features per license
- **Hardware Locking**: Bind licenses to specific Jetson devices

#### Protection Layers
```
┌─────────────────────────────────────────┐
│         User Application                │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│     NRTC Face AI Library (Wheel)        │
│  ┌───────────────────────────────────┐  │
│  │  License Validator                │  │
│  │  1. Check license file            │  │
│  │  2. Verify signature              │  │
│  │  3. Check expiration              │  │
│  │  4. Validate hardware ID          │  │
│  │  5. Check feature permissions     │  │
│  └───────────────────────────────────┘  │
│            │                             │
│  ┌─────────▼──────────────────────────┐ │
│  │  Core Face Recognition             │ │
│  │  - FaceDetector                    │ │
│  │  - FaceRecognizer                  │ │
│  │  - FaceAugmentation                │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         InsightFace (CUDA)              │
└─────────────────────────────────────────┘
```

### 3. Distribution Package

#### Wheel File
- **Location**: `nrtc_faceai/dist/nrtc_faceai-1.0.0-py3-none-any.whl`
- **Size**: ~30KB (without models)
- **Installation**: `pip install nrtc_faceai-1.0.0-py3-none-any.whl`

#### License File Format
```json
{
  "licensee": "Customer Name",
  "organization": "Company Ltd",
  "hardware_id": "6458d6643d893ea3...",
  "issue_date": "2025-11-04T...",
  "expiry_date": "2026-11-04",
  "license_type": "commercial",
  "features": ["*"],
  "signature": "d4f2e8a9..."
}
```

### 4. License Types

#### Development License (Wildcard)
- **Hardware ID**: `*` (works on any device)
- **Use Case**: Development and testing
- **Duration**: Configurable (default: 1 year)
- **File**: `nrtc_license.json` (already created)

#### Commercial License (Hardware-Bound)
- **Hardware ID**: Specific Jetson device hash
- **Use Case**: Production deployment
- **Duration**: Customer-specific
- **Protection**: Cannot be used on different hardware

#### Features Available
- `face_detection` - Face detection only
- `face_recognition` - Face recognition and matching
- `augmentation` - Image augmentation
- `*` - All features (full license)

## How It Works

### 1. Library Initialization

When user imports the library:

```python
from nrtc_faceai import FaceDetector

detector = FaceDetector()  # ← License validation happens here
```

The library automatically:
1. Searches for `nrtc_license.json` in:
   - Current directory
   - `~/.nrtc/license.json`
   - `/etc/nrtc/license.json`

2. Validates the license:
   - Loads JSON file
   - Verifies HMAC-SHA256 signature
   - Checks expiration date
   - Generates current hardware ID
   - Compares with license hardware_id
   - Confirms feature permissions

3. If validation fails:
   - Raises `LicenseError` exception
   - Application cannot use the library
   - Clear error message shown

### 2. Hardware ID Generation

```python
# Hardware ID is computed from:
hw_string = f"{serial}-{model}-{mac}"
hw_id = SHA256(hw_string)

# Example for this Jetson:
Serial:  1421124247808
Model:   NVIDIA Jetson AGX Orin Developer Kit
MAC:     (eth0 or wlan0)
Hash:    6458d6643d893ea39bb7cf48bc27e827...
```

### 3. License Generation

For NRTC internal use to generate customer licenses:

```bash
# Interactive mode
python3 nrtc_faceai/generate_license.py --generate

# Get customer's hardware ID
python3 nrtc_faceai/generate_license.py --get-hardware-id
```

## Usage Examples

### Example 1: Basic Face Detection

```python
from nrtc_faceai import FaceDetector
import cv2

# License validation happens on initialization
detector = FaceDetector(min_detection_confidence=0.5)

# Use detector
image = cv2.imread('photo.jpg')
faces = detector.detect_faces(image)

for face in faces:
    print(f"Face detected: {face.bbox}, confidence: {face.confidence}")
```

### Example 2: Face Recognition

```python
from nrtc_faceai import FaceRecognizer
import cv2

# License validation happens on initialization
recognizer = FaceRecognizer()

# Extract embedding
image = cv2.imread('person.jpg')
result = recognizer.extract_embedding(image)

if result:
    embedding = result.embedding
    print(f"Extracted 512-D embedding: {embedding.shape}")
```

### Example 3: Face Matching

```python
from nrtc_faceai import FaceRecognizer
import pickle

recognizer = FaceRecognizer()

# Compare two faces
similarity = recognizer.compare_embeddings(embedding1, embedding2)

if similarity > 0.6:
    print("Same person")
else:
    print("Different person")
```

## Tools and Scripts

### 1. Development License Generator

**File**: `create_dev_license.py`

Quickly create a development license for current device:

```bash
python3 create_dev_license.py
```

Generates `nrtc_license.json` with:
- Wildcard hardware ID (`*`)
- 1 year expiration
- All features enabled
- Development license type

### 2. Customer License Generator

**File**: `nrtc_faceai/generate_license.py`

Full-featured license generation tool for NRTC staff:

```bash
# Interactive mode
python3 nrtc_faceai/generate_license.py --generate

# Get hardware ID
python3 nrtc_faceai/generate_license.py --get-hardware-id
```

## Deployment Workflow

### For Development

1. Install library:
```bash
pip install nrtc_faceai/dist/nrtc_faceai-1.0.0-py3-none-any.whl
```

2. Generate development license:
```bash
python3 create_dev_license.py
```

3. Use library:
```python
from nrtc_faceai import FaceDetector, FaceRecognizer
# ... your code ...
```

### For Production (Customer Deployment)

1. Customer provides hardware ID:
```bash
# On customer's Jetson
python3 -c "from nrtc_faceai.license import HardwareIdentifier; print(HardwareIdentifier.generate_hardware_id())"
```

2. NRTC generates hardware-bound license:
```bash
python3 nrtc_faceai/generate_license.py --generate
# Enter customer details
# Enter hardware ID from step 1
# Select commercial license, expiration, features
```

3. Send to customer:
- `nrtc_faceai-1.0.0-py3-none-any.whl`
- `nrtc_license_<customer>.json`

4. Customer installs:
```bash
pip install nrtc_faceai-1.0.0-py3-none-any.whl
cp nrtc_license_<customer>.json nrtc_license.json
```

5. Application uses library (license validated automatically)

## Benefits

### 1. Intellectual Property Protection
- Core algorithms hidden in wheel package
- License enforcement prevents unauthorized use
- Hardware binding stops license sharing
- Signature verification prevents tampering

### 2. Commercial Distribution
- Professional Python wheel package
- Standard pip installation
- Clean API surface
- Comprehensive documentation

### 3. Flexible Licensing
- Feature-gated licensing (basic/pro/enterprise)
- Time-based expiration
- Hardware-bound or development licenses
- Easy license renewal process

### 4. Customer Experience
- Simple `pip install` workflow
- Clear license validation errors
- No performance overhead
- Same API as open-source version

## Advanced Topics

### Code Obfuscation (Optional Next Step)

For additional protection, consider using PyArmor:

```bash
# Install PyArmor
pip install pyarmor

# Obfuscate code
cd nrtc_faceai
pyarmor pack -e "--onefile" nrtc_faceai

# Creates obfuscated version
```

This would:
- Encrypt Python bytecode
- Add runtime decryption
- Make reverse engineering much harder
- Increase protection level

### License Server (Future Enhancement)

Instead of file-based licenses, implement license server:

```
Customer Device          NRTC License Server
      │                          │
      ├──── Check License ──────▶│
      │     (Hardware ID)         │
      │                          │
      │◀──── License Token ───────│
      │     (Signed JWT)         │
      │                          │
      └─── Use Library
```

Benefits:
- Real-time license validation
- Remote license revocation
- Usage analytics
- Cloud-based management

## Current System Status

### Completed
- ✓ Proprietary library structure
- ✓ Hardware binding implementation
- ✓ License validation system
- ✓ Wheel distribution package
- ✓ Development license created
- ✓ Comprehensive documentation
- ✓ License generation tools

### Files Created
- `nrtc_faceai/` - Complete library package
- `nrtc_faceai-1.0.0-py3-none-any.whl` - Distributable wheel
- `nrtc_license.json` - Development license
- `create_dev_license.py` - Quick license generator
- `PROPRIETARY_LIBRARY.md` - This documentation

### Next Steps (Optional)
1. Integrate library into main application
2. Replace direct InsightFace calls with NRTC Face AI
3. Test with production workloads
4. Generate production licenses for customers
5. Add code obfuscation (PyArmor)
6. Implement license server (if needed)

## Support and Contact

For licensing inquiries or technical support:

**NRTC (National Radio & Telecommunication Corporation)**
- Website: https://nrtc.com.pk
- Email: info@nrtc.com.pk
- Phone: +92-51-111-678-200

---

**Copyright © 2025 NRTC. All rights reserved.**

This library is protected by copyright and license agreement. Unauthorized use, distribution, or reverse engineering is prohibited.
