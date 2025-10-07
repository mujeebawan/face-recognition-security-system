# Package Update Analysis & GPU Installation Guide

**Date**: October 6, 2025
**System**: NVIDIA Jetson AGX Orin (JetPack 5.1.2, GLIBC 2.31)
**Current Status**: CPU-only processing, 10-15 FPS

---

## 🚨 Critical Finding: GLIBC 2.31 Limitation

**Your system has GLIBC 2.31** but **onnxruntime-gpu 1.16+** requires **GLIBC 2.32+**

This is THE blocker for GPU acceleration with ONNX Runtime.

---

## 📊 Package Update Priority Analysis

### ✅ SAFE TO UPDATE (Recommended)

These packages are behind and **should be updated**:

| Package | Current | Latest | Impact | Update Command |
|---------|---------|--------|--------|----------------|
| **alembic** | 1.13.1 | 1.14.1 | Database migrations | `pip install --upgrade alembic` |
| **pandas** | 0.25.3 | 2.1.4 | Alert statistics | `pip install --upgrade pandas` |
| **pydantic-settings** | 2.1.0 | 2.6.x | Settings management | `pip install --upgrade pydantic-settings` |
| **aiofiles** | 23.2.1 | 24.x | Async file I/O | `pip install --upgrade aiofiles` |
| **python-dateutil** | 2.8.2 | 2.9.x | Date handling | `pip install --upgrade python-dateutil` |
| **httpx** | 0.26.0 | 0.28.x | HTTP client | `pip install --upgrade httpx` |
| **requests** | 2.31.0 | 2.32.x | HTTP requests | `pip install --upgrade requests` |
| **pytest** | 7.4.4 | 8.3.x | Testing framework | `pip install --upgrade pytest` |
| **pytest-asyncio** | 0.23.3 | 0.24.x | Async testing | `pip install --upgrade pytest-asyncio` |
| **pytest-cov** | 4.1.0 | 6.0.x | Coverage | `pip install --upgrade pytest-cov` |
| **black** | 24.1.1 | 24.10.x | Code formatter | `pip install --upgrade black` |
| **flake8** | 7.0.0 | 7.1.x | Linter | `pip install --upgrade flake8` |
| **mypy** | 1.8.0 | 1.13.x | Type checker | `pip install --upgrade mypy` |

**Why safe?**
- No breaking API changes
- Better bug fixes and security patches
- Won't affect core face recognition functionality

---

### ⚠️ UPDATE WITH CAUTION

These **might** have breaking changes:

| Package | Current | Latest | Risk | Recommendation |
|---------|---------|--------|------|----------------|
| **Pillow** | 10.2.0 | 11.0.x | Low | Test first, likely safe |
| **scikit-learn** | 1.3.2 | 1.5.x | Low | Safe for cosine similarity |
| **albumentations** | 1.3.1 | 1.4.x | Low | Test augmentation pipeline |
| **scikit-image** | 0.22.0 | 0.24.x | Low | If using advanced features |
| **python-json-logger** | 2.0.7 | 3.2.x | Medium | Major version jump |

**Recommendation**: Update in development, test thoroughly before production.

---

### ❌ DO NOT UPDATE (Breaking Changes or Incompatible)

| Package | Current | Latest | Why NOT to Update |
|---------|---------|--------|-------------------|
| **Python** | 3.8.10 | 3.12.x | Ubuntu 20.04 LTS default; Jetson compatibility |
| **PyTorch** | 2.1.0 | 2.5.x | Jetson wheels only available for 2.1.0 or older |
| **torchvision** | 0.16.0 | 0.20.x | Must match PyTorch version |
| **NumPy** | 1.24.3 | 2.1.x | Many AI libs need <2.0; breaking changes |
| **MediaPipe** | 0.10.9 | 0.10.18+ | Google deprecated Python support; API changes |
| **onnxruntime** | 1.19.2 | 1.20.x | Works fine; 1.20 might have Jetson issues |
| **CUDA** | 11.4 | 12.x | Tied to JetPack 5.1.2; can't update independently |
| **JetPack** | 5.1.2 | 6.0 | JetPack 6.0 still in Developer Preview (unstable) |

**Why keep these versions?**
- Platform constraints (Jetson compatibility)
- Stability over bleeding edge
- Working fine for current use case

---

### 🤔 ALREADY AUTO-UPDATED

These packages were **auto-upgraded** by pip (good!):

| Package | requirements.txt | Actually Installed | Status |
|---------|------------------|-------------------|--------|
| **FastAPI** | 0.109.0 | 0.118.0 | ✅ Good |
| **Uvicorn** | 0.27.0 | 0.33.0 | ✅ Good |
| **SQLAlchemy** | 2.0.25 | 2.0.43 | ✅ Good |
| **Pydantic** | 2.5.3 | 2.10.6 | ✅ Good |
| **OpenCV** | 4.9.0 | 4.12.0 | ✅ Good |

**Action**: Update `requirements.txt` to reflect actual versions.

---

## 🎯 GPU Installation Options

### Option 1: TensorRT (RECOMMENDED for Jetson)

**This is the BEST option for Jetson GPU acceleration!**

#### What is TensorRT?
- NVIDIA's high-performance inference optimizer
- **Native Jetson support** (designed for it!)
- 2-5x faster than ONNX Runtime
- Already installed with JetPack 5.1.2

#### Check if TensorRT is available:
```bash
dpkg -l | grep tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

#### How to use TensorRT with InsightFace:
1. **Convert ONNX model to TensorRT**:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
     --onnx=buffalo_l.onnx \
     --saveEngine=buffalo_l.trt \
     --fp16  # Use FP16 for Jetson optimization
   ```

2. **Update recognizer.py** to use TensorRT:
   ```python
   # Instead of onnxruntime, use TensorRT
   import tensorrt as trt
   import pycuda.driver as cuda
   import pycuda.autoinit
   ```

3. **Benefits**:
   - ✅ No GLIBC issue (native to Jetson)
   - ✅ 2-5x faster inference
   - ✅ FP16 support (2x faster with minimal accuracy loss)
   - ✅ Already installed with JetPack

4. **Drawbacks**:
   - Requires model conversion (one-time)
   - More complex code (TensorRT API)
   - Model file is hardware-specific

---

### Option 2: Try onnxruntime-gpu 1.15.1 (Older Version)

**Status**: You already tried this - it FAILED with GLIBC error

Available wheels in your directory:
- `onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl` ❌ Failed
- `onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl` ❌ Failed
- `onnxruntime_gpu-1.18.0-cp38-cp38-linux_aarch64.whl` ❌ Failed

**All require GLIBC 2.32+ but you have 2.31**

---

### Option 3: Upgrade GLIBC (NOT RECOMMENDED - DANGEROUS!)

**Why dangerous?**
- GLIBC is core system library
- Upgrading can **break Ubuntu 20.04 entirely**
- Many system tools depend on GLIBC 2.31
- **Risk**: Unbootable system

**Only safe path**: Upgrade to Ubuntu 22.04 LTS (GLIBC 2.35)
- **But**: Jetson L4T R35 is based on Ubuntu 20.04
- **Requires**: JetPack 6.0 (Ubuntu 22.04 based, GLIBC 2.35)
- **Status**: JetPack 6.0 still in Developer Preview (unstable)

**Verdict**: **DO NOT attempt manual GLIBC upgrade!**

---

### Option 4: Build onnxruntime-gpu from Source

**Theoretically possible** but extremely complex:

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release \
  --use_cuda \
  --cuda_home /usr/local/cuda-11.4 \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --build_wheel \
  --parallel
```

**Challenges**:
- 4-8 hour build time on Jetson
- Requires 16GB+ RAM (Jetson has 32GB, OK)
- Complex dependency management
- May still fail with GLIBC compatibility

**Verdict**: **Not worth the effort** when TensorRT is available.

---

### Option 5: Use PyTorch with CUDA (Alternative)

**If you're using PyTorch** (you have it in requirements but not installed):

```bash
# Install PyTorch with CUDA for Jetson
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
```

**Then modify InsightFace** to use PyTorch backend instead of ONNX:
- InsightFace supports PyTorch models
- Use `providers=['CUDAExecutionProvider']` won't work
- **BUT** you can use PyTorch directly

**Verdict**: Possible but TensorRT is still better for Jetson.

---

## 🎯 RECOMMENDED ACTION PLAN

### Phase 1: Update Safe Packages (NOW)

```bash
# Navigate to project
cd /home/mujeeb/Downloads/face_recognition_system

# Update safe packages
pip install --upgrade \
  alembic \
  pandas \
  pydantic-settings \
  aiofiles \
  python-dateutil \
  httpx \
  requests \
  pytest \
  pytest-asyncio \
  pytest-cov \
  black \
  flake8 \
  mypy

# Update requirements.txt to match installed versions
pip freeze > requirements_new.txt
```

### Phase 2: TensorRT GPU Acceleration (NEXT)

**This is your best path to GPU acceleration!**

1. **Check TensorRT installation**:
   ```bash
   dpkg -l | grep tensorrt
   ls /usr/src/tensorrt/
   ```

2. **Install pycuda** (Python bindings):
   ```bash
   pip install pycuda
   ```

3. **Research TensorRT integration**:
   - Convert InsightFace buffalo_l model to TensorRT
   - Modify `app/core/recognizer.py` to use TensorRT
   - Benchmark performance

**Expected improvement**: 300-400ms → 50-100ms per face (3-8x faster!)

### Phase 3: Test Before Production

```bash
# Run tests after updates
pytest tests/

# Test face recognition
python3 test_recognizer.py

# Monitor performance
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# Check FPS in dashboard
```

---

## 📊 Update Summary Table

| Action | Package | Safe? | Priority | Impact |
|--------|---------|-------|----------|--------|
| ✅ UPDATE | alembic, pandas, pydantic-settings | Yes | High | Bug fixes, features |
| ✅ UPDATE | pytest, black, flake8, mypy | Yes | Medium | Dev tools |
| ✅ UPDATE | httpx, requests, aiofiles | Yes | Low | Minor improvements |
| ⚠️ TEST | Pillow, scikit-learn, albumentations | Caution | Low | Test first |
| ❌ KEEP | Python, PyTorch, NumPy, MediaPipe | No | - | Platform constraints |
| 🚀 TENSORRT | GPU acceleration | Yes | **HIGH** | 3-8x speed boost |

---

## 🔧 GPU Installation: TensorRT Method (RECOMMENDED)

### Step 1: Verify TensorRT is Installed

```bash
# Check TensorRT packages
dpkg -l | grep tensorrt

# Expected output:
# libnvinfer8
# libnvinfer-plugin8
# libnvparsers8
# libnvonnxparsers8
# python3-libnvinfer
```

### Step 2: Install Python Bindings

```bash
# Install pycuda for GPU memory management
pip install pycuda

# Verify TensorRT Python module
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

### Step 3: Convert InsightFace Model to TensorRT

```bash
# Navigate to models directory
cd data/models/

# Download InsightFace buffalo_l ONNX model
# (It's usually in ~/.insightface/models/)
cp ~/.insightface/models/buffalo_l/w600k_r50.onnx ./

# Convert to TensorRT engine
/usr/src/tensorrt/bin/trtexec \
  --onnx=w600k_r50.onnx \
  --saveEngine=w600k_r50_fp16.trt \
  --fp16 \
  --workspace=2048
```

**Expected output**:
- Conversion time: 5-15 minutes (one-time)
- Engine file: `w600k_r50_fp16.trt` (~50MB)
- 2x faster with FP16 precision

### Step 4: Update recognizer.py (Phase 7 Task)

This requires significant code changes - creating a TensorRT inference wrapper.

**Complexity**: Medium (2-4 hours of development)
**Expected speedup**: 3-8x faster (300ms → 40-100ms per face)

---

## 🚀 Quick Commands

### Update Safe Packages Now:
```bash
cd /home/mujeeb/Downloads/face_recognition_system
pip install --upgrade alembic pandas pydantic-settings aiofiles python-dateutil httpx requests pytest pytest-asyncio pytest-cov black flake8 mypy
```

### Check TensorRT Availability:
```bash
dpkg -l | grep tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "TensorRT Python not installed"
```

### Install TensorRT Python Support:
```bash
pip install pycuda
```

---

## ❓ FAQ

### Q: Why not just upgrade to Ubuntu 22.04?
**A**: Jetson L4T R35 (JetPack 5.1.2) is based on Ubuntu 20.04. Upgrading OS would break Jetson drivers.

### Q: When will JetPack 6.0 be stable?
**A**: Expected Q1-Q2 2025 (it's currently Developer Preview as of late 2024).

### Q: Is TensorRT hard to use?
**A**: More complex than ONNX Runtime, but well-documented. NVIDIA provides examples.

### Q: Can I use both CPU and GPU?
**A**: Yes! Keep CPU as fallback, use GPU when available (graceful degradation).

### Q: Will TensorRT break my current system?
**A**: No, it's already installed with JetPack. You're just enabling Python bindings.

---

**Next Steps**:
1. ✅ Update safe packages (10 minutes)
2. ✅ Update requirements.txt (1 minute)
3. 🔬 Research TensorRT integration (Phase 7)
4. 🚀 Implement GPU acceleration (2-3 days)

**Expected Result**: 20-30 FPS live stream with GPU acceleration!
