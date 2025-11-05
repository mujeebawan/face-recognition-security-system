# Jetson AGX Orin Setup Guide for Face Recognition System

**Last Updated:** 2025-11-03
**System Verified:** Jetson AGX Orin, JetPack 6.2 (L4T R36.4.7)

---

## 🖥️ Hardware & Software Specifications

### Hardware
- **Platform:** NVIDIA Jetson AGX Orin
- **Architecture:** aarch64 (ARM64)
- **GPU:** Integrated NVIDIA GPU with CUDA support
- **RAM:** 32GB+ recommended

### Software Environment
- **OS:** Ubuntu 20.04/22.04 (JetPack 6.2)
- **Linux Kernel:** 5.15.148-tegra
- **JetPack:** 6.2 (L4T R36.4.7)
- **Python:** 3.10.12

---

## 🔧 Critical Dependencies

### CUDA & GPU Stack
```bash
CUDA Version: 12.6 (Release 12.6.68)
cuDNN Version: 9.3.0
CUDA Compiler: nvcc 12.6.68
```

**CUDA Installation Path:**
```bash
/usr/local/cuda
/usr/local/cuda/lib64
```

**Important Library Paths:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

### PyTorch (CUDA-enabled)
```bash
PyTorch: 2.8.0
torchvision: 0.23.0
CUDA Support: ✅ Enabled
cuDNN: 9.3.0
```

**Installation Source:**
- Custom wheel built for JetPack 6.2 with CUDA 12.6

**Verification:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
# Output:
# PyTorch: 2.8.0
# CUDA Available: True
# CUDA Version: 12.6
```

---

## ⚡ ONNX Runtime GPU (CRITICAL)

### Version & Providers
```bash
ONNX Runtime: 1.20.0 (GPU-enabled)
Available Providers:
  - TensorrtExecutionProvider ✅
  - CUDAExecutionProvider ✅
  - CPUExecutionProvider
```

### Installation (DO NOT USE pip install onnxruntime)

**❌ WRONG (CPU-only):**
```bash
pip3 install onnxruntime  # This will break GPU acceleration!
```

**✅ CORRECT (GPU-enabled wheel for Jetson):**
```bash
# Uninstall any existing ONNX Runtime
pip3 uninstall onnxruntime onnxruntime-gpu -y

# Download GPU-enabled wheel for JetPack 6 + CUDA 12.6
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# Install the wheel
pip3 install onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

**Verification:**
```python
import onnxruntime as ort
print(f"ONNX Runtime: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")

# Should output:
# ONNX Runtime: 1.20.0
# Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Why This is Critical:**
- Face detection (SCRFD) runs on ONNX Runtime
- Face recognition (ArcFace) runs on ONNX Runtime
- Without GPU providers, performance drops 10-20x slower
- System uses CUDAExecutionProvider for all inference

---

## 📦 Complete Python Dependencies

### Core ML/AI Stack
```
torch==2.8.0
torchvision==0.23.0
onnxruntime-gpu==1.20.0 (from wheel above)
onnx==1.17.0
diffusers==0.25.0
transformers==4.36.2
```

### Computer Vision
```
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
facenet-pytorch==2.5.3
```

### Face Recognition Models
```
# Models are loaded via ONNX Runtime:
# - SCRFD (face detection): det_10g.onnx
# - ArcFace (recognition): w600k_r50.onnx
```

### Web Framework
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
```

### Database
```
sqlalchemy==2.0.23
```

### Other Dependencies
```
pillow==10.1.0
numpy==1.24.3
```

---

## 🚀 Installation Steps

### 1. System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0
```

### 2. Verify CUDA Installation
```bash
# Check CUDA compiler
nvcc --version
# Should show: Cuda compilation tools, release 12.6

# Check GPU
nvidia-smi  # or jetson_release for Jetson-specific info

# Test CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Install ONNX Runtime GPU (CRITICAL STEP)
```bash
# Remove any existing ONNX Runtime
pip3 uninstall onnxruntime onnxruntime-gpu -y

# Download correct wheel for JetPack 6.2 + CUDA 12.6
cd /tmp
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# Install
pip3 install onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# Verify (MUST show CUDAExecutionProvider)
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### 4. Install PyTorch (if not already installed)
```bash
# PyTorch 2.8.0 with CUDA 12.6 support
# Use NVIDIA's official PyTorch wheel for JetPack 6.2
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/
```

### 5. Install Project Dependencies
```bash
cd /home/mujeeb/Downloads/face-recognition-security-system

# Install all requirements
pip3 install -r requirements.txt
```

### 6. Configure Environment Variables
Add to `~/.bashrc` or in `start_server.sh`:
```bash
# CUDA paths for PyTorch + ONNX Runtime + Stable Diffusion
export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Fix OpenBLAS/OpenMP threading conflicts (prevents hangs)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
```

### 7. Download Models
```bash
# Models are automatically downloaded on first run
# Stored in: ~/.cache/huggingface/

# Face detection/recognition models (ONNX):
# - models/det_10g.onnx (SCRFD face detector)
# - models/w600k_r50.onnx (ArcFace face recognition)

# Stable Diffusion models:
# - runwayml/stable-diffusion-v1-5 (~4GB, auto-downloaded)
```

---

## 🐛 Common Issues & Solutions

### Issue 1: ONNX Runtime shows only CPUExecutionProvider
**Symptoms:**
- Face detection/recognition very slow
- Logs show "CPUExecutionProvider" instead of "CUDAExecutionProvider"

**Solution:**
```bash
# You installed the wrong ONNX Runtime!
pip3 uninstall onnxruntime onnxruntime-gpu -y
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

### Issue 2: PyTorch CUDA not available
**Symptoms:**
- `torch.cuda.is_available()` returns False

**Solution:**
```bash
# Check CUDA installation
ls /usr/local/cuda/lib64/libcudart.so*

# Reinstall PyTorch with CUDA support for JetPack
# Use official NVIDIA wheel for Jetson
```

### Issue 3: Server hangs on startup
**Symptoms:**
- Server starts but freezes during model loading

**Solution:**
```bash
# OpenBLAS threading conflict
# Add these to start_server.sh:
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
```

### Issue 4: Stable Diffusion out of memory
**Symptoms:**
- CUDA out of memory error during SD generation

**Solution:**
```python
# In generative_augmentation.py, enable attention slicing:
pipeline.enable_attention_slicing(slice_size=1)

# Use FP16 precision (already enabled):
torch_dtype=torch.float16
```

---

## 📊 Performance Benchmarks (Jetson AGX Orin)

### Face Detection (SCRFD - GPU)
- Single face: ~15-25ms
- Multiple faces: ~30-50ms
- Resolution: 640x640

### Face Recognition (ArcFace - GPU)
- Embedding extraction: ~10-20ms per face
- Database matching: <1ms for 100 persons

### Stable Diffusion (FP16)
- Image generation (512x512): ~1.5-3s per image
- GPU Memory: ~6-8GB
- 5 variations: ~10-15 seconds total

---

## 🔍 System Verification Script

Save as `verify_system.py` and run to check everything:

```python
#!/usr/bin/env python3
"""System verification script for Face Recognition System"""

print("=" * 80)
print("JETSON FACE RECOGNITION SYSTEM - VERIFICATION")
print("=" * 80)

# 1. Check PyTorch + CUDA
print("\n[1/5] Checking PyTorch + CUDA...")
import torch
print(f"  ✓ PyTorch: {torch.__version__}")
print(f"  ✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ✓ CUDA Version: {torch.version.cuda}")
    print(f"  ✓ cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("  ✗ CUDA NOT AVAILABLE!")

# 2. Check ONNX Runtime
print("\n[2/5] Checking ONNX Runtime...")
import onnxruntime as ort
print(f"  ✓ ONNX Runtime: {ort.__version__}")
providers = ort.get_available_providers()
print(f"  ✓ Providers: {providers}")
if 'CUDAExecutionProvider' in providers:
    print("  ✅ GPU acceleration ENABLED")
else:
    print("  ❌ GPU acceleration DISABLED - reinstall onnxruntime-gpu!")

# 3. Check OpenCV
print("\n[3/5] Checking OpenCV...")
import cv2
print(f"  ✓ OpenCV: {cv2.__version__}")

# 4. Check Diffusers
print("\n[4/5] Checking Diffusers...")
import diffusers
print(f"  ✓ Diffusers: {diffusers.__version__}")

# 5. Check Models
print("\n[5/5] Checking Models...")
import os
models = [
    "models/det_10g.onnx",
    "models/w600k_r50.onnx"
]
for model in models:
    if os.path.exists(model):
        print(f"  ✓ {model}")
    else:
        print(f"  ✗ {model} - MISSING!")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
```

Run with:
```bash
python3 verify_system.py
```

---

## 📝 Notes

- Always use the GPU-enabled ONNX Runtime wheel from Ultralytics
- Never install `onnxruntime` from pip - it's CPU-only
- Keep `LD_LIBRARY_PATH` configured for CUDA libraries
- Monitor GPU memory with `nvidia-smi` or `tegrastats`
- Use FP16 for Stable Diffusion to save memory

---

## 🆘 Support Resources

- **NVIDIA Jetson Forums:** https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
- **PyTorch for Jetson:** https://forums.developer.nvidia.com/t/pytorch-for-jetson/
- **ONNX Runtime Issues:** https://github.com/microsoft/onnxruntime/issues
- **Project Repository:** [Your GitHub URL]
