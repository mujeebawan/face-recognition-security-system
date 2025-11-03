# System Configuration & Current State

**Last Updated:** 2025-11-03
**Version:** Phase 5.4 (Stable Diffusion Img2Img + ControlNet Ready)

---

## 📋 Current System Specifications

### Hardware Platform
- **Device:** NVIDIA Jetson AGX Orin
- **Architecture:** aarch64 (ARM64)
- **JetPack Version:** 6.2 (L4T R36.4.7)
- **Linux Kernel:** 5.15.148-tegra
- **Python:** 3.10.12

### GPU & CUDA Stack
```
CUDA Version: 12.6 (Release 12.6.68)
cuDNN: 9.3.0
CUDA Compute: Enabled
TensorRT: Enabled
```

### ML Framework Versions
```
PyTorch: 2.8.0 (CUDA 12.6)
torchvision: 0.23.0
ONNX Runtime GPU: 1.20.0 (CUDAExecutionProvider + TensorRT)
Diffusers: 0.25.0
Transformers: 4.36.2
```

### Computer Vision
```
OpenCV: 4.9.0.80 (with contrib modules)
facenet-pytorch: 2.5.3
```

---

## 🔑 Critical Configuration

### ONNX Runtime GPU Setup
**❗ IMPORTANT:** The system requires GPU-enabled ONNX Runtime for face detection/recognition.

**Current Installation:**
```bash
Package: onnxruntime-gpu 1.20.0
Wheel: onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
Source: https://github.com/ultralytics/assets/releases/download/v0.0.0/
Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Verification:**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# MUST show: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Re-installation (if needed):**
```bash
# Remove any existing installation
pip3 uninstall onnxruntime onnxruntime-gpu -y

# Install GPU-enabled wheel
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# Verify
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Environment Variables
```bash
# CUDA Library Paths (required for PyTorch, ONNX Runtime, Stable Diffusion)
export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Threading Configuration (prevents OpenBLAS/OpenMP conflicts)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
```

These are configured in `start_server.sh` and applied automatically.

---

## 🤖 AI Models & Inference

### Face Detection (SCRFD)
```
Model: det_10g.onnx
Location: models/det_10g.onnx
Input Size: 640x640
Precision: FP16 (TensorRT optimized)
Execution: CUDAExecutionProvider
Performance: 15-25ms per frame
```

### Face Recognition (ArcFace)
```
Model: w600k_r50.onnx
Location: models/w600k_r50.onnx
Embedding Size: 512-D
Precision: FP16
Execution: CUDAExecutionProvider
Performance: 10-20ms per face
```

### Generative Augmentation (Stable Diffusion)
```
Model: runwayml/stable-diffusion-v1-5
Mode: Img2Img Pipeline (identity-preserving)
Cache: ~/.cache/huggingface/
Precision: FP16
Memory: ~6-8GB VRAM
Performance: 1.5-3s per 512x512 image
Optimizations: Attention slicing enabled
```

**Current Status:** ✅ Img2Img mode enabled (Phase 5.4)
**Upcoming:** 🚧 ControlNet integration for precise pose control

---

## 📊 Performance Metrics

### Real-Time Processing
- **Frame Rate:** 15-20 FPS (live RTSP stream)
- **Latency:** <100ms end-to-end
- **Max Faces:** 10 concurrent faces per frame
- **Recognition Speed:** <50ms per face

### Enrollment Processing
- **Single Image:** <1s (original embedding only)
- **With AI Augmentation:** 10-15s (5 angle variations via SD)
- **Multi-Image Manual:** ~2-3s (5 manual captures)

### Memory Usage
- **Idle:** ~3GB
- **Processing:** ~4-5GB
- **With SD Loaded:** ~10-12GB
- **Peak:** ~14GB (SD generation active)

---

## 🗄️ Data Storage

### Database
```
Type: SQLite
Location: data/face_recognition.db
Tables:
  - persons (enrollment data)
  - face_embeddings (512-D vectors)
  - recognition_logs (audit trail)
  - alerts (security events)
  - system_configuration (runtime config)
```

### Images
```
Directory: data/images/
Format: JPEG
Naming:
  - Original: {cnic}_{filename}.jpg
  - SD Generated: {cnic}_sd_gen_{n}.jpg
  - Camera Captured: {cnic}_camera_{angle}_{timestamp}.jpg
```

### Logs
```
Server Log: server.log
Format: Text (timestamped)
Rotation: Manual
```

---

## 🌐 Network Configuration

### Camera Source (RTSP)
```
Main Stream (Channel 101): rtsp://admin:password@192.168.0.107:554/Streaming/Channels/101
Sub Stream (Channel 102):  rtsp://admin:password@192.168.0.107:554/Streaming/Channels/102

Current Use: Sub-stream (102) for better performance
Resolution: 640x480 (auto-negotiated)
Frame Buffer: 2 frames (configurable)
```

### API Server
```
Host: 0.0.0.0
Port: 8000
Protocol: HTTP (uvicorn ASGI)
Workers: 1 (single-threaded for GPU)

Endpoints:
  - Admin Panel: /admin
  - Live Stream: /live
  - Dashboard: /dashboard
  - API Docs: /docs
  - Camera Snapshot: /api/camera/snapshot
  - Recognition: /api/recognize
  - Enrollment: /api/enroll
```

### Access URLs
```
Local: http://localhost:8000
Network: http://192.168.0.117:8000
```

---

## 🔧 Feature Status

### ✅ Completed (Production Ready)
- [x] Real-time face detection (SCRFD)
- [x] Face recognition with ArcFace embeddings
- [x] Person enrollment (single/multiple images)
- [x] Live RTSP camera integration
- [x] Web admin panel
- [x] Alert system for unknown persons
- [x] Recognition audit logging
- [x] Camera snapshot capture
- [x] Multi-angle enrollment workflow
- [x] Person details with case history
- [x] Stable Diffusion img2img augmentation
- [x] Database persistence (SQLite)
- [x] GPU acceleration (CUDA + TensorRT)

### 🚧 In Progress
- [ ] ControlNet integration for precise pose control
- [ ] Identity-preserving face angle generation

### 📋 Planned
- [ ] Multi-camera support
- [ ] PostgreSQL migration option
- [ ] Advanced analytics dashboard
- [ ] Face tracking across frames
- [ ] Age/gender estimation
- [ ] Mask detection
- [ ] Mobile app integration

---

## 🔐 Security & Privacy

### Data Protection
- All face embeddings stored encrypted
- Images stored locally (no cloud upload)
- HTTPS support (configurable)
- Role-based access control (planned)

### Compliance
- GDPR-ready data deletion
- Audit logging for all operations
- Retention policy support

---

## 📝 Configuration Files

### Important Files
```
start_server.sh       - Server startup script (with env vars)
stop_server.sh        - Server shutdown script
app/config.py         - Application configuration
app/main.py           - FastAPI application entry
requirements.txt      - Python dependencies
.env                  - Environment variables (optional)
```

### Model Paths
```
models/det_10g.onnx   - SCRFD face detector
models/w600k_r50.onnx - ArcFace recognizer
~/.cache/huggingface/ - Stable Diffusion models
```

---

## 🐛 Known Issues & Workarounds

### Issue 1: Stream Latency
**Status:** ✅ Resolved
**Solution:** Use sub-stream (channel 102) instead of main stream

### Issue 2: Camera Snapshot Cache
**Status:** ✅ Resolved
**Solution:** Cache-busting with timestamps + buffer flushing

### Issue 3: SD Generated Faces Don't Match Person
**Status:** ✅ Resolved (Phase 5.4)
**Solution:** Switched from text-to-image to img2img pipeline

### Issue 4: OpenBLAS Threading Hang
**Status:** ✅ Resolved
**Solution:** Set OMP_NUM_THREADS=4 and OMP_WAIT_POLICY=PASSIVE

---

## 📚 Documentation

- **[JETSON_SETUP.md](JETSON_SETUP.md)** - Complete Jetson setup guide
- **[QUICK_START.md](../QUICK_START.md)** - Quick start commands
- **[README.md](../README.md)** - Project overview
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when server running)

---

## 🔄 Recent Changes

### Phase 5.4 (2025-11-03)
- ✅ Fixed original image display in person details
- ✅ Switched Stable Diffusion to img2img mode for identity preservation
- ✅ Added case history/alerts to person details modal
- ✅ Added comprehensive Jetson setup documentation

### Phase 5.3
- Added Stable Diffusion augmentation to enrollment API
- Implemented UI controls for SD generation

### Phase 5.2
- Initial Stable Diffusion face augmentation module
- FP16 optimization for Jetson

---

## ⚠️ Important Notes

1. **Always use GPU-enabled ONNX Runtime** - CPU version is 10-20x slower
2. **Check CUDA providers** after any pip install that touches onnxruntime
3. **Set environment variables** before running server (done in start_server.sh)
4. **Monitor GPU memory** during SD generation (can use 10-14GB)
5. **Backup database** before major updates

---

## 🆘 Quick Recovery

If system breaks after updates:

```bash
# 1. Stop server
./stop_server.sh

# 2. Verify ONNX Runtime GPU
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# 3. Reinstall if needed (see JETSON_SETUP.md)

# 4. Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Restart server
./start_server.sh
```

Full recovery guide: See [JETSON_SETUP.md](JETSON_SETUP.md)

---

**For detailed installation instructions, see [JETSON_SETUP.md](JETSON_SETUP.md)**
