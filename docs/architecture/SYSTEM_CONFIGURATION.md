# System Configuration & Current State

**Last Updated:** November 5, 2025
**Version:** 0.7.0 (70% Complete - Production Core Features)

---

## 📋 Current System Specifications

### Hardware Platform
- **Device:** NVIDIA Jetson AGX Orin (64GB)
- **Architecture:** aarch64 (ARM64)
- **JetPack Version:** 6.1 (L4T R36.4.0)
- **Linux Kernel:** 5.15.148-tegra
- **Python:** 3.10.12

### GPU & CUDA Stack
```
CUDA Version: 12.6
cuDNN: 9.x
CUDA Compute: Enabled
TensorRT: Enabled
```

### ML Framework Versions (ACTUAL)
```
PyTorch: 2.8.0 (CUDA 12.6)
ONNX Runtime GPU: 1.20.0 (TensorRT EP + CUDA EP)
OpenCV: 4.10.0
InsightFace: Latest (buffalo_l models)
```

### Computer Vision
```
OpenCV: 4.10.0 (CUDA-enabled)
InsightFace: buffalo_l model pack
Face Detection: SCRFD (det_10g.onnx)
Face Recognition: ArcFace (w600k_r50.onnx)
```

---

## 🔑 Critical Configuration

### ONNX Runtime GPU Setup
**Current Installation:**
```bash
Package: onnxruntime-gpu 1.20.0
Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Verification:**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# MUST show: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Environment Variables
```bash
# CUDA Library Paths
export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH

# Threading Configuration (prevents conflicts)
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
```

---

## 🤖 AI Models & Inference (ACTUALLY IMPLEMENTED)

### Face Detection (SCRFD)
```
Model: det_10g.onnx (SCRFD)
Location: ~/.insightface/models/buffalo_l/
Input Size: 640x640
Precision: FP16 (TensorRT optimized)
Execution: CUDAExecutionProvider
Performance: ~30-50ms per frame
```

### Face Recognition (ArcFace)
```
Model: w600k_r50.onnx (ArcFace)
Location: ~/.insightface/models/buffalo_l/
Embedding Size: 512-D
Precision: FP32 (CPU)
Execution: CPUExecutionProvider
Performance: ~200-300ms per face
Matching Threshold: 0.35 (configurable)
```

**Note:** Recognition runs on CPU - TensorRT optimization planned for future

---

## 📊 Performance Metrics (ACTUAL)

### Real-Time Processing
- **Frame Rate:** 15-20 FPS (live RTSP stream)
- **Detection Time:** ~30-50ms per frame (SCRFD + TensorRT)
- **Recognition Time:** ~200-300ms per face (ArcFace on CPU)
- **GPU Utilization:** 40-60% during active detection
- **Max Faces:** Up to 10 faces per frame
- **End-to-End Latency:** <500ms

### Recognition Accuracy
- **Single Image Enrollment:** ~90-95%
- **Multi-Image Enrollment:** ~95-98%
- **Recognition Threshold:** 0.35 (cosine similarity)

### Memory Usage
- **Idle:** ~2-3GB
- **Processing (Live Stream):** ~4-5GB
- **Peak:** ~6GB

---

## 🗄️ Data Storage

### Database
```
Type: SQLite
Location: ./face_recognition.db
Size: ~2-5MB (grows with enrollments)

Tables:
  - persons (enrollment data)
  - face_embeddings (512-D vectors)
  - recognition_logs (audit trail)
  - alerts (security events)
  - system_configuration (runtime config)
  - users (JWT authentication)
```

### Images & Snapshots
```
Directories:
  - data/images/         - Enrolled person photos
  - data/alert_snapshots/ - Alert snapshots

Format: JPEG
Naming:
  - Person photos: {cnic}_{uuid}.jpg
  - Alert snapshots: alert_{id}_{timestamp}.jpg
```

---

## 🌐 Network Configuration

### Camera Source (RTSP)
```
**CURRENT CAMERA IP: 192.168.1.64**

Main Stream (Channel 101):  rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
Sub Stream (Channel 102):   rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102

Current Use: Main stream (101) - high quality
Resolution: Auto-negotiated by camera
Frame Buffer: Default OpenCV buffer
```

### API Server
```
Host: 0.0.0.0
Port: 8000
Protocol: HTTP (uvicorn ASGI)
Workers: 1 (single-threaded for GPU)

Web Pages:
  - Dashboard: http://192.168.0.117:8000/dashboard
  - Admin Panel: http://192.168.0.117:8000/admin
  - Alerts: http://192.168.0.117:8000/alerts
  - Reports: http://192.168.0.117:8000/reports
  - Live Stream: http://192.168.0.117:8000/live
  - API Docs: http://192.168.0.117:8000/docs

API Endpoints:
  - Authentication: /api/auth/*
  - Persons: /api/persons/*
  - Recognition: /api/recognize/*
  - Alerts: /api/alerts/*
  - Analytics: /api/analytics/*
  - Detection: /api/detect/*
  - WebSocket: /ws/alerts
```

### Authentication
```
Method: JWT (JSON Web Tokens)
Default User: admin / admin123
Token Expiry: 7 days
Role-Based Access: Admin, Operator, Viewer
```

---

## 🔧 Feature Status (ACTUAL CURRENT STATE)

### ✅ Completed & Production-Ready
- [x] Real-time face detection (SCRFD + TensorRT FP16)
- [x] Face recognition (ArcFace 512-D embeddings)
- [x] Person enrollment (single/multiple images)
- [x] Live RTSP camera integration (Hikvision)
- [x] JWT authentication & user management
- [x] Web admin panel (person management)
- [x] Dashboard (real-time stats, live preview)
- [x] Alert management page (filtering, export, acknowledge)
- [x] Reports & analytics page (Chart.js visualizations)
- [x] Live stream viewer (full-screen with recognition)
- [x] Alert system (known/unknown persons, 60s cooldown)
- [x] Recognition audit logging
- [x] Alert snapshots with authenticated serving
- [x] WebSocket real-time notifications
- [x] RESTful API (FastAPI with OpenAPI docs)
- [x] Analytics API (time-series, distributions, person stats)
- [x] Database persistence (SQLite)
- [x] GPU acceleration (CUDA + TensorRT for detection)

### 🚧 Next Up (Planned - NOT YET IMPLEMENTED)
- [ ] System Settings Page (web UI for configuration)
- [ ] SD Card Portability System
- [ ] Enhanced Enrollment Workflow
- [ ] AI Data Augmentation (Stable Diffusion - Future)
- [ ] ControlNet Integration (Future)
- [ ] Multi-camera support
- [ ] PostgreSQL migration option
- [ ] TensorRT optimization for ArcFace recognition

---

## 📁 Project Structure

```
face-recognition-security-system/
├── app/
│   ├── api/
│   │   └── routes/      - API endpoints
│   ├── core/            - Core logic (detector, recognizer, alerts, auth)
│   ├── models/          - Database models & schemas
│   ├── static/          - Web interface HTML files
│   ├── config.py        - Configuration
│   └── main.py          - FastAPI application
├── data/
│   ├── images/          - Person photos
│   └── alert_snapshots/ - Alert snapshots
├── docs/                - Documentation
├── nrtc_faceai/         - Proprietary library
├── face_recognition.db  - SQLite database
├── requirements.txt     - Python dependencies
└── .env                 - Environment variables
```

---

## 🐛 Known Issues & Workarounds

### 1. Recognition Runs on CPU
**Status:** Known Limitation
**Impact:** ~200-300ms per face (vs <50ms with GPU)
**Workaround:** Recognition throttled to every 5th frame for live stream
**Future:** TensorRT optimization planned

### 2. Single Camera Stream Limitation
**Status:** Known Limitation (RTSP)
**Impact:** Only one viewer can access live stream at a time
**Workaround:** Considering frame buffer for multiple viewers
**Future:** Multi-camera support planned

---

## 📝 Configuration Files

### Important Files
```
.env                  - Environment variables (camera IP, settings)
app/config.py         - Application configuration
app/main.py           - FastAPI application entry
requirements.txt      - Python dependencies
```

### Model Paths
```
~/.insightface/models/buffalo_l/
├── det_10g.onnx      - SCRFD face detector
├── w600k_r50.onnx    - ArcFace recognizer
├── 1k3d68.onnx       - 3D landmarks
├── 2d106det.onnx     - 2D landmarks
└── genderage.onnx    - Gender/age estimation
```

---

## 🔄 Recent Changes

### Version 0.7.0 (2025-11-05)
- ✅ Added Reports & Analytics Dashboard with Chart.js
- ✅ Added Analytics API (5 endpoints)
- ✅ Updated navigation across all pages
- ✅ Documentation reorganization

### Version 0.6.0 (2025-11-04)
- ✅ Added Alert Management Page
- ✅ Fixed snapshot authentication
- ✅ Added bulk operations for alerts

### Version 0.5.0 (2025-10-29)
- ✅ JWT authentication system
- ✅ User management
- ✅ Protected API endpoints

---

## ⚠️ Important Notes

1. **Always verify ONNX Runtime GPU** after pip install
2. **Set environment variables** before running (LD_LIBRARY_PATH, OMP settings)
3. **Monitor GPU memory** during live streaming
4. **Backup database** before major updates
5. **Camera IP** is configured in .env file (currently 192.168.1.64)

---

## 🆘 Quick Recovery

If system breaks:

```bash
# 1. Verify ONNX Runtime GPU providers
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# 2. Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Check camera connectivity
ffprobe rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101

# 4. Restart server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Full setup guide: See **[docs/deployment/JETSON_SETUP.md](../deployment/JETSON_SETUP.md)**

---

**This file reflects ACTUAL CURRENT STATE only - no future features listed as "completed"**
