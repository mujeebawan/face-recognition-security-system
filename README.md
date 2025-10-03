# Face Recognition Security System

face recognition system for security applications using Hikvision IP camera on NVIDIA Jetson AGX Orin.

## 🎯 Project Overview

This system is designed for security purposes, capable of identifying individuals from a single reference image (like NADRA database records). It leverages advanced computer vision techniques and diffusion models to achieve accurate recognition even with limited training data.

### Key Features

- **Single Image Recognition**: Advanced augmentation techniques to recognize faces from just one reference image
- **Real-time Processing**: Optimized for NVIDIA Jetson AGX Orin with GPU acceleration
- **High-Quality Camera Support**: Hikvision 4MP IP camera with RTSP streaming
- **RESTful API**: FastAPI-based backend for easy integration
- **Database Ready**: SQLite for development, PostgreSQL-ready for production
- **Augmentation Pipeline**: Traditional and diffusion-based image augmentation

## 🔧 Hardware Requirements

- **Computing Platform**: NVIDIA Jetson AGX Orin
- **Camera**: Hikvision DS-2CD7A47EWD-XZS (4MP Fisheye)
- **Network**: Camera accessible via IP (192.168.1.64)

## 🚀 Technology Stack

### Backend
- **Framework**: FastAPI
- **Computer Vision**: OpenCV, MediaPipe/InsightFace
- **Deep Learning**: PyTorch (ArcFace for face recognition)
- **Database**: SQLAlchemy ORM (SQLite → PostgreSQL)
- **Streaming**: RTSP (via OpenCV)

### AI Models
- **Face Detection**: MediaPipe / RetinaFace
- **Face Recognition**: InsightFace (ArcFace)
- **Augmentation**: Diffusion models (Stable Diffusion + ControlNet)

## 📁 Project Structure

```
face_recognition_system/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── models/                 # Database models
│   ├── api/routes/             # API endpoints
│   ├── core/                   # Core logic (detection, recognition)
│   └── utils/                  # Utilities
├── data/
│   ├── images/                 # Reference images
│   ├── embeddings/             # Face embeddings
│   └── models/                 # Pre-trained models
├── tests/
├── PROJECT_PLAN.md             # Detailed project roadmap
├── requirements.txt
└── .env.example
```

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-recognition-security-system.git
cd face-recognition-security-system
```

### 2. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your camera credentials
nano .env
```

### 4. Initialize Database

```bash
# Run database migrations
alembic upgrade head
```

### 5. Run the Application

```bash
# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🎥 Camera Configuration

The system uses Hikvision DS-2CD7A47EWD-XZS IP camera via RTSP:

```
RTSP URL: rtsp://username:password@192.168.1.64:554/Streaming/Channels/101
```

Camera credentials are stored in `.env` file (not committed to git).

## 📊 Development Phases

See [PROJECT_PLAN.md](PROJECT_PLAN.md) and [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) for detailed development roadmap and progress:

1. ✅ **Phase 1**: Environment Setup & Infrastructure - **COMPLETE**
   - FastAPI application with CORS middleware
   - SQLite database with SQLAlchemy ORM
   - Hikvision IP camera RTSP integration
   - Configuration management with Pydantic

2. ✅ **Phase 2**: Face Detection Pipeline - **COMPLETE**
   - MediaPipe face detection integration
   - Real-time face detection from camera feed
   - API endpoints for detection
   - Bounding box and landmark visualization

3. ✅ **Phase 3**: Face Recognition Core - **COMPLETE**
   - InsightFace (ArcFace) integration
   - 512-D face embedding extraction
   - Person enrollment and recognition APIs
   - Database storage for embeddings
   - Recognition audit logging

4. ✅ **Phase 4A**: Multi-Image Enrollment & Live Streaming - **COMPLETE**
   - Traditional image augmentation (rotation, brightness, contrast)
   - Multi-image enrollment endpoint (1-10 images)
   - Camera-based enrollment endpoint
   - Live MJPEG video stream with real-time recognition
   - Web UI for live stream viewing
   - Multiple face detection and recognition

5. ⚠️ **Phase 5**: GPU Acceleration & Optimization - **PARTIALLY COMPLETE**
   - ✅ CPU optimizations (MediaPipe + frame skipping)
   - ✅ Two-stage processing pipeline (fast detection + slow recognition)
   - ✅ Performance improved to ~10-15 FPS
   - ❌ GPU acceleration blocked (GLIBC incompatibility with onnxruntime-gpu)
   - Current: CPU-only processing with optimized pipeline

6. ⏳ **Phase 4B**: Advanced Augmentation - **PENDING**
   - Diffusion model integration (Stable Diffusion + ControlNet)
   - Synthetic face generation for pose variation
   - GAN-based augmentation

7. ⏳ **Phase 6**: Real-time Recognition Enhancement - **PARTIALLY COMPLETE**
   - ✅ Live video stream with recognition
   - ⏳ Multi-client streaming support
   - ⏳ Alert system for unknown persons
   - ⏳ Recognition confidence tuning interface

8. ⏳ **Phase 7**: Production Optimization - **PENDING**
   - TensorRT model conversion (when GPU available)
   - Advanced caching strategies
   - Database query optimization
   - PostgreSQL migration

9. ⏳ **Phase 8**: Security & Production Features - **PENDING**
   - JWT-based authentication
   - API rate limiting
   - Data encryption for embeddings
   - Backup and recovery system

10. ⏳ **Phase 9**: UI/Frontend - **BASIC COMPLETE**
    - ✅ Basic live stream viewer
    - ⏳ Full-featured admin dashboard
    - ⏳ Person management interface
    - ⏳ Recognition history viewer

## 🔌 API Endpoints

### System Endpoints
```bash
GET  /                       # API information
GET  /health                 # Health check
GET  /docs                   # Swagger UI documentation
GET  /live                   # Live stream web viewer
```

### Face Detection
```bash
POST /api/detect-faces       # Detect faces in uploaded image
GET  /api/camera/snapshot    # Capture frame with optional detection overlay
GET  /api/camera/detect      # Quick face detection from camera
```

### Face Enrollment
```bash
# Single image enrollment
POST /api/enroll
Body: {
  "name": "John Doe",
  "cnic": "12345-1234567-1",
  "image": "base64_encoded_image"
}

# Multi-image enrollment (1-10 images)
POST /api/enroll/multiple
Form Data:
  - name: string
  - cnic: string
  - files: List[UploadFile]
  - use_augmentation: boolean (default: true)

# Camera-based enrollment (captures multiple frames)
POST /api/enroll/camera
Body: {
  "name": "John Doe",
  "cnic": "12345-1234567-1",
  "num_captures": 5,           # 3-10 frames
  "use_augmentation": true
}
```

### Face Recognition
```bash
# Recognize from uploaded image
POST /api/recognize
Body: {
  "image": "base64_encoded_image"
}

# Recognize from camera
GET /api/recognize/camera
```

### Live Video Stream
```bash
# MJPEG video stream with real-time recognition
GET /api/stream/live
# Returns: multipart/x-mixed-replace MJPEG stream
# Features:
#   - Real-time face detection and recognition
#   - Green boxes for known persons
#   - Red boxes for unknown persons
#   - Confidence scores displayed
```

### Person Management
```bash
GET    /api/persons          # List all enrolled persons
DELETE /api/persons/{id}     # Delete person (cascade deletes embeddings)
```

## 🔐 Security Considerations

- **Data Encryption**: Face embeddings are encrypted at rest
- **API Authentication**: JWT-based authentication
- **Access Control**: Role-based access control (RBAC)
- **Audit Logging**: All recognition attempts are logged
- **Privacy Compliance**: GDPR-compliant data handling

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## 📈 Performance Metrics

### Current Performance (CPU-Optimized)
- **Live Stream FPS**: ~10-15 FPS
- **Face Detection**: ~5-10ms per frame (MediaPipe)
- **Face Recognition**: ~300-400ms per frame (InsightFace, CPU)
- **Recognition Frequency**: Every 20th frame
- **Frame Processing**: Every 2nd frame (50% skip rate)
- **Multi-Face Support**: Yes (IoU-based matching)

### Target Performance (GPU-Accelerated)
- **Face Detection**: >20 FPS
- **Recognition Latency**: <100ms
- **GPU Utilization**: >80%
- **Accuracy**: >95% with multi-image augmentation

### Known Limitations
- GPU acceleration blocked by GLIBC incompatibility (JetPack 5.1.2)
- Single camera stream (concurrent access limited)
- CPU-only processing currently in use

## 🤝 Contributing

This is a security-focused project. Please ensure all contributions:
- Follow security best practices
- Include appropriate tests
- Update documentation
- Respect privacy and data protection guidelines

## 📝 License

[Specify your license here - MIT, Apache 2.0, etc.]

## 📧 Contact

For questions or support, please contact: [Your contact information]

## 🙏 Acknowledgments

- NVIDIA Jetson platform for edge AI computing
- Hikvision for professional camera hardware
- InsightFace team for excellent face recognition models
- FastAPI framework for modern API development

---

**⚠️ Important**: This system is designed for authorized security applications only. Ensure compliance with local privacy laws and regulations.
