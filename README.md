# Face Recognition Security System

Production-ready face recognition system for security applications using Hikvision IP camera on NVIDIA Jetson AGX Orin.

## 🎯 Project Overview

This system is designed for security purposes, capable of identifying individuals from a single reference image (like NADRA database records or CCTV footage). It leverages GPU-accelerated face detection, deep learning-based recognition, and AI-powered data augmentation to achieve accurate recognition even with limited training data.

### Key Features

- **GPU-Accelerated Detection**: SCRFD face detection optimized with TensorRT on Jetson AGX Orin
- **High-Accuracy Recognition**: InsightFace (ArcFace) with 512-D embeddings
- **AI-Powered Data Augmentation**: Stable Diffusion + ControlNet for generating multiple angles from single images
- **Camera Capture Enrollment**: Capture images directly from live camera for registration
- **Multi-Image Support**: Upload multiple images per person for improved accuracy
- **Real-time Processing**: Live MJPEG stream with real-time detection and recognition
- **Alert System**: Configurable alerts for known/unknown persons
- **Admin Dashboard**: Web-based interface for person management and system monitoring
- **SD Card Portability**: Portable data storage for easy deployment and scaling
- **RESTful API**: FastAPI-based backend for easy integration

## 🔧 Hardware Requirements

- **Computing Platform**: NVIDIA Jetson AGX Orin (64GB recommended)
- **Camera**: Hikvision DS-2CD7A47EWD-XZS (4MP Fisheye)
- **Network**: Camera accessible via IP (192.168.1.64)
- **Storage**: SD card recommended for portable database and image storage
- **OS**: JetPack 6.1 (L4T 36.4.0)

## 🚀 Technology Stack

### Backend
- **Framework**: FastAPI (async)
- **Computer Vision**: OpenCV (CUDA-enabled)
- **Deep Learning**: ONNX Runtime with TensorRT execution provider
- **Database**: SQLAlchemy ORM (SQLite for development, PostgreSQL-ready)
- **Streaming**: RTSP via OpenCV, MJPEG for browser

### AI Models
- **Face Detection**: SCRFD (GPU-accelerated with TensorRT FP16)
- **Face Recognition**: InsightFace ArcFace (buffalo_l model)
- **Data Augmentation (Planned)**: Stable Diffusion 1.5 + ControlNet (pose-guided generation)

### Optimization
- **TensorRT**: FP16 optimization for SCRFD and InsightFace models
- **CUDA Streams**: Parallel GPU execution
- **Frame Skipping**: Intelligent frame processing (every 2nd frame)
- **Recognition Throttling**: Every 5th frame for recognition (cached in between)

## 📁 Project Structure

```
face-recognition-security-system/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Pydantic configuration settings
│   ├── models/
│   │   └── database.py            # SQLAlchemy database models
│   ├── api/routes/
│   │   ├── detection.py           # Face detection endpoints
│   │   ├── recognition.py         # Enrollment & recognition endpoints
│   │   ├── alerts.py              # Alert configuration endpoints
│   │   └── websocket.py           # WebSocket for real-time updates
│   ├── core/
│   │   ├── detector.py            # SCRFD face detector
│   │   ├── recognizer.py          # InsightFace ArcFace recognizer
│   │   ├── camera.py              # Camera handler (RTSP)
│   │   ├── augmentation.py        # Traditional augmentation
│   │   ├── alerts.py              # Alert manager
│   │   └── database.py            # Database session management
│   ├── static/                    # Web UI files (admin.html, dashboard.html)
│   └── utils/                     # Utility functions
├── data/
│   ├── images/                    # Reference face images
│   ├── alert_snapshots/           # Alert snapshot images
│   ├── models/                    # Model files (yolov8n.pt, etc.)
│   ├── tensorrt_engines/          # TensorRT cached engines
│   └── sd_card_ready/             # For SD card deployment
├── scripts/
│   ├── migration/                 # Database migration scripts
│   │   └── init_db.py
│   └── utilities/                 # Debug and capture utilities
│       ├── capture_live_frame.py
│       ├── capture_test_frame.py
│       └── debug_recognition.py
├── tests/
│   ├── integration/               # Integration tests
│   ├── unit/                      # Unit tests (TBD)
│   └── fixtures/                  # Test fixtures
├── docs/
│   ├── api/                       # API documentation
│   ├── deployment/                # Deployment guides
│   │   └── JETPACK_UPGRADE_GUIDE.md
│   └── development/               # Development documentation
├── archive/
│   ├── old_docs/                  # Archived documentation
│   └── backup_configs/            # Pre-upgrade configuration backups
├── PROJECT_PLAN.md                # Detailed project roadmap
├── ARCHITECTURE.md                # System architecture documentation
├── TECHNOLOGY_STACK.md            # Complete technology stack
├── CURRENT_STATUS.md              # Current development status
├── requirements.txt               # Python dependencies
└── .env.example                   # Environment variables template
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

# Install InsightFace models (buffalo_l)
# Models will be downloaded automatically on first run to ~/.insightface/
```

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your camera credentials
nano .env
```

Required environment variables:
```env
CAMERA_IP=192.168.1.64
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_password
CAMERA_RTSP_PORT=554

DATABASE_URL=sqlite:///./data/face_recognition.db
FACE_RECOGNITION_THRESHOLD=0.35
```

### 4. Initialize Database

```bash
# Run database initialization script
python3 scripts/migration/init_db.py
```

### 5. Run the Application

```bash
# Start the FastAPI server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Live Stream**: http://localhost:8000/live
- **Admin Panel**: http://localhost:8000/admin
- **Dashboard**: http://localhost:8000/dashboard

## 🎥 Camera Configuration

The system uses Hikvision DS-2CD7A47EWD-XZS IP camera via RTSP:

```
Main Stream: rtsp://username:password@192.168.1.64:554/Streaming/Channels/101
Sub Stream:  rtsp://username:password@192.168.1.64:554/Streaming/Channels/102
```

The system uses **sub-stream** by default for better performance (640x480). Camera credentials are stored in `.env` file (not committed to git).

## 📊 Current Development Status

**Current Phase**: Phase 3 - Production Enhancements

### ✅ Completed Milestones

**Milestone 1: Core Face Recognition System**
- FastAPI application with SQLite database
- Hikvision IP camera RTSP integration
- InsightFace (ArcFace) face recognition
- Person enrollment and recognition APIs
- Admin panel for person management
- Alert system with configurable thresholds
- Live MJPEG video stream with real-time recognition

**Milestone 2: GPU Acceleration**
- SCRFD GPU-accelerated face detection with TensorRT
- JetPack 6.1 upgrade (L4T 36.4.0)
- TensorRT FP16 optimization for SCRFD models
- Performance: ~15-20 FPS with GPU detection
- Removed multi-agent parallel system (simplified architecture)

### 🚧 In Progress

**Phase 3: Production Enhancements**
- SD card data portability system
- Camera capture enrollment in admin panel
- Image quality enhancement pipeline
- Stable Diffusion + ControlNet augmentation

### 📋 Roadmap

**Phase 4: AI-Powered Data Augmentation**
- Stable Diffusion 1.5 integration
- ControlNet for pose-guided generation
- Generate 5-10 angles per enrolled person
- Automatic quality filtering of generated images

**Phase 5: Production Deployment**
- PostgreSQL migration
- Docker containerization
- System monitoring and logging
- Backup and recovery procedures

**Phase 6: Advanced Features**
- JWT-based authentication
- Multi-camera support
- Advanced alert rules (time-based, zone-based)
- Export recognition logs to CSV

## 🔌 API Endpoints

### System Endpoints
```bash
GET  /                       # API information
GET  /health                 # Health check
GET  /docs                   # Swagger UI documentation
GET  /live                   # Live stream web viewer
GET  /admin                  # Admin panel
GET  /dashboard              # Dashboard with recent detections
```

### Face Enrollment
```bash
# Single image enrollment
POST /api/enroll
Form Data:
  - name: string
  - cnic: string (unique identifier)
  - file: UploadFile

# Multi-image enrollment (1-10 images)
POST /api/enroll/multiple
Form Data:
  - name: string
  - cnic: string
  - files: List[UploadFile]
  - use_augmentation: boolean (default: true)

# Camera-based enrollment (captures multiple frames)
POST /api/enroll/camera
Form Data:
  - name: string
  - cnic: string
  - num_captures: int (3-10 frames, default: 5)
  - use_augmentation: boolean (default: true)
```

### Face Recognition
```bash
# Recognize from uploaded image
POST /api/recognize
Form Data:
  - file: UploadFile

# Live video stream with real-time recognition
GET /api/stream/live
# Returns: multipart/x-mixed-replace MJPEG stream
# Features:
#   - Real-time SCRFD face detection (GPU)
#   - ArcFace face recognition
#   - Green boxes for known persons
#   - Red boxes for unknown persons
#   - Confidence scores displayed
```

### Person Management
```bash
GET    /api/persons          # List all enrolled persons
DELETE /api/persons/{id}     # Delete person (cascade deletes embeddings)
GET    /api/access-logs      # Get recognition logs (last 50)
```

### Alert Configuration
```bash
GET  /api/alerts/config      # Get current alert configuration
POST /api/alerts/config      # Update alert configuration
GET  /api/alerts/recent      # Get recent alerts (last 20)
```

### WebSocket (Real-time Updates)
```bash
WS /ws/alerts                # WebSocket for real-time alert notifications
# Sends JSON messages:
# {"type": "alert", "data": {...}}
```

## 📈 Performance Metrics

### Current Performance (JetPack 6.1 + GPU)
- **Live Stream FPS**: ~15-20 FPS
- **Face Detection**: ~30-50ms per frame (SCRFD GPU + TensorRT FP16)
- **Face Recognition**: ~200-300ms per face (InsightFace ArcFace)
- **Detection Frequency**: Every 2nd frame (50% frame skip)
- **Recognition Frequency**: Every 5th processed frame (cached between)
- **Multi-Face Support**: Yes (up to 10 faces per frame)
- **GPU Utilization**: 40-60% during active detection

### Target Performance (After Full Optimization)
- **Live Stream FPS**: 25-30 FPS
- **Face Detection**: <20ms per frame
- **Face Recognition**: <100ms per face
- **GPU Utilization**: 70-80%
- **Accuracy**: >95% with multi-image + SD augmentation

## 🧪 Testing

```bash
# Run integration tests
python3 -m pytest tests/integration/ -v

# Test camera connection
python3 scripts/utilities/capture_test_frame.py

# Test face detection
python3 tests/integration/test_face_detection.py

# Test live stream
python3 tests/integration/test_live_stream.py
```

## 🔐 Security Considerations

- **Data Protection**: All camera credentials stored in `.env` (not committed)
- **Audit Logging**: All recognition attempts logged with timestamps
- **Alert System**: Configurable thresholds for unknown person detection
- **Privacy Compliance**: Local processing, no cloud dependencies
- **Access Control**: Admin panel for authorized personnel only

## 🤝 Contributing

This is a security-focused project. Please ensure all contributions:
- Follow security best practices
- Include appropriate tests
- Update documentation (see docs/development/DOCUMENTATION_MAINTENANCE.md)
- Respect privacy and data protection guidelines

## 📝 License

[Specify your license here - MIT, Apache 2.0, etc.]

## 📧 Contact

For questions or support, please contact: [Your contact information]

## 🙏 Acknowledgments

- NVIDIA Jetson platform for edge AI computing
- Hikvision for professional camera hardware
- InsightFace team for excellent face recognition models (ArcFace)
- SCRFD authors for GPU-optimized face detection
- FastAPI framework for modern API development
- TensorRT team for inference optimization

---

**⚠️ Important**: This system is designed for authorized security applications only. Ensure compliance with local privacy laws and regulations (GDPR, CCPA, etc.).
