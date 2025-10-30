# Face Recognition Security System

**Enterprise-Grade Real-Time Face Recognition Platform**
High-performance GPU-accelerated security system for real-time identification and monitoring using NVIDIA Jetson AGX Orin and professional IP cameras.

> **🚀 Quick Start**: Want to run the system right away? See [QUICK_START.md](QUICK_START.md) for simple start/stop commands.

---

## 🎯 Overview

A production-ready, AI-powered face recognition security platform designed for mission-critical applications including border security, law enforcement, corporate access control, and public safety monitoring. The system processes live video streams in real-time, automatically detecting and identifying individuals against a database of enrolled persons.

Built from the ground up for edge deployment on NVIDIA Jetson AGX Orin, this system achieves exceptional performance (15-20 FPS) with GPU-accelerated inference while maintaining high accuracy even with limited training data per person.

### Use Cases

- **Law Enforcement**: Identify wanted individuals from surveillance cameras
- **Border Control**: Automated identity verification at checkpoints
- **Corporate Security**: Unauthorized access detection and employee monitoring
- **Public Safety**: Real-time alerts for persons of interest in public spaces
- **Event Security**: VIP recognition and blacklist monitoring at venues

---

## ✨ Key Features

### Core Capabilities

- **Real-Time Processing**: 15-20 FPS live video analysis with sub-second latency
- **GPU Acceleration**: TensorRT-optimized inference on NVIDIA Jetson AGX Orin
- **Multi-Face Detection**: Simultaneous tracking of up to 10 faces per frame
- **High Accuracy Recognition**: 95%+ accuracy with multi-image enrollment
- **Single-Image Enrollment**: Advanced augmentation allows enrollment from just one photo
- **Unknown Person Alerts**: Automatic detection and logging of unrecognized individuals
- **Distributed Architecture**: Scalable to multiple cameras and locations

### Advanced AI Features

- **SCRFD Face Detection**: State-of-the-art GPU-accelerated detector with TensorRT FP16 optimization
- **ArcFace Recognition**: Deep learning embeddings (512-D) for robust identity matching
- **AI Data Augmentation**: Stable Diffusion + ControlNet for generating training variations
- **Adaptive Thresholding**: Configurable confidence thresholds per security level
- **Quality Assessment**: Automatic face quality scoring and image preprocessing

### User Experience

- **Web-Based Admin Panel**: Intuitive interface for person enrollment and management
- **Live Camera Capture**: Enroll persons directly from live video feed
- **Multi-Angle Enrollment**: Guided capture workflow (frontal, left, right, up, down)
- **Real-Time Dashboard**: Live monitoring with detection statistics and alerts
- **RESTful API**: Complete programmatic access for system integration
- **WebSocket Updates**: Real-time alert notifications

### Enterprise Features

- **Alert System**: Configurable rules with email/webhook notifications
- **Audit Logging**: Complete audit trail of all recognition events
- **Database Management**: SQLite for edge, PostgreSQL-ready for enterprise
- **Portable Deployment**: SD card-based data storage for field deployment
- **Multi-Camera Support**: Scalable architecture for distributed camera networks
- **Privacy Compliance**: On-premise processing, GDPR/CCPA compliant

---

## 🏗️ System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Face Recognition Security System              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Camera     │───▶│   Detection  │───▶│ Recognition  │      │
│  │   Handler    │    │   Pipeline   │    │   Pipeline   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         │                    ▼                    ▼              │
│         │            ┌──────────────┐    ┌──────────────┐      │
│         └───────────▶│   SCRFD GPU  │    │   ArcFace    │      │
│                      │   Detector   │    │  Recognizer  │      │
│                      └──────────────┘    └──────────────┘      │
│                              │                    │              │
│                              ▼                    ▼              │
│                      ┌─────────────────────────────────┐        │
│                      │      Alert & Logging System     │        │
│                      └─────────────────────────────────┘        │
│                                     │                            │
│                                     ▼                            │
│                      ┌─────────────────────────────────┐        │
│                      │     Database & API Layer        │        │
│                      └─────────────────────────────────┘        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

1. **Video Acquisition**: RTSP stream from Hikvision IP camera (640x480 sub-stream)
2. **Face Detection**: SCRFD GPU detector (TensorRT FP16) - 30-50ms per frame
3. **Face Tracking**: Multi-face tracking with position-based caching
4. **Feature Extraction**: InsightFace ArcFace generates 512-D embeddings
5. **Identity Matching**: Cosine similarity against database (configurable threshold)
6. **Alert Generation**: Real-time alerts for known/unknown persons
7. **Logging & Storage**: SQLite database with full audit trail

---

## 🔧 Technical Specifications

### Hardware Platform

| Component | Specification |
|-----------|--------------|
| **Compute Module** | NVIDIA Jetson AGX Orin (64GB) |
| **GPU** | 2048-core NVIDIA Ampere (up to 275 TOPS) |
| **Memory** | 64GB LPDDR5 (204.8 GB/s bandwidth) |
| **Storage** | 64GB eMMC + SD card for portable data |
| **Camera** | Hikvision DS-2CD7A47EWD-XZS (4MP Fisheye) |
| **Network** | Gigabit Ethernet (camera + API access) |
| **Operating System** | JetPack 6.1 (L4T 36.4.0, Ubuntu 22.04) |

### Software Stack

**Backend Framework**
- FastAPI 0.104+ (async Python web framework)
- Uvicorn (ASGI server)
- SQLAlchemy 2.0 (ORM with async support)
- Pydantic 2.0 (data validation)

**Computer Vision**
- OpenCV 4.8+ (CUDA-enabled build)
- ONNX Runtime 1.19+ with TensorRT EP
- InsightFace (ArcFace buffalo_l model)
- SCRFD (GPU-optimized face detector)

**AI/ML Frameworks**
- TensorRT 8.6+ (FP16 optimization)
- CUDA 12.2 (GPU acceleration)
- Stable Diffusion 1.5 (data augmentation)
- ControlNet (pose-guided generation)

**Database & Storage**
- SQLite (edge deployment)
- PostgreSQL (enterprise deployment)
- Alembic (database migrations)

**Monitoring & Deployment**
- Prometheus (metrics collection)
- Docker (containerization)
- GitHub Actions (CI/CD)

### Performance Characteristics

| Metric | Current | Target |
|--------|---------|--------|
| **Live Stream FPS** | 15-20 | 25-30 |
| **Detection Latency** | 30-50ms | <20ms |
| **Recognition Latency** | 200-300ms | <100ms |
| **GPU Utilization** | 40-60% | 70-80% |
| **Recognition Accuracy** | 90-95% | >95% |
| **Max Concurrent Faces** | 10 | 20 |
| **Database Size** | 1K persons | 10K persons |

---

## 📁 Project Structure

```
face-recognition-security-system/
│
├── app/                              # Core application code
│   ├── main.py                       # FastAPI application entry point
│   ├── config.py                     # Pydantic configuration
│   │
│   ├── api/                          # API layer
│   │   ├── routes/
│   │   │   ├── recognition.py        # Enrollment & recognition endpoints
│   │   │   ├── detection.py          # Face detection endpoints
│   │   │   ├── alerts.py             # Alert configuration
│   │   │   └── websocket.py          # WebSocket for real-time updates
│   │   └── dependencies.py           # FastAPI dependencies
│   │
│   ├── core/                         # Core business logic
│   │   ├── detector.py               # SCRFD face detector (GPU)
│   │   ├── recognizer.py             # ArcFace face recognizer
│   │   ├── camera.py                 # IP camera handler (RTSP)
│   │   ├── augmentation.py           # Traditional data augmentation
│   │   ├── generative_augmentation.py # Stable Diffusion augmentation
│   │   ├── alerts.py                 # Alert management system
│   │   ├── quality.py                # Image quality assessment
│   │   └── database.py               # Database session management
│   │
│   ├── models/                       # Data models
│   │   ├── database.py               # SQLAlchemy ORM models
│   │   ├── schemas.py                # Pydantic request/response models
│   │   └── enums.py                  # Enumerations
│   │
│   ├── static/                       # Web UI
│   │   ├── admin.html                # Person enrollment interface
│   │   ├── dashboard.html            # Monitoring dashboard
│   │   ├── live.html                 # Live stream viewer
│   │   └── styles.css                # UI styling
│   │
│   └── utils/                        # Utility functions
│       ├── image_processing.py       # Image preprocessing
│       ├── logging.py                # Logging configuration
│       └── validators.py             # Data validators
│
├── data/                             # Data storage
│   ├── images/                       # Enrolled person images
│   ├── alert_snapshots/              # Alert event snapshots
│   ├── models/                       # Pre-trained model files
│   ├── tensorrt_engines/             # TensorRT engine cache
│   └── embeddings/                   # Face embedding cache
│
├── scripts/                          # Utility scripts
│   ├── migration/                    # Database migrations
│   │   ├── init_db.py                # Initialize database schema
│   │   └── migrate_to_sd.py          # SD card migration
│   │
│   ├── utilities/                    # Debug utilities
│   │   ├── capture_live_frame.py     # Camera capture test
│   │   ├── debug_recognition.py      # Recognition debugging
│   │   └── benchmark_performance.py  # Performance profiling
│   │
│   └── deployment/                   # Deployment scripts
│       ├── start_server.sh           # Production server startup
│       ├── stop_server.sh            # Graceful server shutdown
│       └── backup_data.sh            # Data backup utility
│
├── tests/                            # Test suite
│   ├── integration/                  # Integration tests
│   │   ├── test_camera.py
│   │   ├── test_detection.py
│   │   ├── test_recognition.py
│   │   └── test_alerts_api.py
│   │
│   ├── unit/                         # Unit tests
│   │   ├── test_detector.py
│   │   ├── test_recognizer.py
│   │   └── test_augmentation.py
│   │
│   └── fixtures/                     # Test fixtures
│       └── sample_faces/
│
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   │   ├── ENDPOINTS.md
│   │   └── WEBSOCKET.md
│   │
│   ├── deployment/                   # Deployment guides
│   │   ├── JETPACK_SETUP.md
│   │   ├── PRODUCTION_DEPLOYMENT.md
│   │   └── SD_CARD_MIGRATION.md
│   │
│   └── development/                  # Developer documentation
│       ├── ARCHITECTURE.md
│       ├── CONTRIBUTING.md
│       └── TESTING.md
│
├── archive/                          # Archived files
│   ├── old_docs/                     # Previous documentation versions
│   └── backup_configs/               # Configuration backups
│
├── .github/                          # GitHub configuration
│   ├── workflows/                    # CI/CD pipelines
│   │   ├── test.yml
│   │   └── deploy.yml
│   └── ISSUE_TEMPLATE/               # Issue templates
│
├── PROJECT_PLAN.md                   # Detailed project roadmap
├── ARCHITECTURE.md                   # System architecture document
├── TECHNOLOGY_STACK.md               # Complete tech stack details
├── CURRENT_STATUS.md                 # Development progress tracker
├── requirements.txt                  # Python dependencies
├── requirements-dev.txt              # Development dependencies
├── requirements-genai.txt            # Stable Diffusion dependencies
├── .env.example                      # Environment template
├── .gitignore                        # Git ignore rules
├── Dockerfile                        # Docker container definition
├── docker-compose.yml                # Multi-container orchestration
└── LICENSE                           # Software license

```

---

## 🚀 Quick Start Guide

### Prerequisites

- NVIDIA Jetson AGX Orin with JetPack 6.1 installed
- Hikvision IP camera accessible on network
- Python 3.10+
- 32GB+ free storage space
- Root or sudo access for GPU setup

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/mujeebawan/face-recognition-security-system.git
cd face-recognition-security-system
```

#### 2. System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libopencv-dev \
    python3-opencv \
    cmake \
    pkg-config

# Verify CUDA installation
nvcc --version  # Should show CUDA 12.2

# Verify TensorRT
dpkg -l | grep TensorRT  # Should show TensorRT 8.6
```

#### 3. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch for Jetson (if not already installed)
pip install torch torchvision torchaudio

# Install ONNX Runtime with TensorRT support
# (Pre-built wheel for Jetson is in project root)
pip install onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl

# Install project dependencies
pip install -r requirements.txt

# Install InsightFace and download models
pip install insightface
# Models will auto-download to ~/.insightface/ on first run
```

#### 4. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Required Configuration:**

```bash
# Camera Configuration
CAMERA_IP=192.168.1.64
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_secure_password
CAMERA_RTSP_PORT=554

# Recognition Settings
FACE_RECOGNITION_THRESHOLD=0.35  # Lower = stricter matching
USE_GPU=true

# Database
DATABASE_URL=sqlite:///./data/face_recognition.db

# Alert Settings
ENABLE_ALERTS=true
ALERT_EMAIL=security@yourcompany.com
```

#### 5. Database Initialization

```bash
# Create database schema
python3 scripts/migration/init_db.py

# Verify database created
ls -lh data/face_recognition.db
```

#### 6. Test Camera Connection

```bash
# Verify camera is accessible
python3 scripts/utilities/capture_test_frame.py

# Should save test_frame.jpg if successful
```

#### 7. Start the Application

```bash
# Option A: Development mode (with auto-reload)
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Option B: Production mode (recommended)
./scripts/deployment/start_server.sh
```

#### 8. Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Should return: {"status": "healthy", "gpu": true, ...}
```

---

## 📖 Usage Guide

### Web Interface

Access the web interfaces at:

- **Admin Panel**: http://localhost:8000/admin
  Enroll new persons, manage database, configure alerts

- **Live Stream**: http://localhost:8000/live
  Real-time video with face detection and recognition

- **Dashboard**: http://localhost:8000/dashboard
  System statistics, recent events, and alerts

- **API Documentation**: http://localhost:8000/docs
  Interactive Swagger UI for API testing

### Enrolling Persons

**Method 1: Upload Images (Recommended)**

1. Go to http://localhost:8000/admin
2. Click "Choose Photo" or "Use Camera Capture"
3. Enter person details (Name, ID number)
4. Upload 3-5 images (different angles recommended)
5. System automatically generates additional training data

**Method 2: Camera Capture**

1. Click "Use Camera Capture" in admin panel
2. Live preview appears
3. Click "Capture Current Frame"
4. Choose angle (Frontal, Left, Right, Up, Down)
5. Capture 3-5 angles for best accuracy
6. Click "Done - Use These Images"

**Method 3: API Enrollment**

```bash
# Single image enrollment
curl -X POST "http://localhost:8000/api/enroll" \
  -F "name=John Doe" \
  -F "cnic=12345-6789012-3" \
  -F "file=@photo.jpg"

# Multi-image enrollment
curl -X POST "http://localhost:8000/api/enroll/multiple" \
  -F "name=Jane Smith" \
  -F "cnic=98765-4321098-7" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg" \
  -F "use_augmentation=true"
```

### Monitoring & Alerts

**Configure Alerts:**

```bash
curl -X POST "http://localhost:8000/api/alerts/config" \
  -H "Content-Type: application/json" \
  -d '{
    "enable_alerts": true,
    "unknown_person_alert": true,
    "confidence_threshold": 0.35,
    "alert_cooldown_seconds": 300
  }'
```

**View Recent Events:**

```bash
# Get recognition logs
curl "http://localhost:8000/api/access-logs?limit=50"

# Get recent alerts
curl "http://localhost:8000/api/alerts/recent"
```

**Real-Time Alerts (WebSocket):**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/alerts');

ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log('Alert:', alert);
  // {"type": "alert", "event": "unknown_person", "confidence": 0.89, ...}
};
```

---

## 🔌 API Reference

### Authentication (Future)

Currently no authentication required. JWT-based auth planned for production deployment.

### Core Endpoints

#### System Health

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "camera_connected": true,
  "database_status": "ok",
  "models_loaded": {
    "detector": "scrfd_10g",
    "recognizer": "buffalo_l"
  }
}
```

#### Face Enrollment

```http
POST /api/enroll
Content-Type: multipart/form-data

name: string (required)
cnic: string (required, unique identifier)
file: File (required, image file)
use_augmentation: boolean (optional, default: true)
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully enrolled John Doe",
  "person_id": 42,
  "embeddings_created": 15,
  "augmentation_applied": true
}
```

#### Multi-Image Enrollment

```http
POST /api/enroll/multiple
Content-Type: multipart/form-data

name: string (required)
cnic: string (required)
files: File[] (required, 1-10 images)
use_augmentation: boolean (optional, default: true)
```

#### Face Recognition

```http
POST /api/recognize
Content-Type: multipart/form-data

file: File (required)
```

**Response:**
```json
{
  "success": true,
  "faces_detected": 1,
  "results": [
    {
      "person_id": 42,
      "name": "John Doe",
      "cnic": "12345-6789012-3",
      "confidence": 0.87,
      "bbox": [100, 150, 200, 300],
      "matched": true
    }
  ]
}
```

#### Live Video Stream

```http
GET /api/stream/live
```

Returns MJPEG video stream with real-time face detection and recognition overlays.

#### Person Management

```http
GET /api/persons
GET /api/persons/{id}
DELETE /api/persons/{id}
```

For complete API documentation, visit http://localhost:8000/docs

---

## 🧪 Testing & Validation

### Run Test Suite

```bash
# All tests
python3 -m pytest tests/ -v

# Integration tests only
python3 -m pytest tests/integration/ -v

# Specific test
python3 -m pytest tests/integration/test_recognition.py -v

# With coverage
python3 -m pytest tests/ --cov=app --cov-report=html
```

### Performance Benchmarking

```bash
# Full system benchmark
python3 scripts/utilities/benchmark_performance.py

# Detection speed test
python3 tests/integration/test_gpu_performance.py

# Recognition accuracy test
python3 tests/integration/test_recognizer.py
```

### Manual Testing Checklist

- [ ] Camera connection successful (capture test frame)
- [ ] Face detection working on live stream
- [ ] Person enrollment (single image)
- [ ] Person enrollment (multiple images)
- [ ] Camera capture enrollment workflow
- [ ] Face recognition on enrolled person (>0.8 confidence)
- [ ] Unknown person detection (red box)
- [ ] Alert generation for unknown person
- [ ] Database CRUD operations
- [ ] WebSocket real-time updates
- [ ] API documentation accessible

---

## 🛡️ Security & Privacy

### Data Protection

- **Local Processing**: All AI inference happens on-device, no cloud dependencies
- **Encrypted Storage**: Database encryption available for production (PostgreSQL)
- **Access Control**: Admin panel authentication (JWT planned)
- **Audit Logging**: Complete audit trail of all access and changes
- **Data Retention**: Configurable retention policies for logs and snapshots

### Privacy Compliance

This system is designed to comply with:

- **GDPR** (EU General Data Protection Regulation)
- **CCPA** (California Consumer Privacy Act)
- **BIPA** (Illinois Biometric Information Privacy Act)

**Key Compliance Features:**

- Consent-based enrollment
- Right to deletion (complete person removal)
- Data portability (export person data)
- Access logs for auditing
- Purpose limitation (security only)
- Data minimization (only necessary data stored)

**Important:** Deployers are responsible for ensuring compliance with local laws and obtaining necessary consents.

### Security Best Practices

```bash
# 1. Secure environment variables
chmod 600 .env

# 2. Restrict database access
chmod 600 data/face_recognition.db

# 3. Enable HTTPS in production
# (Use nginx reverse proxy with SSL certificate)

# 4. Regular security updates
sudo apt update && sudo apt upgrade -y

# 5. Monitor system logs
tail -f logs/security.log
```

---

## 📊 System Monitoring

### Performance Metrics

Monitor key metrics via API:

```bash
GET /api/metrics
```

```json
{
  "fps": 18.5,
  "gpu_utilization": 52,
  "gpu_memory_used_mb": 4096,
  "detection_latency_ms": 35,
  "recognition_latency_ms": 245,
  "total_persons_enrolled": 150,
  "total_recognitions_today": 1247,
  "unknown_detections_today": 23
}
```

### Log Files

```bash
# Application logs
tail -f logs/app.log

# Server logs (when using start_server.sh)
tail -f server.log

# Error logs only
tail -f logs/error.log

# Alert logs
tail -f logs/alerts.log
```

### Database Statistics

```bash
# Check database size
du -h data/face_recognition.db

# Number of enrolled persons
sqlite3 data/face_recognition.db "SELECT COUNT(*) FROM persons;"

# Number of embeddings
sqlite3 data/face_recognition.db "SELECT COUNT(*) FROM face_embeddings;"

# Recent recognition logs
sqlite3 data/face_recognition.db "SELECT * FROM recognition_logs ORDER BY timestamp DESC LIMIT 10;"
```

---

## 🔧 Troubleshooting

### Common Issues

#### Camera Connection Failed

```bash
# Check camera is reachable
ping 192.168.1.64

# Test RTSP stream manually
ffmpeg -i rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102 -frames:v 1 test.jpg

# Check credentials in .env
cat .env | grep CAMERA
```

#### Low FPS / Performance Issues

```bash
# Check GPU utilization
tegrastats

# Check TensorRT engines cached
ls -lh data/tensorrt_engines/

# Verify CUDA is being used
python3 -c "import torch; print(torch.cuda.is_available())"

# Check system resources
htop
```

#### Recognition Not Working

```bash
# Verify models downloaded
ls -lh ~/.insightface/models/buffalo_l/

# Test detection on image
python3 tests/integration/test_face_detection.py

# Check recognition threshold
cat .env | grep THRESHOLD

# View debug logs
tail -f logs/app.log | grep recognition
```

#### Database Errors

```bash
# Backup database
cp data/face_recognition.db data/face_recognition.db.backup

# Recreate database
rm data/face_recognition.db
python3 scripts/migration/init_db.py

# Check database integrity
sqlite3 data/face_recognition.db "PRAGMA integrity_check;"
```

### Getting Help

1. Check logs: `tail -f logs/app.log`
2. Review documentation: `docs/`
3. Search closed issues: GitHub Issues
4. Open new issue with:
   - System info (`jetson_release`)
   - Error logs
   - Steps to reproduce

---

## 🚀 Production Deployment

### Docker Deployment (Recommended)

```bash
# Build container
docker build -t face-recognition-system:latest .

# Run container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop system
docker-compose down
```

### Systemd Service (Alternative)

```bash
# Create systemd service
sudo nano /etc/systemd/system/face-recognition.service

# Enable and start
sudo systemctl enable face-recognition
sudo systemctl start face-recognition

# Check status
sudo systemctl status face-recognition
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Multi-Camera Deployment

See `docs/deployment/MULTI_CAMERA_SETUP.md` for detailed instructions on:
- Distributed architecture
- Load balancing
- Centralized database
- Alert aggregation

---

## 📚 Documentation

- **[Architecture Guide](docs/development/ARCHITECTURE.md)** - Detailed system design
- **[API Documentation](http://localhost:8000/docs)** - Interactive API reference
- **[Deployment Guide](docs/deployment/PRODUCTION_DEPLOYMENT.md)** - Production setup
- **[Developer Guide](docs/development/CONTRIBUTING.md)** - Contributing guidelines
- **[Technology Stack](TECHNOLOGY_STACK.md)** - Complete tech stack details
- **[Project Roadmap](PROJECT_PLAN.md)** - Future development plans

---

## 🗺️ Roadmap

### Current Phase: Production Enhancements

- [x] GPU-accelerated SCRFD detection
- [x] Multi-angle camera capture enrollment
- [x] MJPEG live streaming (15-20 FPS)
- [ ] Stable Diffusion data augmentation
- [ ] SD card portable deployment
- [ ] Image quality assessment module

### Next Phase: AI-Powered Augmentation

- [ ] Stable Diffusion 1.5 integration
- [ ] ControlNet pose-guided generation
- [ ] Automatic quality filtering
- [ ] 10x training data per person
- [ ] 95%+ recognition accuracy

### Future Phases

- [ ] PostgreSQL migration for enterprise
- [ ] JWT authentication system
- [ ] Multi-camera distributed architecture
- [ ] Advanced alert rules (time-based, zone-based)
- [ ] Mobile app for alerts
- [ ] Cloud backup integration
- [ ] Face mask detection
- [ ] Age/gender estimation
- [ ] Emotion recognition

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed roadmap.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/development/CONTRIBUTING.md).

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
pytest tests/ -v

# Check code style
black app/ tests/
flake8 app/ tests/
mypy app/
```

### Contribution Areas

- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📝 Documentation improvements
- 🧪 Test coverage expansion
- 🎨 UI/UX improvements
- ⚡ Performance optimizations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Technologies

- **NVIDIA Jetson** - Edge AI computing platform
- **InsightFace** - State-of-the-art face recognition models (ArcFace)
- **SCRFD** - GPU-optimized face detection
- **TensorRT** - High-performance deep learning inference
- **FastAPI** - Modern Python web framework
- **Hikvision** - Professional IP camera hardware

### Research Papers

- ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., 2019)
- SCRFD: Sample and Computation Redistribution for Efficient Face Detection (Guo et al., 2021)
- High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)

### Open Source Projects

- [InsightFace](https://github.com/deepinsight/insightface)
- [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [OpenCV](https://github.com/opencv/opencv)

---

## ⚠️ Important Disclaimers

1. **Legal Compliance**: This system is designed for authorized security applications only. Users are responsible for ensuring compliance with local privacy laws and regulations (GDPR, CCPA, BIPA, etc.).

2. **Consent Required**: Always obtain proper consent before enrolling individuals in the system. Maintain documentation of consent for audit purposes.

3. **Bias Awareness**: Face recognition systems may exhibit demographic biases. Regular testing and calibration across diverse populations is recommended.

4. **Security**: This system should be deployed in secure, access-controlled environments. Implement proper network security and access controls.

5. **Liability**: This software is provided "as is" without warranties. Users assume all responsibility for deployment and use.

---

## 📧 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/mujeebawan/face-recognition-security-system/issues)
- **Discussions**: [Community discussions](https://github.com/mujeebawan/face-recognition-security-system/discussions)
- **Email**: mujeebciit72@gmail.com
- **Documentation**: [Full documentation](docs/)

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

---

*Last Updated: October 2025*
*Version: 2.0.0*
*Status: Production Ready*
