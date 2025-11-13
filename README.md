# Face Recognition Security System

**Real-Time GPU-Accelerated Face Recognition Platform for NVIDIA Jetson AGX Orin**

A production-ready security system for real-time face detection, recognition, and monitoring using professional IP cameras and edge AI processing.

---

## 🚀 Quick Start

```bash
# Start the system
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access the application
http://192.168.0.117:8000
```

**📚 Full Documentation**: See **[docs/](docs/)** directory for complete guides

**📊 Current Status**: See **[CURRENT_STATUS.md](CURRENT_STATUS.md)** for latest features and progress

---

## ✨ Current Features

###  Core Capabilities
- ✅ **Real-Time Face Detection** - GPU-accelerated SCRFD with TensorRT FP16 optimization (~15-20 FPS)
- ✅ **Face Recognition** - InsightFace ArcFace with 512-D embeddings
- ✅ **Multi-Face Processing** - Detect and track up to 10 faces simultaneously
- ✅ **RTSP Camera Integration** - Hikvision IP camera support (main/sub-stream)
- ✅ **Person Enrollment** - Single/multiple image enrollment with augmentation
- ✅ **Alert System** - Real-time alerts for known/unknown persons with cooldown
- ✅ **Live Streaming** - MJPEG video stream with real-time recognition overlay

### 📊 Web Interface
- ✅ **Dashboard** - Real-time statistics, live stream preview, recent alerts
- ✅ **Admin Panel** - Person management (add, edit, delete enrolled persons)
- ✅ **Alert Management** - Advanced filtering, acknowledgment, bulk operations, CSV export
- ✅ **Reports & Analytics** - Interactive charts, time-series analysis, person statistics
- ✅ **Live Stream Viewer** - Full-screen video monitoring with recognition data
- ✅ **Camera Zoom Control** - Remote motorized zoom with speed control and keyboard shortcuts
- ✅ **System Control Panel** - Live monitoring, settings verification, and test endpoints
- ✅ **JWT Authentication** - Secure login with role-based access control

### 🔌 API Features
- ✅ **RESTful API** - Complete FastAPI endpoints with OpenAPI documentation
- ✅ **WebSocket Support** - Real-time alert notifications
- ✅ **Analytics API** - Summary stats, time-series data, distribution analysis
- ✅ **PTZ Control API** - Camera zoom control via ISAPI (Hikvision)
- ✅ **System Status API** - Real-time GPU, memory, and settings monitoring
- ✅ **Dynamic Settings** - Database-backed configuration with live reload
- ✅ **Image Authentication** - Secure snapshot serving with JWT tokens
- ✅ **CORS Support** - Configurable cross-origin resource sharing

### 🗄️ Data Management
- ✅ **SQLite Database** - Person records, embeddings, alerts, recognition logs
- ✅ **Dynamic Configuration** - Runtime settings loaded from database with caching
- ✅ **Alert Snapshots** - Automatic snapshot capture and storage
- ✅ **Recognition Audit Log** - Complete history of all recognition attempts
- ✅ **Person Photos** - Secure storage and authenticated serving

---

## 🏗️ Architecture

### Hardware
- **Platform**: NVIDIA Jetson AGX Orin (64GB)
- **OS**: JetPack 6.1 (L4T 36.4.0)
- **Camera**: Hikvision DS-2CD7A47EWD-XZS (4MP Fisheye)

### Software Stack
- **Backend**: FastAPI 0.104.1
- **Detection**: SCRFD (InsightFace) + TensorRT FP16
- **Recognition**: ArcFace (InsightFace buffalo_l)
- **Database**: SQLite 3.x (PostgreSQL-ready via SQLAlchemy)
- **Computer Vision**: OpenCV 4.x (CUDA-enabled)
- **Deep Learning**: ONNX Runtime 1.19.0 (TensorRT Execution Provider)
- **Frontend**: Vanilla JavaScript, Chart.js for visualizations

### AI Models
- **Face Detection**: SCRFD (scrfd_10g_bnkps)
  - Input: 640x640 RGB
  - Output: Bounding boxes + 5 facial keypoints
  - Optimization: TensorRT FP16, ~30-50ms per frame

- **Face Recognition**: ArcFace (buffalo_l)
  - Input: 112x112 aligned face
  - Output: 512-D embedding vector
  - Matching: Cosine similarity with 0.35 threshold

---

## 📈 Performance

- **Live Stream**: 15-20 FPS with GPU acceleration
- **Detection Time**: ~27ms per frame (SCRFD + TensorRT FP16)
- **Recognition Time**: ~31ms per face (ArcFace + TensorRT FP16)
- **Combined Pipeline**: ~58ms total latency
- **GPU Utilization**: 40-60% during active detection
- **Multi-Face**: Up to 10 faces processed simultaneously
- **Recognition Accuracy**: ~90-95% with single image, ~95-98% with multiple images
- **All Models**: 100% GPU-accelerated with TensorRT FP16 optimization

---

## 🎨 Screenshots & Access

### Web Interface URLs
- **Dashboard**: http://192.168.0.117:8000/dashboard
- **Admin Panel**: http://192.168.0.117:8000/admin
- **Alert Management**: http://192.168.0.117:8000/alerts
- **Reports & Analytics**: http://192.168.0.117:8000/reports
- **Live Stream**: http://192.168.0.117:8000/live
- **API Documentation**: http://192.168.0.117:8000/docs

### Default Credentials
- **Username**: `admin`
- **Password**: `admin123`

---

## 📚 Documentation

### Getting Started
- **[Quick Start Guide](docs/getting-started/QUICK_START.md)** - Get up and running in 5 minutes
- **[Project Structure](PROJECT_STRUCTURE.md)** - Master reference for file locations
- **[System Requirements](docs/)** - Hardware and software prerequisites

### Architecture & Technical
- **[System Overview](docs/architecture/SYSTEM_OVERVIEW.md)** - Architecture and design
- **[Technology Stack](docs/architecture/TECHNOLOGY_STACK.md)** - Complete tech stack details
- **[System Configuration](docs/architecture/SYSTEM_CONFIGURATION.md)** - Current configuration

### Deployment
- **[Jetson Setup](docs/deployment/JETSON_SETUP.md)** - Complete Jetson AGX Orin setup
- **[JetPack Upgrade](docs/deployment/JETPACK_UPGRADE.md)** - Upgrading to JetPack 6.1

### Development
- **[Project Roadmap](docs/development/ROADMAP.md)** - Future plans and milestones
- **[Changelog](docs/development/CHANGELOG.md)** - Version history

---

## 🔧 System Requirements

### Hardware (Minimum)
- NVIDIA Jetson AGX Orin (32GB or 64GB)
- IP Camera with RTSP support
- Network connectivity (Gigabit Ethernet recommended)

### Software
- JetPack 6.1 (L4T 36.4.0)
- Python 3.10+
- CUDA 12.2+
- ONNX Runtime 1.19.0 (GPU)
- InsightFace models (buffalo_l)

---

## 📊 Project Status

**Current Version**: 0.7.0 (70% Complete)
**Last Updated**: November 5, 2025
**Status**: ✅ Fully Functional (Production-Ready Core Features)

**Recent Milestones**:
- ✅ Milestone 1: Core Face Recognition System
- ✅ Milestone 2: GPU Acceleration & Cleanup
- ✅ Milestone 3: Alert Management & GUI Enhancements
- ✅ Milestone 4: Reports & Analytics Dashboard

**Next Up**:
- 🚧 System Settings Page (Web UI for configuration)
- 🚧 SD Card Portability System
- 🚧 Enhanced Enrollment Workflow

For detailed status, see **[CURRENT_STATUS.md](CURRENT_STATUS.md)**

---

## 🛠️ Installation

See **[docs/deployment/JETSON_SETUP.md](docs/deployment/JETSON_SETUP.md)** for complete installation guide.

### Quick Setup (if dependencies already installed)
```bash
# Clone repository
git clone <repository-url>
cd face-recognition-security-system

# Install Python dependencies
pip3 install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your camera IP and settings

# Run database migrations
python3 -m app.db.init_db

# Start the server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📄 License

Proprietary - All Rights Reserved
Unauthorized use, reproduction, or distribution prohibited.

---

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Current Status**: [CURRENT_STATUS.md](CURRENT_STATUS.md)
- **API Docs**: http://192.168.0.117:8000/docs

---

**Built on NVIDIA Jetson AGX Orin**
