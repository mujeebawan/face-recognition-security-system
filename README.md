# Face Recognition Security System

**Real-Time GPU-Accelerated Face Recognition Platform for NVIDIA Jetson AGX Orin**

A production-ready security system for real-time face detection, recognition, and monitoring using professional IP cameras and edge AI processing.

---

## Quick Start

```bash
# Start the system
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access the application
http://192.168.0.117:8000
```

**Full Documentation**: See [docs/](docs/) directory for complete guides

**Current Status**: See [CURRENT_STATUS.md](CURRENT_STATUS.md) for latest features and progress

---

## Features

### Core Capabilities
- Real-Time Face Detection - GPU-accelerated SCRFD with TensorRT FP16 optimization (25-30 FPS smooth streaming)
- Ultra-Fast Face Recognition - FAISS GPU for <1ms similarity search (100-200x faster than sequential search)
- IoU-Based Face Tracking - Accurate spatial matching across frames, eliminates cache collisions
- Multi-Face Processing - Detect and track up to 10 faces simultaneously with instant recognition
- Scalable Database - Handle 1000+ faces without performance degradation
- RTSP Camera Integration - Hikvision IP camera support (main/sub-stream, 2K/1080p/720p)
- Person Enrollment - Single/multiple image enrollment with augmentation
- Alert System - Real-time alerts for known/unknown persons with cooldown
- 4 Quality Modes - Dynamic stream quality selection (Smooth/Balanced/High Quality/Maximum)
- Live Streaming - Hardware-accelerated MJPEG stream with GStreamer (nvv4l2decoder), adaptive resolution

### Web Interface
- Dashboard - Real-time statistics, live stream preview, enhanced alert popups with side-by-side image comparison
- Admin Panel - Person management with real-time enrollment progress tracking (9-stage progress bar)
- Alert Management - Advanced filtering, improved button layouts, acknowledgment system, bulk operations, CSV export
- Reports & Analytics - Interactive charts, time-series analysis, person statistics
- Live Stream Viewer - Hardware-accelerated streaming with frame skip control, main/sub-stream switching
- Camera Zoom Control - Remote motorized zoom with speed control and keyboard shortcuts
- System Control Panel - Live monitoring, settings verification, and test endpoints
- JWT Authentication - Secure login with role-based access control
- Responsive Design - Optimized layouts at 100% zoom, no scrolling required

### API Features
- RESTful API - Complete FastAPI endpoints with OpenAPI documentation
- WebSocket Support - Real-time alert notifications
- Analytics API - Summary stats, time-series data, distribution analysis
- PTZ Control API - Camera zoom control via ISAPI (Hikvision)
- System Status API - Real-time GPU, memory, and settings monitoring
- Dynamic Settings - Database-backed configuration with live reload
- Image Authentication - Secure snapshot serving with JWT tokens
- CORS Support - Configurable cross-origin resource sharing

### Data Management
- SQLite Database - Person records, embeddings, alerts, recognition logs
- Dynamic Configuration - Runtime settings loaded from database with caching
- Alert Snapshots - Automatic snapshot capture and storage
- Recognition Audit Log - Complete history of all recognition attempts
- Person Photos - Secure storage and authenticated serving

---

## Architecture

### Hardware
- **Platform**: NVIDIA Jetson AGX Orin (64GB)
- **OS**: JetPack 6.1 (L4T 36.4.0)
- **Camera**: Hikvision DS-2CD7A47EWD-XZS (4MP Fisheye)

### Software Stack
- **Backend**: FastAPI 0.104.1
- **Detection**: SCRFD (InsightFace) + TensorRT FP16
- **Recognition**: ArcFace (InsightFace buffalo_l) + FAISS GPU
- **Similarity Search**: Facebook AI Similarity Search (FAISS) with GPU acceleration
- **Database**: SQLite 3.x (PostgreSQL-ready via SQLAlchemy)
- **Video Processing**: GStreamer 1.0 with NVIDIA hardware acceleration (nvv4l2decoder)
- **Computer Vision**: OpenCV 4.x (CUDA-enabled, FFMPEG fallback)
- **Deep Learning**: ONNX Runtime 1.19.0 (TensorRT Execution Provider)
- **Frontend**: Vanilla JavaScript, Chart.js for visualizations

### AI Models (Pretrained)

**Core Models** (Always Active):
- **Face Detection**: SCRFD det_10g (InsightFace buffalo_l pack)
  - Pretrained on WIDER FACE dataset
  - Input: 640x640 RGB
  - Output: Bounding boxes + 5 facial keypoints
  - Optimization: TensorRT FP16, ~27-50ms per frame
  - Benchmark: Industry-leading accuracy

- **Face Recognition**: ArcFace W600K-R50 (InsightFace buffalo_l pack)
  - Pretrained on MS-Celeb-1M / MS1MV3 dataset
  - Input: 112x112 aligned face
  - Output: 512-D embedding vector
  - Matching: Cosine similarity with 0.35 threshold
  - Benchmark Accuracy: >99.8% on LFW, >98% on CFP-FP

**Optional Augmentation Models** (For Enrollment):
- **Stable Diffusion 1.5**: Face angle generation (~1.5-3s per image)
- **ControlNet Depth**: Pose-guided generation (~3.5-5.5s per image)
- **LivePortrait**: 3D-aware pose changes (~0.5-1s per image)

> **Note**: All models use pretrained weights. No custom training required - works out of the box!

---

## Performance

### Streaming Performance
- **Live Stream FPS**: 25-30 FPS (Smooth mode) | 23-27 FPS (Balanced) | 20-25 FPS (High Quality) | 15-20 FPS (Maximum)
- **Stream Quality Modes**: 4 selectable modes (720p/1080p/2K) with adaptive JPEG encoding
- **Recording Resolution**: Always full 2K (2560x1440) regardless of stream quality

### Processing Benchmarks
- **Detection Time**: ~30ms per frame (SCRFD + TensorRT FP16)
- **Recognition Time**: <1ms per face (FAISS GPU - 100-200x faster!)
- **JPEG Encoding**: 8-10ms (720p) | 15-20ms (1080p) | 25-30ms (2K)
- **Frame Read**: 5-15ms (GStreamer hardware decoding)
- **Combined Pipeline**: ~45ms total latency (Smooth mode)

### Scalability
- **Database Capacity**: 1000+ faces with no performance degradation
- **Multi-Face**: Up to 10 faces processed simultaneously
- **Recognition Frequency**: Every 5th frame (configurable)
- **GPU Utilization**: 40-60% during active detection

### Accuracy
- **Recognition Accuracy**: ~90-95% with single image, ~95-98% with multiple images
- **Face Detection**: Industry-leading SCRFD with TensorRT FP16
- **Face Tracking**: IoU-based spatial matching (0.5 threshold)
- **All Models**: 100% GPU-accelerated with TensorRT FP16 optimization

---

## Access Points

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

## Documentation

### Getting Started
- [Quick Start Guide](docs/getting-started/QUICK_START.md) - Get up and running in 5 minutes
- [Project Structure](PROJECT_STRUCTURE.md) - Master reference for file locations
- [System Requirements](docs/) - Hardware and software prerequisites

### Architecture & Technical
- [System Overview](docs/architecture/SYSTEM_OVERVIEW.md) - Architecture and design
- [Technology Stack](docs/architecture/TECHNOLOGY_STACK.md) - Complete tech stack details
- [System Configuration](docs/architecture/SYSTEM_CONFIGURATION.md) - Current configuration

### Deployment
- [Jetson Setup](docs/deployment/JETSON_SETUP.md) - Complete Jetson AGX Orin setup
- [JetPack Upgrade](docs/deployment/JETPACK_UPGRADE.md) - Upgrading to JetPack 6.1

### Development
- [Project Roadmap](docs/development/ROADMAP.md) - Future plans and milestones
- [Changelog](docs/development/CHANGELOG.md) - Version history

---

## System Requirements

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

## Project Status

**Current Version**: 1.0.0 (Production Ready)
**Last Updated**: November 18, 2025
**Status**: Production-Ready Face Recognition System

**Completed Milestones**:
- Milestone 1: Core Face Recognition System
- Milestone 2: GPU Acceleration & TensorRT Optimization
- Milestone 3: Alert Management & GUI Enhancements
- Milestone 4: Reports & Analytics Dashboard
- Milestone 5: Watchlist System & Production Hardening
- Milestone 6: System Settings & Configuration UI
- Milestone 7: FAISS GPU Integration & Performance Optimization (November 2025)
  - FAISS GPU for ultra-fast similarity search (<1ms vs 100-200ms)
  - IoU-based face tracking for accurate spatial matching
  - Dynamic quality selector (4 modes: Smooth/Balanced/Quality/Maximum)
  - Optimized streaming pipeline (25-30 FPS smooth playback)
  - Scalable to 1000+ faces without performance loss

**Optional Features (Available)**:
- AI Data Augmentation (ControlNet, Stable Diffusion, LivePortrait)
- Multi-Image Enrollment
- Adaptive Recognition Thresholds

For detailed status, see [CURRENT_STATUS.md](CURRENT_STATUS.md)

---

## Installation

See [docs/deployment/JETSON_SETUP.md](docs/deployment/JETSON_SETUP.md) for complete installation guide.

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

## License

Proprietary - All Rights Reserved
Unauthorized use, reproduction, or distribution prohibited.

---

## Support

- **Documentation**: [docs/](docs/)
- **Current Status**: [CURRENT_STATUS.md](CURRENT_STATUS.md)
- **API Docs**: http://192.168.0.117:8000/docs

---

**Built on NVIDIA Jetson AGX Orin**
