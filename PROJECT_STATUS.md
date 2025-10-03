# Face Recognition System - Project Status & Configuration

**Last Updated**: October 3, 2025
**Maintained By**: Mujeeb
**Current Phase**: Phase 6.2 Complete

---

## 🎯 Project Overview

**Purpose**: Security face recognition system using Hikvision IP camera on NVIDIA Jetson AGX Orin
**Primary Use Case**: Identify and alert on KNOWN persons (like employees, family members)
**Current Configuration**: Alerts trigger when KNOWN persons are detected

---

## ⚙️ Current System Configuration

### Alert System Behavior (IMPORTANT!)

**🔔 ALERTS ARE CONFIGURED FOR KNOWN PERSONS**

- ✅ **Alert on KNOWN persons**: `True` (Mujeeb, Safyan)
- ❌ **Alert on UNKNOWN persons**: `False` (disabled)
- ⏱️ **Cooldown**: 60 seconds between alerts
- 📸 **Snapshot**: Enabled (saves image with each alert)

**Why?** The system is configured to alert when recognized persons (Mujeeb, Safyan) appear, not when unknown people appear.

**Location**: `app/core/alerts.py` line 39-40
```python
"alert_on_unknown": False,  # DISABLED
"alert_on_known": True,      # ENABLED
```

### How to Change Alert Behavior

**To alert on UNKNOWN persons instead:**
1. Edit `app/core/alerts.py` line 39-40:
   ```python
   "alert_on_unknown": True,   # Enable unknown alerts
   "alert_on_known": False,    # Disable known alerts
   ```
2. Restart the server
3. Test with someone NOT enrolled

**To alert on BOTH:**
```python
"alert_on_unknown": True,
"alert_on_known": True,
```

---

## 📊 Completed Phases

### ✅ Phase 1: Infrastructure (Oct 2, 2025)
- FastAPI backend
- SQLite database with SQLAlchemy
- Hikvision RTSP camera integration
- Configuration management

### ✅ Phase 2: Face Detection (Oct 2, 2025)
- MediaPipe integration
- Real-time detection from camera
- Bounding box visualization

### ✅ Phase 3: Face Recognition (Oct 2, 2025)
- InsightFace (ArcFace) for embeddings
- Person enrollment API
- Recognition logging
- **Enrolled Persons**: Mujeeb, Safyan

### ✅ Phase 4A: Multi-Image Enrollment (Oct 2, 2025)
- Traditional augmentation (rotation, brightness, contrast)
- Multi-image enrollment (1-10 images)
- Camera-based enrollment
- Live MJPEG streaming

### ⚠️ Phase 5: GPU Optimization (Oct 2-3, 2025)
- ✅ CPU optimizations working
- ❌ GPU acceleration blocked (GLIBC incompatibility)
- Current: CPU-only, ~10-15 FPS

### ✅ Phase 6.1: Alert System (Oct 3, 2025)
- Alert database models
- AlertManager with cooldown
- Snapshot capture
- 7 REST API endpoints
- Webhook support

### ✅ Phase 6.2: WebSocket Real-time (Oct 3, 2025)
- WebSocket connection manager
- Real-time alert broadcasting
- Auto-reconnection
- Live dashboard UI

---

## 🗄️ Database Status

**Database**: `face_recognition.db` (SQLite)
**Size**: ~140 KB

### Tables:
1. **persons** - 2 enrolled (Mujeeb, Safyan)
2. **face_embeddings** - Multiple embeddings per person
3. **recognition_logs** - All recognition events
4. **alerts** - Alert history (5+ alerts from testing)
5. **system_configuration** - Runtime config

### Enrolled Persons:
| ID | Name | CNIC | Embeddings |
|----|------|------|------------|
| 1  | Mujeeb | [CNIC] | Multiple |
| 2  | Safyan | [CNIC] | Multiple |

---

## 🌐 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Swagger documentation

### Web Interfaces
- `GET /live` - Live stream viewer (basic)
- `GET /dashboard` - Real-time dashboard with WebSocket alerts

### Face Detection
- `POST /api/detect-faces` - Detect faces in image
- `GET /api/camera/snapshot` - Capture frame
- `GET /api/camera/detect` - Quick detection

### Face Enrollment
- `POST /api/enroll` - Single image enrollment
- `POST /api/enroll/multiple` - Multi-image enrollment
- `POST /api/enroll/camera` - Camera-based enrollment

### Face Recognition
- `POST /api/recognize` - Recognize from image
- `GET /api/recognize/camera` - Recognize from camera

### Live Streaming
- `GET /api/stream/live` - MJPEG video stream with recognition

### Alert Management
- `GET /api/alerts/active` - Get unacknowledged alerts
- `GET /api/alerts/recent?hours=24` - Get recent alerts
- `POST /api/alerts/acknowledge` - Acknowledge alert
- `GET /api/alerts/statistics` - Alert statistics
- `GET /api/alerts/config` - Get alert config
- `PUT /api/alerts/config` - Update alert config
- `DELETE /api/alerts/{id}` - Delete alert

### WebSocket
- `WS /ws/alerts` - Real-time alert stream
- `GET /ws/stats` - WebSocket statistics

---

## 🧪 Testing Workflow

### To Test Known Person Alerts:

1. **Start Server**:
   ```bash
   cd /home/mujeeb/Downloads/face_recognition_system
   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Open Dashboard**:
   ```
   http://localhost:8000/dashboard
   ```

3. **Trigger Alert**:
   - **Mujeeb or Safyan** stand in front of camera
   - Wait for recognition (~1-2 seconds)
   - Alert appears instantly on dashboard

4. **Expected Behavior**:
   - ✅ Alert shows "✓ KNOWN PERSON"
   - ✅ Name displayed (Mujeeb or Safyan)
   - ✅ Green color theme
   - ✅ Snapshot saved
   - ✅ No page refresh needed

### To Test Unknown Person Alerts:

1. **Change Configuration** (see "How to Change Alert Behavior" above)
2. **Restart Server**
3. Have someone NOT enrolled appear
4. Alert shows "⚠ UNKNOWN PERSON" in red

---

## 📁 Key Files & Locations

### Core Logic
- `app/core/detector.py` - MediaPipe face detection
- `app/core/recognizer.py` - InsightFace recognition
- `app/core/alerts.py` - **Alert configuration HERE** (line 39-40)
- `app/core/websocket_manager.py` - WebSocket connections
- `app/core/camera.py` - RTSP camera handler

### API Routes
- `app/api/routes/detection.py` - Detection endpoints
- `app/api/routes/recognition.py` - Recognition & streaming
- `app/api/routes/alerts.py` - Alert management
- `app/api/routes/websocket.py` - WebSocket endpoint

### Database
- `app/models/database.py` - SQLAlchemy models
- `app/models/schemas.py` - Pydantic schemas
- `face_recognition.db` - SQLite database
- `alembic/` - Database migrations

### Web Interface
- `app/static/live_stream.html` - Basic live stream
- `app/static/dashboard.html` - Real-time dashboard with WebSocket

### Data Storage
- `data/images/` - Reference images
- `data/alert_snapshots/` - Alert snapshots (45-55KB each)
- `data/models/` - AI models

### Documentation
- `README.md` - Project overview
- `PROJECT_PLAN.md` - Development roadmap
- `DEVELOPMENT_LOG.md` - Detailed session logs
- `PROJECT_STATUS.md` - **THIS FILE** (current status)
- `PHASE_6_PROGRESS.md` - Phase 6 implementation
- `PHASE_6.2_TESTING.md` - WebSocket testing guide

---

## ⚡ Performance Metrics

### Current Performance (CPU-only):
- **Live Stream FPS**: 10-15 FPS
- **Face Detection**: 5-10ms (MediaPipe)
- **Face Recognition**: 300-400ms (InsightFace CPU)
- **Recognition Frequency**: Every 20th frame
- **Frame Skip**: 50% (process every 2nd frame)
- **Alert Overhead**: <50ms
- **WebSocket Latency**: <100ms

### Recognition Threshold:
- **Similarity Threshold**: 0.55 (55%)
- Scores above 0.55 = MATCH
- Scores below 0.55 = NO MATCH

---

## 🔧 Configuration Files

### Environment Variables (.env.example):
```bash
# Camera
CAMERA_IP=192.168.1.64
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_password

# Recognition
FACE_RECOGNITION_THRESHOLD=0.6

# Alerts (CURRENT CONFIG)
ALERT_ON_UNKNOWN=false    # DISABLED
ALERT_ON_KNOWN=true       # ENABLED
ALERT_COOLDOWN_SECONDS=60
ALERT_SAVE_SNAPSHOT=true
```

---

## 🐛 Known Issues

### Issue 1: GPU Acceleration Blocked
- **Status**: ❌ Unresolved
- **Error**: GLIBC incompatibility with onnxruntime-gpu
- **Workaround**: CPU-only processing (acceptable performance)
- **Impact**: Cannot use CUDA/TensorRT

### Issue 2: Single Camera Stream
- **Status**: Known limitation
- **Impact**: Only one client can access RTSP stream directly
- **Workaround**: Use `/api/stream/live` endpoint (multiple clients OK)

---

## 🚀 Next Phases

### Phase 6.3: Confidence Tuning (Pending)
- Web interface for threshold adjustment
- Per-person confidence settings
- A/B testing

### Phase 7: Production Optimization (Pending)
- PostgreSQL migration
- Advanced caching
- TensorRT (when GPU available)

### Phase 8: Security Features (Pending)
- JWT authentication
- API rate limiting
- Data encryption
- RBAC

### Phase 9: UI Enhancement (Pending)
- Full admin dashboard
- Person management UI
- Recognition history viewer

---

## 💡 Quick Reference

### Start Server:
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### View Logs:
```bash
tail -f logs/app.log  # If logging to file
```

### Check Database:
```bash
sqlite3 face_recognition.db ".tables"
```

### Test API:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/alerts/active | python3 -m json.tool
```

### Git Status:
```bash
git log --oneline -5
git status
```

---

## 📝 Important Notes for Future Sessions

1. **Alert Configuration**: Always check `app/core/alerts.py` line 39-40 to see if alerts are on KNOWN or UNKNOWN persons

2. **Testing**:
   - KNOWN person alerts: Mujeeb or Safyan must appear
   - UNKNOWN person alerts: Need someone not enrolled

3. **WebSocket**: Dashboard at `/dashboard` shows real-time alerts

4. **Performance**: Initial face detection has 6-7 second model load (one-time)

5. **Database**: SQLite for development, PostgreSQL-ready for production

6. **GPU**: Currently blocked, CPU-only mode working fine

---

**For Next Session**: Check this file first to understand current configuration and what has been done!
