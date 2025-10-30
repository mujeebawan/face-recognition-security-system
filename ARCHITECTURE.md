# System Architecture

## Overview

Face Recognition Security System using SCRFD (GPU detection) + ArcFace (recognition) on NVIDIA Jetson AGX Orin.

**Key Principle**: Simplified, single-pipeline architecture optimized for production deployment.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│           Hikvision IP Camera (4MP Fisheye)                  │
│              RTSP Stream (192.168.1.64:554)                  │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│                   Camera Handler (Singleton)                  │
│           Sub-stream: 640x480 for performance                │
│              OpenCV VideoCapture (RTSP)                      │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│                    Frame Processing Pipeline                  │
│  • Frame skip (every 2nd frame)                              │
│  • Recognition throttle (every 5th processed frame)          │
│  • Caching for smooth visualization                          │
└──────────────────────────────────────────────────────────────┘
                           ↓
         ┌─────────────────┴─────────────────┐
         ↓                                    ↓
┌──────────────────────┐         ┌──────────────────────┐
│   SCRFD Detector     │         │  ArcFace Recognizer  │
│   (GPU + TensorRT)   │         │   (InsightFace)      │
│                      │         │                      │
│ • FP16 optimization  │         │ • buffalo_l model    │
│ • ~30-50ms/frame     │         │ • 512-D embeddings   │
│ • Bounding boxes     │         │ • ~200-300ms/face    │
└──────────────────────┘         └──────────────────────┘
         │                                    │
         └─────────────────┬─────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│              Recognition Logic & Alert System                 │
│  • Match embeddings against database                         │
│  • Threshold: 0.35 (configurable)                            │
│  • Trigger alerts for known/unknown persons                  │
│  • Log all recognition events                                │
└──────────────────────────────────────────────────────────────┘
                           ↓
         ┌─────────────────┴─────────────────┐
         ↓                                    ↓
┌──────────────────────┐         ┌──────────────────────┐
│  SQLite Database     │         │   WebSocket + API    │
│                      │         │                      │
│ • Persons            │         │ • FastAPI routes     │
│ • Face embeddings    │         │ • MJPEG stream       │
│ • Recognition logs   │         │ • Real-time alerts   │
│ • Alert snapshots    │         │ • Admin panel        │
└──────────────────────┘         └──────────────────────┘
```

---

## Component Details

### 1. Camera Handler (`app/core/camera.py`)
- **Singleton pattern** to prevent multiple RTSP connections
- Uses **sub-stream** (640x480) for better performance
- Auto-reconnection on connection loss
- Thread-safe frame access

### 2. SCRFD Face Detector (`app/core/detector.py`)
- **GPU-accelerated** with TensorRT FP16 optimization
- InsightFace SCRFD model (scrfd_10g_bnkps)
- Input size: 640x640
- Returns: Bounding boxes + 5 keypoints per face
- **Singleton** to avoid model reload overhead

### 3. ArcFace Face Recognizer (`app/core/recognizer.py`)
- InsightFace buffalo_l model
- 512-dimensional face embeddings
- Cosine similarity for matching
- Batch embedding extraction support
- **Singleton** for efficiency

### 4. Video Stream Pipeline (`app/api/routes/recognition.py`)
- **Frame Skip Strategy**: Process every 2nd frame (50% skip rate)
- **Recognition Throttle**: Run recognition every 5th processed frame
- **Caching**: Cache last detection/recognition for skipped frames
- **Background Thread**: Recognition runs in separate thread (non-blocking)
- **Output**: MJPEG stream at ~15-20 FPS

### 5. Alert System (`app/core/alerts.py`)
- Configurable thresholds for known/unknown persons
- Snapshot capture on alerts
- WebSocket real-time notifications via `websocket_manager.py`
- Throttling to prevent alert spam (10-second cooldown)

### 6. Traditional Augmentation (`app/core/augmentation.py`)
- FaceAugmentation class for basic CV transformations
- Rotation, brightness, contrast, flip, blur, noise
- Generates 10+ variations per image
- Used until Phase 5 SD augmentation is implemented

### 7. Database Schema (`app/models/database.py`)
```sql
Person:
  - id (PK), uuid (unique), name, cnic (unique)
  - reference_image_path, created_at, updated_at

FaceEmbedding:
  - id (PK), person_id (FK), embedding (BLOB - 512-D)
  - source (original/augmented/diffusion), confidence, created_at

RecognitionLog:
  - id (PK), person_id (FK, nullable), timestamp
  - confidence, matched (0/1), image_path, camera_source

Alert:
  - id (PK), timestamp, event_type, person_id (FK, nullable)
  - person_name, confidence, num_faces, snapshot_path, acknowledged
```

---

## Data Flow

### Enrollment Flow
```
1. User uploads image(s) via /api/enroll or /api/enroll/multiple
2. FaceDetector detects face in image
3. FaceRecognizer extracts 512-D embedding
4. Create Person record in database
5. Store FaceEmbedding(s) linked to Person
6. (Optional) Apply augmentation → generate additional embeddings
7. Return success with person_id
```

### Recognition Flow
```
1. Camera frame received (640x480)
2. Check frame skip counter → skip if needed
3. SCRFD detects faces → bounding boxes
4. Check recognition throttle → use cached if needed
5. For each face:
   a. Extract 512-D embedding (ArcFace)
   b. Match against all database embeddings (cosine similarity)
   c. If similarity > threshold (0.35) → known person
   d. If similarity <= threshold → unknown person
6. Draw bounding boxes (green=known, red=unknown)
7. Log recognition event to database
8. Trigger alerts if configured
9. Encode frame as JPEG
10. Yield to MJPEG stream
```

---

## Future Enhancements (Planned)

### Phase 5: AI-Powered Data Augmentation (Stable Diffusion)
```
┌──────────────────────────────────────────────────────────────┐
│         Stable Diffusion + ControlNet Pipeline               │
│                                                              │
│  Input: Single enrollment image                             │
│     ↓                                                        │
│  ControlNet (pose-guided)                                   │
│     ↓                                                        │
│  Generate 5-10 different angles                             │
│     ↓                                                        │
│  Quality filtering                                          │
│     ↓                                                        │
│  Extract embeddings from generated images                   │
│     ↓                                                        │
│  Store as augmented embeddings                              │
└──────────────────────────────────────────────────────────────┘
```

### Phase 6+: SD Card Portability
- Automatic migration script
- Portable database on SD card
- Easy deployment across multiple devices
- Configurable storage paths

---

## Performance Optimizations

1. **Singleton Pattern**: Detector, Recognizer, Camera (avoid reload overhead)
2. **Frame Skipping**: Process 50% of frames (2x throughput)
3. **Recognition Throttling**: Run recognition every 5th frame (5x faster)
4. **Background Threading**: Non-blocking recognition processing
5. **TensorRT FP16**: GPU optimization for SCRFD
6. **Result Caching**: Reuse last result for skipped frames

---

## Security & Privacy

- **Local Processing**: No cloud dependencies
- **Encrypted Storage**: Face embeddings stored securely
- **Audit Logging**: All recognition events logged
- **Access Control**: Admin panel restricted
- **RTSP Encryption**: Camera credentials in .env

---

## Deployment Architecture

```
Jetson AGX Orin
├── FastAPI Server (Port 8000)
├── SQLite Database (data/face_recognition.db)
├── TensorRT Cache (data/tensorrt_engines/)
├── Alert Snapshots (data/alert_snapshots/)
└── Reference Images (data/images/)

Optional: SD Card
├── Database (portable)
├── Images (portable)
└── Models (portable)
```

---

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI | Async API framework |
| Detection | SCRFD + TensorRT | GPU face detection |
| Recognition | InsightFace ArcFace | Face embeddings |
| Database | SQLite | Data storage |
| Streaming | MJPEG | Browser-compatible video |
| Augmentation (Future) | Stable Diffusion + ControlNet | Synthetic data generation |

---

**Last Updated**: October 29, 2025
**Architecture Version**: 2.0 (Post Multi-Agent Removal)
