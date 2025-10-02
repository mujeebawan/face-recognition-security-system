# Face Recognition System - Project Plan

## Project Overview
A face recognition system designed for security purposes using computer vision, capable of identifying individuals from a single reference image. The system will run on Jetson AGX Orin with a Logitech HD1080 USB webcam.

## System Architecture

### Hardware
- **Device**: NVIDIA Jetson AGX Orin
- **Camera**: Hikvision IP Camera DS-2CD7A47EWD-XZS
  - IP Address: 192.168.1.64
  - Protocol: RTSP
  - Resolution: 4MP (2688×1520)
- **OS**: Linux (Tegra)

### Software Stack
- **Backend Framework**: FastAPI
- **Computer Vision**: OpenCV (with RTSP support), dlib/MTCNN/face-recognition
- **Deep Learning**: PyTorch/TensorFlow (for face embeddings)
- **Data Augmentation**: Diffusion models (for single image enhancement)
- **Database**: SQLite (with SQLAlchemy ORM for easy PostgreSQL migration)
- **Camera Stream**: RTSP (via OpenCV/FFmpeg)

## Project Phases

### Phase 1: Environment Setup & Basic Infrastructure
**Goal**: Set up development environment and basic FastAPI server

#### Steps:
1. Create project structure
2. Set up virtual environment
3. Install core dependencies (FastAPI, OpenCV, uvicorn, SQLAlchemy)
4. Create configuration management (camera credentials, RTSP URL)
5. Create basic FastAPI server with health check endpoint
6. Test Hikvision IP camera RTSP connection
7. Set up SQLite database with SQLAlchemy (migration-ready structure)

**Camera Connection Details**:
- RTSP URL format: `rtsp://admin:Mujeeb@321@192.168.1.64:554/Streaming/Channels/101`
- Fallback: `rtsp://admin:Mujeeb@321@192.168.1.64:554/h264/ch1/main/av_stream`

**Deliverables**:
- Working FastAPI server
- RTSP stream from Hikvision camera accessible
- SQLite database initialized with SQLAlchemy
- Environment configuration (.env file)
- Project structure established

---

### Phase 2: Face Detection Pipeline
**Goal**: Implement robust face detection from camera feed

#### Steps:
1. Integrate face detection library (MTCNN/RetinaFace/MediaPipe)
2. Create API endpoint for live face detection
3. Implement face alignment and preprocessing
4. Add bounding box visualization
5. Optimize for Jetson AGX Orin (GPU acceleration)

**Deliverables**:
- Real-time face detection from webcam
- API endpoint: `/detect-faces`
- Preprocessed face crops

---

### Phase 3: Face Recognition Core
**Goal**: Extract face embeddings and implement matching logic

#### Steps:
1. Integrate face recognition model (FaceNet/ArcFace/CosFace)
2. Extract face embeddings (512-D vectors)
3. Implement similarity matching (cosine similarity/Euclidean distance)
4. Create enrollment endpoint (register face from single image)
5. Create recognition endpoint (match against database)
6. Set up embedding database/storage

**Deliverables**:
- Face embedding extraction
- API endpoints: `/enroll`, `/recognize`
- Face database with embeddings

---

### Phase 4: Single Image Enhancement (Diffusion Models)
**Goal**: Augment single reference images to improve recognition accuracy

#### Steps:
1. Research and select appropriate diffusion model (Stable Diffusion/ControlNet)
2. Generate synthetic variations of reference image
   - Different angles (pose variation)
   - Different lighting conditions
   - Different expressions
3. Extract embeddings from augmented images
4. Store multiple embeddings per person
5. Improve matching algorithm with multi-embedding comparison

**Deliverables**:
- Image augmentation pipeline
- Enhanced enrollment with synthetic data
- Improved recognition accuracy

---

### Phase 5: Database Integration & NADRA-like Structure
**Goal**: Create production-ready database for citizen records

#### Steps:
1. Design database schema (person info + face embeddings)
2. Implement CRUD operations for person records
3. Create bulk enrollment endpoint
4. Add search and filtering capabilities
5. Implement secure data storage

**Database Schema** (SQLAlchemy Models):
```python
Person:
- id (Integer, Primary Key, Auto-increment)
- uuid (String, Unique)  # For future PostgreSQL UUID compatibility
- name (String)
- cnic (String, Unique)  # National ID number
- reference_image_path (String)
- created_at (DateTime)
- updated_at (DateTime)

FaceEmbedding:
- id (Integer, Primary Key)
- person_id (Foreign Key -> Person.id)
- embedding (Blob/PickleType)  # 512-D vector
- source (String)  # 'original', 'augmented', 'diffusion'
- created_at (DateTime)
```

**Migration Strategy**:
- Using SQLAlchemy ORM abstracts database differences
- Simple Alembic migration to PostgreSQL when needed
- Embeddings stored as JSON/Blob (compatible with both DBs)

**Deliverables**:
- SQLite database with SQLAlchemy models
- API endpoints: `/persons/create`, `/persons/search`, `/persons/bulk-enroll`
- Database initialization script
- Alembic setup for future migrations

---

### Phase 6: Real-time Recognition System
**Goal**: Continuous face recognition from live camera feed

#### Steps:
1. Implement video stream processing
2. Frame-by-frame face detection and recognition
3. Add confidence thresholds and filtering
4. Implement tracking (to avoid duplicate detections)
5. Create WebSocket endpoint for live updates
6. Add logging and alerts for recognized persons

**Deliverables**:
- Live recognition from camera feed
- WebSocket endpoint: `/ws/live-recognition`
- Alert system for matches

---

### Phase 7: Optimization & Performance
**Goal**: Optimize for Jetson AGX Orin hardware

#### Steps:
1. Convert models to TensorRT for GPU acceleration
2. Implement frame skipping and smart processing
3. Optimize memory usage
4. Add model quantization (INT8/FP16)
5. Benchmark and profile performance
6. Implement multi-threading/async processing

**Performance Targets**:
- Face detection: >20 FPS
- Recognition latency: <100ms
- GPU utilization: >80%

---

### Phase 8: Security & Production Features
**Goal**: Add security and production-ready features

#### Steps:
1. Add authentication/authorization (JWT tokens)
2. Implement API rate limiting
3. Add audit logging
4. Create admin dashboard endpoints
5. Add data encryption for stored embeddings
6. Implement backup and recovery

---

### Phase 9: UI/Frontend (Optional)
**Goal**: Create web interface for system management

#### Steps:
1. Create React/Vue.js frontend
2. Live camera feed display
3. Person enrollment interface
4. Recognition results dashboard
5. Database management UI

---

## Project Structure

```
face_recognition_system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py         # Database models
│   │   └── schemas.py          # Pydantic schemas
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── health.py       # Health check
│   │   │   ├── camera.py       # Camera endpoints
│   │   │   ├── detection.py   # Face detection
│   │   │   ├── recognition.py # Face recognition
│   │   │   └── persons.py     # Person management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── camera.py           # Camera handling
│   │   ├── detector.py         # Face detection logic
│   │   ├── recognizer.py       # Face recognition logic
│   │   ├── augmentation.py    # Image augmentation/diffusion
│   │   └── database.py         # Database operations
│   └── utils/
│       ├── __init__.py
│       ├── image_processing.py
│       └── logger.py
├── data/
│   ├── embeddings/             # Stored face embeddings
│   ├── images/                 # Reference images
│   └── models/                 # Pre-trained models
├── tests/
│   └── ...
├── requirements.txt
├── README.md
├── PROJECT_PLAN.md
├── .env                        # Camera credentials & config (DO NOT COMMIT)
├── .env.example                # Template for environment variables
├── alembic.ini                 # Database migration config
├── alembic/                    # Migration files
└── face_recognition.db         # SQLite database file (DO NOT COMMIT)
```

## Technology Decisions

### Face Detection
**Options**:
1. **MTCNN** - Multi-task CNN, accurate but slower
2. **RetinaFace** - State-of-the-art, good balance
3. **MediaPipe** - Fast, optimized for edge devices ✅ (Recommended for Jetson)

**Choice**: Start with MediaPipe for speed, can switch to RetinaFace if needed

### Face Recognition Model
**Options**:
1. **FaceNet** - Classic, well-tested
2. **ArcFace** - State-of-the-art accuracy ✅ (Recommended)
3. **InsightFace** - Fast, optimized implementation

**Choice**: InsightFace (includes ArcFace, optimized for deployment)

### Diffusion Model for Augmentation
**Options**:
1. **Stable Diffusion + ControlNet** - Precise control over generation
2. **Face-specific diffusion models** - Optimized for faces
3. **Traditional augmentation first** - Rotation, lighting, etc. ✅ (Start here)

**Choice**: Phase approach - traditional augmentation first, then explore diffusion models

## Current Phase: Phase 1
**Next Steps**:
1. Create project structure (folders and files)
2. Create .env and .env.example files with camera credentials
3. Set up requirements.txt with initial dependencies
4. Create basic FastAPI server with health check
5. Test Hikvision camera RTSP stream connection
6. Initialize SQLite database with SQLAlchemy

## Camera Configuration Notes

### Hikvision DS-2CD7A47EWD-XZS
- **Model**: High-end 4MP fisheye camera with excellent low-light performance
- **Features**:
  - 4MP resolution (2688×1520)
  - Wide angle fisheye lens
  - Built-in analytics capabilities
  - H.264/H.265 compression

### RTSP Stream URLs to Try (in order):
1. Main stream: `rtsp://admin:Mujeeb@321@192.168.1.64:554/Streaming/Channels/101`
2. Sub stream: `rtsp://admin:Mujeeb@321@192.168.1.64:554/Streaming/Channels/102`
3. Alternative: `rtsp://admin:Mujeeb@321@192.168.1.64:554/h264/ch1/main/av_stream`
4. ONVIF: `rtsp://admin:Mujeeb@321@192.168.1.64:554/onvif1`

**Note**: For better performance, we might use the sub-stream (lower resolution) for real-time detection and main stream for enrollment/high-quality captures.

## Security Considerations
- Encrypt stored face embeddings
- Implement access control for APIs
- Secure database connections
- Audit logging for all recognition attempts
- GDPR/privacy compliance for face data storage
- Rate limiting to prevent abuse

## Performance Metrics to Track
- Face detection accuracy
- Face recognition accuracy (precision/recall)
- False positive rate (critical for security)
- Processing latency
- FPS (frames per second)
- GPU/CPU utilization
- Memory usage

## Notes
- Single image recognition is challenging - augmentation is crucial
- Jetson AGX Orin has good GPU - leverage TensorRT
- Consider liveness detection to prevent spoofing
- May need threshold tuning for security vs. usability balance
