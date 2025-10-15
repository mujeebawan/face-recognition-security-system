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

### Phase 1: Environment Setup & Basic Infrastructure ✅ COMPLETE
**Goal**: Set up development environment and basic FastAPI server
**Status**: ✅ Completed October 2, 2025

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

### Phase 2: Face Detection Pipeline ✅ COMPLETE
**Goal**: Implement robust face detection from camera feed
**Status**: ✅ Completed October 2, 2025

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

### Phase 3: Face Recognition Core ✅ COMPLETE
**Goal**: Extract face embeddings and implement matching logic
**Status**: ✅ Completed October 2, 2025

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

### Phase 4A: Traditional Image Augmentation & Multi-Image Enrollment ✅ COMPLETE
**Goal**: Augment reference images and support multiple images per person
**Status**: ✅ Completed October 2, 2025 (Traditional augmentation done, diffusion models pending)

### Phase 4B: Advanced Augmentation (Diffusion Models) ⏳ PENDING
**Goal**: Use diffusion models for synthetic face generation
**Status**: ⏳ Not started (Deprioritized - focusing on multi-agent system first)

---

### Phase 7: Multi-Agent Recognition System & Advanced Augmentation 🚧 IN PROGRESS
**Goal**: Build production-ready multi-agent face recognition with diffusion-based augmentation
**Status**: 🚧 Phase 7.1 Complete (SCRFD), Phase 7.2 Starting (AdaFace)
**Current**: Session 12 - AdaFace Testing & ONNX Conversion
**Timeline**: 4-6 weeks total

---

#### **Milestone 2: SCRFD GPU Detection** ✅ COMPLETE (October 15, 2025)
**Baseline Achieved:**
- Detection: SCRFD (GPU, TensorRT FP16) - 97.6% accuracy, 2-5ms
- Recognition: ArcFace buffalo_l (GPU, TensorRT FP16) - 96.8% accuracy, 30-40ms
- Performance: Stable, GPU-accelerated pipeline (50-60% GPU usage)

---

#### **Sub-Phase 7.1: SCRFD Detection Upgrade** ✅ COMPLETE
**Goal**: Replace MediaPipe with SCRFD for GPU-accelerated detection
**Status**: ✅ Deployed October 15, 2025 (Milestone 2)

**What Changed:**
```
Before:  MediaPipe (CPU, TFLite)   + ArcFace (GPU, TensorRT) = 35-50ms
After:   SCRFD (GPU, TensorRT FP16) + ArcFace (GPU, TensorRT) = 32-45ms
Result:  Both detection + recognition on GPU, 2x faster detection, +27.6% accuracy
```

**Deliverables:**
- ✅ SCRFD det_10g integrated with TensorRT FP16
- ✅ Singleton pattern for detector (prevents recreation lag)
- ✅ TensorRT engine caching (5 engines, 169MB, sm87 architecture)
- ✅ Benchmark results documented
- ✅ Git milestone-2-scrfd tag created

---

#### **Sub-Phase 7.2: Recognition Model Testing** 🚧 IN PROGRESS (Week 1-2)
**Goal**: Test each recognition model individually, convert to ONNX, benchmark
**Status**: 🚧 Starting Session 12 (October 15, 2025)

**Step 1: AdaFace (PyTorch → ONNX → TensorRT)** ⏳ CURRENT
- Install PyTorch (Jetson-compatible)
- Download AdaFace pre-trained weights
- Test with sample images in `model_experiments/recognition_tests/`
- Convert to ONNX format
- Load with TensorRT provider
- Benchmark vs ArcFace (accuracy, speed, memory)
- **Expected**: 97.5% accuracy, 30-40ms with TensorRT FP16
- **Effort**: 2-3 days

**Step 2: CosFace Testing** ⏳ PENDING
- Test CosFace from InsightFace (already ONNX)
- Benchmark vs ArcFace + AdaFace
- **Expected**: 96.5% accuracy, 30-40ms
- **Effort**: 1 day

**Step 3: Side-by-Side Comparison** ⏳ PENDING
- Compare all 3 models: ArcFace, AdaFace, CosFace
- Decision: Keep models with accuracy ≥96% and speed ≤50ms
- Document results in DEVELOPMENT_LOG.md
- **Effort**: 1 day

**Deliverables:**
- [ ] AdaFace ONNX model with TensorRT FP16
- [ ] CosFace tested and benchmarked
- [ ] Comparison report (accuracy, speed, memory)
- [ ] Decision on which models to keep for multi-agent system

**Testing Structure:**
```
model_experiments/
├── recognition_tests/
│   ├── test_adaface.py           # AdaFace PyTorch testing
│   ├── convert_adaface_onnx.py   # PyTorch → ONNX conversion
│   ├── test_cosface.py           # CosFace testing
│   ├── benchmark_all.py          # Side-by-side comparison
│   └── results/
│       ├── adaface_results.json
│       ├── cosface_results.json
│       └── comparison_report.md
```

---

#### **Sub-Phase 7.3: Multi-Agent Parallel System** ⏳ PENDING (Week 3)
**Goal**: Run 3+ recognition models in parallel with consensus voting
**Status**: ⏳ After Phase 7.2 complete

**Architecture:**
```python
class MultiAgentRecognizer:
    def __init__(self):
        # All models ONNX + TensorRT for consistency
        self.models = {
            'arcface': load_onnx_model('arcface_buffalo_l.onnx'),
            'adaface': load_onnx_model('adaface_r50.onnx'),
            'cosface': load_onnx_model('cosface_r100.onnx')
        }
        # Separate CUDA streams for parallel execution
        self.streams = [create_cuda_stream() for _ in range(3)]

    def recognize_parallel(self, face_image):
        # Run all 3 models in parallel on GPU
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(model.run, face_image)
                      for model in self.models.values()]
            results = [f.result() for f in futures]

        # Consensus voting
        return self.vote(results)
```

**Implementation:**
```
app/core/
├── multi_agent_recognizer.py     # NEW: Multi-agent engine
├── consensus_voting.py           # NEW: Voting/ensemble logic
└── cuda_stream_manager.py        # NEW: CUDA stream management
```

**Performance Target:**
- 3 models in parallel: ~50-60ms (not 3×35ms = 105ms!)
- Accuracy: 98-99%+ with consensus voting
- GPU usage: 80-90%

**Deliverables:**
- [ ] Parallel inference engine (3+ models)
- [ ] Consensus voting system
- [ ] CUDA stream management
- [ ] Benchmark: parallel vs sequential
- [ ] Integration with live stream

**Effort**: 3-4 days

---

#### **Sub-Phase 7.4: Multi-Image Enrollment** ⏳ PENDING (Week 4)
**Goal**: Support uploading multiple images per person during enrollment
**Status**: ⏳ After Phase 7.3 complete

**Database Schema Update:**
```sql
-- Current schema
persons (id, name, cnic, reference_image_path)
embeddings (person_id, embedding_blob)

-- NEW schema
persons (id, name, cnic, description, created_at, updated_at)

person_images (
    id, person_id, image_path,
    image_type,        -- 'original', 'synthetic_30deg', 'synthetic_45deg', etc.
    quality_score,     -- 0.0-1.0 (brightness, sharpness, size, pose)
    created_at
)

person_embeddings (
    id, person_id, image_id,
    model_name,        -- 'arcface', 'adaface', 'cosface'
    embedding_blob,    -- 512-D vector
    created_at
)
```

**Features:**
- Upload 1-10 images during enrollment
- Automatic quality assessment for each image
- Store embeddings for each model × each image
- Example: 3 models × 10 images = 30 embeddings per person

**API Endpoints:**
```python
POST /persons/enroll-multi
    - Upload multiple images
    - Extract embeddings with all models
    - Store in database

POST /persons/{id}/add-images
    - Add more images to existing person
```

**Deliverables:**
- [ ] Updated database schema
- [ ] Multi-image enrollment API
- [ ] Quality assessment per image
- [ ] UI for multi-image upload

**Effort**: 2-3 days

---

#### **Sub-Phase 7.5: Image Quality Assessment** ⏳ PENDING (Week 4)
**Goal**: Assess image quality and trigger enhancement if needed
**Status**: ⏳ After Phase 7.4 complete

**Quality Metrics:**
```python
def assess_quality(image):
    return {
        'brightness': check_brightness(image),      # Too dark/bright?
        'sharpness': check_sharpness(image),        # Blurry?
        'face_size': check_face_size(image),        # Too small?
        'occlusion': check_occlusion(image),        # Face visible?
        'pose_angle': estimate_pose(image),         # Frontal vs profile
        'overall': weighted_average(...)            # 0.0-1.0 score
    }
```

**Decision Logic:**
```
if quality_score >= 0.8:
    → Use image as-is (good quality)
elif quality_score >= 0.5:
    → Apply enhancement (denoise, super-resolution, sharpen)
else:
    → Need synthetic augmentation (diffusion model)
```

**Implementation:**
```
app/core/
├── image_quality.py              # NEW: Quality assessment
└── image_enhancement.py          # NEW: Denoise, super-resolution
```

**Deliverables:**
- [ ] Quality assessment module
- [ ] Image enhancement (denoise, sharpen, super-res)
- [ ] Automatic quality scoring
- [ ] Integration with enrollment pipeline

**Effort**: 2 days

---

#### **Sub-Phase 7.6: Diffusion Model Augmentation** ⏳ PENDING (Week 5-6)
**Goal**: Generate synthetic face angles from single NADRA image
**Status**: ⏳ After Phase 7.5 complete

**Use Case:**
```
Input:  Single frontal image from NADRA
Output: 8-10 synthetic angles (±30°, ±45°, ±90°, variations)
When:   ONLY during enrollment (offline), NOT real-time
```

**Diffusion Models to Test:**
1. **Stable Diffusion + ControlNet** (Best for pose control)
2. **DreamBooth** (Person-specific fine-tuning)
3. **FaceAdapter** (Face-specific augmentation)

**Implementation:**
```
app/core/
├── diffusion_augmentor.py        # NEW: Diffusion model wrapper
└── pose_controller.py            # NEW: ControlNet pose control

model_experiments/
├── diffusion_tests/
│   ├── test_controlnet.py
│   ├── test_dreambooth.py
│   ├── generate_angles.py
│   └── results/
```

**Augmentation Pipeline:**
```python
def augment_enrollment_image(original_image):
    # Generate 8-10 angles
    angles = [0, 30, -30, 45, -45, 90, -90]
    synthetic_images = []

    for angle in angles:
        synthetic = diffusion_model.generate(
            original_image,
            pose_angle=angle,
            preserve_identity=True,
            num_inference_steps=50
        )
        synthetic_images.append((synthetic, f"synthetic_{angle}deg"))

    return synthetic_images
```

**Performance:**
- Speed: 1-2 seconds per angle (acceptable for offline enrollment)
- Quality: High fidelity, preserve identity
- Hardware: Can run on Jetson (may take 30-60 sec total for all angles)

**Deliverables:**
- [ ] Diffusion model integration
- [ ] Pose control (generate specific angles)
- [ ] Identity preservation
- [ ] Batch generation for enrollment
- [ ] Quality validation of synthetic images

**Effort**: 7-10 days (diffusion models are complex)

---

#### **Sub-Phase 7.7: Complete Integration** ⏳ PENDING (Week 7)
**Goal**: End-to-end pipeline from enrollment to recognition
**Status**: ⏳ After Phase 7.6 complete

**Complete System Architecture:**
```
┌───────────────────────────────────────────────────────────────┐
│                    ENROLLMENT PHASE (Offline)                  │
└───────────────────────────────────────────────────────────────┘

User uploads 1-3 images of wanted person
    ↓
[Image Quality Assessment]
    ├── Good quality (≥0.8) → Use as-is
    ├── Medium quality (0.5-0.8) → Enhancement (denoise, sharpen)
    └── Low quality (<0.5) → Diffusion augmentation
    ↓
[Diffusion Model] (if needed)
    → Generate 8-10 angles: 0°, ±30°, ±45°, ±90°
    → Generate lighting variations
    ↓
[Multi-Agent Embedding Extraction]
    ├── ArcFace (ONNX + TensorRT)
    ├── AdaFace (ONNX + TensorRT)
    └── CosFace (ONNX + TensorRT)
    ↓
Store in Database:
    - 3 models × 10 images = 30 embeddings per person

┌───────────────────────────────────────────────────────────────┐
│                   RECOGNITION PHASE (Real-time)                │
└───────────────────────────────────────────────────────────────┘

RTSP Camera Stream (30 fps)
    ↓
[SCRFD Detection] (GPU, TensorRT, 2-5ms)
    ↓
[Multi-Agent Recognition] (Parallel, 50-60ms)
    ├── ArcFace (30-40ms)
    ├── AdaFace (30-40ms)
    └── CosFace (30-40ms)
    ↓
[Consensus Voting]
    - Compare against 30 embeddings per person
    - Each model votes: Match or No Match
    - Confidence scoring
    ↓
[Decision]
    ├── 3/3 models agree + high confidence → ALERT (High Priority)
    ├── 2/3 models agree → ALERT (Medium Priority)
    └── 1/3 or 0/3 → No match

Total Latency: ~55-70ms (real-time capable at 15-20 fps)
```

**Deliverables:**
- [ ] End-to-end enrollment pipeline
- [ ] End-to-end recognition pipeline
- [ ] Integration testing
- [ ] Performance benchmarking
- [ ] Documentation update
- [ ] Session 20+ in DEVELOPMENT_LOG.md

**Effort**: 3-4 days

---

### **Phase 7 Summary**

**Timeline**: 4-6 weeks total

| Sub-Phase | Task | Status | Effort |
|-----------|------|--------|--------|
| **7.1** | SCRFD Detection | ✅ COMPLETE | - |
| **7.2** | Recognition Models (AdaFace, CosFace) | 🚧 IN PROGRESS | 4-5 days |
| **7.3** | Multi-Agent Parallel System | ⏳ PENDING | 3-4 days |
| **7.4** | Multi-Image Enrollment | ⏳ PENDING | 2-3 days |
| **7.5** | Image Quality Assessment | ⏳ PENDING | 2 days |
| **7.6** | Diffusion Augmentation | ⏳ PENDING | 7-10 days |
| **7.7** | Complete Integration | ⏳ PENDING | 3-4 days |

**Expected Final Performance:**
- Detection: SCRFD (97.6% accuracy, 2-5ms)
- Recognition: Multi-agent (98-99%+ accuracy, 50-60ms)
- Total Latency: 55-70ms per face
- GPU Utilization: 80-90%
- False Alarm Rate: <1% (critical for LEA)

**Key Principles:**
1. ✅ Test each model **individually** before combining
2. ✅ Convert everything to **ONNX + TensorRT** for consistency
3. ✅ Keep production system **isolated** in `model_experiments/`
4. ✅ **Incremental progress**: Small verified steps, not big leaps
5. ✅ **Git milestones**: Checkpoint after each major phase

---

### Phase 5: Database Integration & NADRA-like Structure ✅ COMPLETE
**Goal**: Create production-ready database for citizen records
**Status**: ✅ Completed October 2, 2025 (SQLite implementation, PostgreSQL-ready)

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

### Phase 6: Real-time Recognition System ⚠️ PARTIALLY COMPLETE
**Goal**: Continuous face recognition from live camera feed
**Status**: ⚠️ Live stream working, enhancements pending

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

### Phase 7: Optimization & Performance ⚠️ PARTIALLY COMPLETE
**Goal**: Optimize for Jetson AGX Orin hardware
**Status**: ⚠️ CPU optimizations done, GPU blocked by GLIBC incompatibility

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

### Phase 8: Security & Production Features ⏳ PENDING (RECOMMENDED NEXT)
**Goal**: Add security and production-ready features
**Status**: ⏳ Not started - HIGH PRIORITY for production deployment

#### Steps:
1. Add authentication/authorization (JWT tokens)
2. Implement API rate limiting
3. Add audit logging
4. Create admin dashboard endpoints
5. Add data encryption for stored embeddings
6. Implement backup and recovery

---

### Phase 9: UI/Frontend ⚠️ PARTIALLY COMPLETE
**Goal**: Create web interface for system management
**Status**: ⚠️ Basic live stream viewer complete, admin dashboard pending

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

## Current Status (Updated: October 3, 2025)

### ✅ Completed Phases:
- **Phase 1**: Environment Setup & Infrastructure - COMPLETE
- **Phase 2**: Face Detection Pipeline - COMPLETE
- **Phase 3**: Face Recognition Core - COMPLETE
- **Phase 4A**: Multi-Image Enrollment & Live Streaming - COMPLETE (Traditional Augmentation)

### ⚠️ Partially Complete:
- **Phase 5**: GPU Acceleration & Optimization - CPU optimizations done, GPU blocked by GLIBC

### ⏳ Pending Phases:
- **Phase 4B**: Advanced Augmentation (Diffusion Models)
- **Phase 6**: Real-time Recognition Enhancements
- **Phase 7**: Production Optimization
- **Phase 8**: Security & Production Features (RECOMMENDED NEXT)
- **Phase 9**: UI/Frontend Enhancement

### 📊 Project Statistics:
- **Total API Endpoints**: 15+
- **Enrolled Persons**: 2 (Mujeeb, Safyan)
- **Database Size**: ~80 KB (SQLite)
- **Live Stream Performance**: ~10-15 FPS (CPU-optimized)
- **Lines of Code**: ~3,000+
- **Git Commits**: 9

### 🎯 Recommended Next Steps:
1. **Phase 8 - Security Features** (High Priority for Production)
   - JWT authentication
   - API rate limiting
   - Data encryption
2. **Phase 6 - Real-time Enhancements** (Improve User Experience)
   - Alert system for unknown persons
   - Confidence tuning
3. **Phase 9 - UI Enhancement** (Better Management)
   - Admin dashboard
   - Person management UI

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
