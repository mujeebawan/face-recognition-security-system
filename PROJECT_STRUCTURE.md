# Project Structure Reference
## Face Recognition Security System

**Last Updated:** November 7, 2025
**Purpose:** Master reference document for file locations and organization

---

## 📁 Root Directory

```
face-recognition-security-system/
├── app/                    # Main application code
├── data/                   # Runtime data and caches
├── docs/                   # All documentation
├── logs/                   # Application logs
├── nrtc_faceai/           # Proprietary package wrapper
├── scripts/               # Utility scripts
├── tests/                 # Test suites
├── .env                   # Environment configuration (not in git)
├── .env.example           # Environment template
├── alembic.ini            # Database migration config
├── LICENSE                # MIT License
├── README.md              # Main project documentation
├── requirements.txt       # Python dependencies
├── requirements-genai.txt # Generative AI dependencies (optional)
├── start_server.sh        # Server startup script
└── stop_server.sh         # Server shutdown script
```

---

## 🎯 Application Code (`app/`)

### Structure
```
app/
├── api/                    # FastAPI routes and endpoints
│   └── routes/
│       ├── alerts.py      # Alert management API
│       ├── analytics.py   # Analytics and reporting API
│       ├── auth.py        # Authentication API
│       ├── detection.py   # Face detection API
│       ├── recognition.py # Face recognition & enrollment API
│       ├── settings.py    # System settings API
│       └── websocket.py   # WebSocket handlers
│
├── core/                   # Core business logic
│   ├── alerts.py          # Alert creation and management
│   ├── augmentation.py    # Traditional augmentation (rotation, brightness, etc.)
│   ├── auth.py            # Authentication logic
│   ├── camera.py          # Camera/RTSP handling
│   ├── controlnet_augmentation.py   # ControlNet face augmentation
│   ├── database.py        # Database session management
│   ├── detector.py        # Face detection (SCRFD)
│   ├── generative_augmentation.py  # Stable Diffusion img2img
│   ├── liveportrait_augmentation.py # LivePortrait pose generation
│   ├── recognizer.py      # Face recognition (ArcFace)
│   └── websocket_manager.py # WebSocket manager
│
├── models/                 # Data models
│   ├── database.py        # SQLAlchemy ORM models
│   └── schemas.py         # Pydantic validation schemas
│
├── static/                 # Frontend HTML/CSS/JS
│   ├── admin.html         # Admin panel
│   ├── alerts.html        # Alerts page
│   ├── dashboard.html     # Main dashboard
│   ├── live_stream.html   # Live camera view
│   ├── login.html         # Login page
│   ├── reports.html       # Analytics & reports
│   └── settings.html      # System settings
│
├── utils/                  # Utility functions
├── config.py              # Application configuration
└── main.py                # FastAPI application entry point
```

### Key Files Purpose

| File | Purpose | When Used |
|------|---------|-----------|
| **app/core/detector.py** | SCRFD face detection (TensorRT FP16) | Live streaming, all face operations |
| **app/core/recognizer.py** | ArcFace recognition (TensorRT FP16) | Face matching, enrollment |
| **app/core/augmentation.py** | Traditional augmentation | Enrollment (optional) |
| **app/core/generative_augmentation.py** | SD img2img | Enrollment (optional) |
| **app/core/controlnet_augmentation.py** | ControlNet + IP-Adapter | Enrollment (optional) |
| **app/core/liveportrait_augmentation.py** | LivePortrait 3D pose | Enrollment (optional) |
| **app/api/routes/recognition.py** | Enrollment & recognition endpoints | All face operations |
| **app/main.py** | FastAPI app & startup | Server initialization |

---

## 💾 Data Directory (`data/`)

### Structure
```
data/
├── alert_snapshots/        # Alert snapshots (auto-generated)
├── models/                 # (Empty - models auto-downloaded to ~/.insightface)
├── person_images/          # Enrolled person images (organized by name)
│   ├── Person_Name_1/
│   │   ├── original_*.jpg
│   │   ├── traditional_aug_*.jpg
│   │   ├── liveportrait_gen_*.jpg
│   │   ├── controlnet_gen_*.jpg
│   │   └── img2img_gen_*.jpg
│   └── Person_Name_2/
│       └── ...
└── tensorrt_engines/       # TensorRT engine cache (FP16 optimized)
    ├── det_10g_*.engine   # SCRFD detection engine
    ├── w600k_r50_*.engine # ArcFace recognition engine
    └── ... (other models)
```

### Data Organization Rules

1. **Alert Snapshots** (`data/alert_snapshots/`)
   - Auto-generated when alerts are triggered
   - Format: `alert_{id}_{timestamp}.jpg`
   - Kept indefinitely (manual cleanup needed)

2. **Person Images** (`data/person_images/`)
   - **NEW Structure** (used now): One folder per person
     - Folder name: `Person_Name` (sanitized)
     - Contains: original + all augmented images
   - Organized by enrollment name
   - Preserved across system restarts

3. **TensorRT Engines** (`data/tensorrt_engines/`)
   - Auto-generated on first model load
   - Cached for fast subsequent startups
   - GPU-specific (SM87 for Jetson AGX Orin)
   - Total size: ~166MB
   - Safe to delete (will regenerate)

---

## 📚 Documentation (`docs/`)

### Structure
```
docs/
├── api/                    # API documentation
│   └── (planned)
│
├── architecture/           # System architecture docs
│   ├── SYSTEM_CONFIGURATION.md  # Current system configuration
│   ├── SYSTEM_OVERVIEW.md       # High-level architecture
│   └── TECHNOLOGY_STACK.md      # Tech stack details
│
├── deployment/             # Deployment guides
│   ├── JETPACK_UPGRADE.md       # JetPack upgrade guide
│   └── JETSON_SETUP.md          # Jetson setup instructions
│
├── development/            # Development docs
│   ├── CHANGELOG.md             # Version history
│   └── ROADMAP.md               # Future plans
│
├── demos/                  # Demo HTML files
│   ├── multi_agent_viewer.html
│   └── presentation.html
│
├── getting-started/        # Getting started guides
│   └── QUICK_START.md
│
├── proprietary/            # Proprietary/commercial docs
│   └── NRTC_FACEAI.md
│
└── README.md               # Documentation index
```

### Documentation Organization

| Category | Location | Purpose |
|----------|----------|---------|
| **System Architecture** | `docs/architecture/` | How the system works |
| **Deployment** | `docs/deployment/` | How to set up and deploy |
| **Development** | `docs/development/` | Changelog, roadmap |
| **Getting Started** | `docs/getting-started/` | Quick start guides |
| **API Docs** | `docs/api/` | API reference (planned) |
| **Demos** | `docs/demos/` | Demo/presentation files |

---

## 🧪 Tests (`tests/`)

### Structure
```
tests/
├── integration/            # Integration tests
│   ├── test_alerts_api.py
│   ├── test_camera_direct.py
│   ├── test_camera.py
│   ├── test_face_detection.py
│   ├── test_gpu_performance.py
│   ├── test_live_stream.py
│   └── test_recognizer.py
│
└── utilities/              # Test utility scripts
    └── test_all_models.py  # Comprehensive model analysis
```

### Test Categories

| Test Type | Location | Purpose |
|-----------|----------|---------|
| **Integration** | `tests/integration/` | API, camera, detection tests |
| **Utilities** | `tests/utilities/` | One-off test scripts, analysis tools |
| **Performance** | `tests/integration/test_gpu_performance.py` | GPU/TensorRT benchmarks |

---

## 🔧 Scripts (`scripts/`)

### Structure
```
scripts/
├── migration/              # Database migrations
│   ├── add_user_table.py
│   └── init_db.py
│
├── setup/                  # Setup scripts
│   └── create_default_admin.py
│
└── utilities/              # Utility scripts
    ├── capture_live_frame.py
    ├── capture_test_frame.py
    └── debug_recognition.py
```

### Script Categories

| Category | Location | Purpose |
|----------|----------|---------|
| **Migration** | `scripts/migration/` | Database schema changes |
| **Setup** | `scripts/setup/` | Initial setup and configuration |
| **Utilities** | `scripts/utilities/` | Helper scripts for debugging/testing |

---

## 📦 NRTC FaceAI Package (`nrtc_faceai/`)

### Structure
```
nrtc_faceai/
├── nrtc_faceai/            # Package source
│   ├── augmentation/       # Augmentation modules
│   ├── core/               # Core detection/recognition
│   ├── license/            # License validation
│   └── utils/              # Utilities
│
├── generate_license.py     # License generation script
├── README.md               # Package documentation
└── setup.py                # Package installation
```

### Purpose
- Proprietary package wrapper for commercial deployment
- Includes license validation system
- Wraps core face detection/recognition functionality
- **Note:** Not currently used in main app (uses InsightFace directly)

---

## 📊 Runtime Files

### Database
- **Location:** `face_recognition.db` (root directory)
- **Type:** SQLite3
- **Contains:** Users, persons, embeddings, alerts, logs
- **Migrations:** Managed by Alembic (`alembic/versions/`)

### Logs
- **Location:** `logs/` (created at runtime)
- **Files:** `server.log`, access logs, error logs
- **Rotation:** Not yet implemented (planned)

### Cache
- **TensorRT Engines:** `data/tensorrt_engines/`
- **InsightFace Models:** `~/.insightface/models/buffalo_l/`
- **Hugging Face Models:** `~/.cache/huggingface/` (for GenAI)

---

## 🚀 Startup Scripts

### Server Management

| Script | Purpose | Usage |
|--------|---------|-------|
| **start_server.sh** | Start FastAPI server | `./start_server.sh` |
| **stop_server.sh** | Stop running server | `./stop_server.sh` |

### Manual Startup
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 🔍 Finding Things Quickly

### "Where is...?"

| What | Location |
|------|----------|
| **Face detection code** | `app/core/detector.py` |
| **Face recognition code** | `app/core/recognizer.py` |
| **Enrollment logic** | `app/api/routes/recognition.py` (line 86+) |
| **Live stream** | `app/api/routes/recognition.py` (line 1564+) |
| **Authentication** | `app/core/auth.py` + `app/api/routes/auth.py` |
| **Database models** | `app/models/database.py` |
| **API schemas** | `app/models/schemas.py` |
| **Frontend pages** | `app/static/*.html` |
| **Configuration** | `app/config.py` + `.env` |
| **Augmentation** | `app/core/*_augmentation.py` |
| **Camera handling** | `app/core/camera.py` |
| **Alert system** | `app/core/alerts.py` |
| **System architecture** | `docs/architecture/SYSTEM_OVERVIEW.md` |
| **Current status** | `CURRENT_STATUS.md` (root) |
| **Deployment guide** | `docs/deployment/JETSON_SETUP.md` |
| **API documentation** | `http://localhost:8000/docs` (when running) |

---

## 📝 File Naming Conventions

### Python Files
- **Snake case:** `face_detector.py`, `control_net_augmentation.py`
- **Modules:** `core/`, `api/`, `models/`
- **Tests:** `test_*.py`

### HTML Files
- **Lowercase:** `dashboard.html`, `live_stream.html`
- **Underscores:** For multi-word names

### Documentation
- **UPPERCASE:** `README.md`, `CHANGELOG.md`, `PROJECT_STRUCTURE.md`
- **Title Case:** For section names

### Data Files
- **Person folders:** `Person_Name` (no special chars)
- **Images:** `original_*.jpg`, `liveportrait_gen_1.jpg`
- **Alerts:** `alert_{id}_{timestamp}.jpg`

---

## 🗑️ What NOT to Commit

Covered by `.gitignore`:
- `.env` (environment secrets)
- `*.db` (databases)
- `data/person_images/` (private data)
- `data/alert_snapshots/` (private data)
- `data/tensorrt_engines/` (GPU-specific cache)
- `logs/` (runtime logs)
- `__pycache__/`, `*.pyc` (Python cache)
- `venv/` (virtual environment)
- `.claude/` (Claude Code config)

---

## 🔄 Migration from Old Structure

### If You Have Old Files

**Old person images** (flat structure in `data/images/`):
- Location: `data/images/{cnic}_*.jpg`
- Migration: Run person enrollment again to use new structure
- Compatibility: Old structure still supported in read operations

**Old documentation** (archived):
- Location: Previously in `archive/` (now removed)
- Current docs: `docs/` with proper organization

---

## 📖 Related Documents

- **Main README:** [README.md](README.md)
- **Current Status:** [CURRENT_STATUS.md](CURRENT_STATUS.md)
- **System Architecture:** [docs/architecture/SYSTEM_OVERVIEW.md](docs/architecture/SYSTEM_OVERVIEW.md)
- **Quick Start:** [docs/getting-started/QUICK_START.md](docs/getting-started/QUICK_START.md)
- **Deployment:** [docs/deployment/JETSON_SETUP.md](docs/deployment/JETSON_SETUP.md)

---

**Maintained By:** Development Team
**For Updates:** Create an issue or PR on GitHub
