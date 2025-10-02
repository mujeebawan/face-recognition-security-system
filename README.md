# Face Recognition Security System

AI-powered face recognition system for security applications using Hikvision IP camera on NVIDIA Jetson AGX Orin.

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

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed development roadmap:

1. ✅ **Phase 1**: Environment Setup & Infrastructure
2. ⏳ **Phase 2**: Face Detection Pipeline
3. ⏳ **Phase 3**: Face Recognition Core
4. ⏳ **Phase 4**: Single Image Enhancement (Diffusion Models)
5. ⏳ **Phase 5**: Database Integration
6. ⏳ **Phase 6**: Real-time Recognition System
7. ⏳ **Phase 7**: Optimization for Jetson AGX Orin
8. ⏳ **Phase 8**: Security & Production Features
9. ⏳ **Phase 9**: UI/Frontend (Optional)

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Face Detection
```bash
POST /api/detect-faces
```

### Face Enrollment
```bash
POST /api/enroll
Body: {
  "name": "John Doe",
  "cnic": "12345-1234567-1",
  "image": "base64_encoded_image"
}
```

### Face Recognition
```bash
POST /api/recognize
Body: {
  "image": "base64_encoded_image"
}
```

### Person Management
```bash
GET    /api/persons          # List all persons
GET    /api/persons/{id}     # Get person by ID
POST   /api/persons          # Create person
PUT    /api/persons/{id}     # Update person
DELETE /api/persons/{id}     # Delete person
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

## 📈 Performance Targets

- **Face Detection**: >20 FPS
- **Recognition Latency**: <100ms
- **GPU Utilization**: >80%
- **Accuracy**: >95% with single image augmentation

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
