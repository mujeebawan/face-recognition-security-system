"""
System Control API - Testing, monitoring, and control endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from sqlalchemy.orm import Session
from typing import Dict, Any
import torch
import cv2
import numpy as np
import logging
import psutil
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.settings_manager import settings_manager, reload_settings
from app.core.camera import CameraHandler
from app.core.detector import FaceDetector
from app.core.recognizer import FaceRecognizer

router = APIRouter(prefix="/api/system", tags=["system-control"])
logger = logging.getLogger(__name__)


@router.get("/status")
async def get_system_status(current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive system status including:
    - Active settings (from memory)
    - Database settings
    - GPU information
    - Memory usage
    - Process information
    """
    try:
        # Get settings comparison
        comparison = settings_manager.get_active_vs_db_comparison()

        # Get GPU information
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["device_capability"] = torch.cuda.get_device_capability(0)
            memory = torch.cuda.mem_get_info(0)
            gpu_info["memory_free_gb"] = round(memory[0] / 1024**3, 2)
            gpu_info["memory_total_gb"] = round(memory[1] / 1024**3, 2)
            gpu_info["memory_used_gb"] = round((memory[1] - memory[0]) / 1024**3, 2)
            gpu_info["memory_usage_percent"] = round(((memory[1] - memory[0]) / memory[1]) * 100, 1)

        # Get system memory
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        system_info = {
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / 1024**3, 2),
            "memory_used_gb": round(memory.used / 1024**3, 2),
            "memory_available_gb": round(memory.available / 1024**3, 2),
            "memory_percent": memory.percent
        }

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "settings": comparison,
            "gpu": gpu_info,
            "system": system_info
        }

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload-settings")
async def reload_system_settings(current_user: dict = Depends(get_current_user)):
    """
    Force reload settings from database.
    This applies any changes made in the settings UI.
    """
    try:
        reload_settings()

        return {
            "success": True,
            "message": "Settings reloaded successfully from database",
            "timestamp": datetime.utcnow().isoformat(),
            "active_settings": settings_manager.get_all()
        }

    except Exception as e:
        logger.error(f"Error reloading settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/camera")
async def test_camera_connection(current_user: dict = Depends(get_current_user)):
    """
    Test camera connection using current RTSP URL from settings.
    Attempts to read a frame and returns success/failure.
    """
    try:
        rtsp_url = settings_manager.get("camera_rtsp_url")

        if not rtsp_url:
            return {
                "success": False,
                "error": "Camera RTSP URL not configured in settings"
            }

        # Try to connect
        logger.info(f"Testing camera connection to: {rtsp_url[:30]}...")
        camera = CameraHandler(use_main_stream=False, rtsp_url=rtsp_url)

        if camera.connect():
            # Try to read a frame
            frame = camera.read_frame()
            camera.disconnect()

            if frame is not None:
                return {
                    "success": True,
                    "message": "Camera connection successful",
                    "frame_shape": frame.shape,
                    "rtsp_url": rtsp_url[:30] + "..." if len(rtsp_url) > 30 else rtsp_url
                }
            else:
                return {
                    "success": False,
                    "error": "Connected but failed to read frame"
                }
        else:
            return {
                "success": False,
                "error": "Failed to connect to camera"
            }

    except Exception as e:
        logger.error(f"Camera test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/test/detection")
async def test_face_detection(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Test face detection on uploaded image using current settings.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "success": False,
                "error": "Invalid image file"
            }

        # Get current detection confidence from settings
        detection_confidence = settings_manager.get("detection_confidence", 0.5)

        # Initialize detector
        detector = FaceDetector()
        faces = detector.detect_faces(image)

        return {
            "success": True,
            "message": f"Detection completed with confidence threshold {detection_confidence}",
            "faces_detected": len(faces) if faces else 0,
            "detection_confidence": detection_confidence,
            "image_shape": image.shape
        }

    except Exception as e:
        logger.error(f"Detection test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/test/recognition")
async def test_face_recognition(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Test face recognition on uploaded image using current settings.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "success": False,
                "error": "Invalid image file"
            }

        # Get current recognition threshold from settings
        recognition_threshold = settings_manager.get("recognition_threshold", 0.35)

        # Initialize recognizer
        recognizer = FaceRecognizer()
        result = recognizer.extract_embedding(image)

        if result:
            return {
                "success": True,
                "message": f"Recognition completed with threshold {recognition_threshold}",
                "face_detected": True,
                "embedding_shape": result.embedding.shape,
                "confidence": result.confidence,
                "recognition_threshold": recognition_threshold
            }
        else:
            return {
                "success": True,
                "message": "No face detected in image",
                "face_detected": False
            }

    except Exception as e:
        logger.error(f"Recognition test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/test/gpu")
async def test_gpu_access(current_user: dict = Depends(get_current_user)):
    """
    Test GPU access and CUDA availability.
    """
    try:
        gpu_available = torch.cuda.is_available()

        if not gpu_available:
            return {
                "success": False,
                "error": "CUDA not available",
                "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None
            }

        # Try a simple GPU operation
        test_tensor = torch.randn(100, 100).cuda()
        result = test_tensor @ test_tensor.T
        del test_tensor, result
        torch.cuda.empty_cache()

        memory = torch.cuda.mem_get_info(0)

        return {
            "success": True,
            "message": "GPU test successful",
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_free_gb": round(memory[0] / 1024**3, 2),
            "memory_total_gb": round(memory[1] / 1024**3, 2)
        }

    except Exception as e:
        logger.error(f"GPU test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
