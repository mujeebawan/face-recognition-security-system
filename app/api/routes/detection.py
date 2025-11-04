"""
Face detection API endpoints.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import time
from datetime import datetime
import logging

from app.core.detector import FaceDetector
from app.core.camera import CameraHandler
from app.models.schemas import (
    DetectionResponse,
    FaceDetectionResult,
    BoundingBox,
    Landmark,
    CameraFrameResponse
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["detection"])

# Initialize face detector (singleton)
face_detector = FaceDetector(min_detection_confidence=settings.face_detection_confidence)


@router.post("/detect-faces", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image.

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        Detection results with bounding boxes and landmarks
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        height, width = image.shape[:2]

        # Detect faces
        start_time = time.time()
        detections = face_detector.detect_faces(image)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Convert to response format
        detection_results = []
        for det in detections:
            bbox = BoundingBox(
                x=det.bbox[0],
                y=det.bbox[1],
                width=det.bbox[2],
                height=det.bbox[3]
            )

            landmarks = None
            if det.landmarks:
                landmarks = [Landmark(x=lm[0], y=lm[1]) for lm in det.landmarks]

            detection_results.append(
                FaceDetectionResult(
                    bbox=bbox,
                    confidence=det.confidence,
                    landmarks=landmarks
                )
            )

        logger.info(f"Detected {len(detections)} face(s) in {processing_time:.2f}ms")

        return DetectionResponse(
            faces_detected=len(detections),
            detections=detection_results,
            image_width=width,
            image_height=height,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera/snapshot")
async def get_camera_snapshot(draw_detections: bool = True):
    """
    Capture a snapshot from the camera with optional face detection overlay.

    Args:
        draw_detections: Whether to draw face detection boxes

    Returns:
        JPEG image
    """
    try:
        # Import get_camera from recognition routes to use singleton
        from app.api.routes.recognition import get_camera

        camera = get_camera()  # Use SAME singleton camera as preview stream

        # Read frame from the already-connected camera
        # Light flush (3 frames) for responsiveness while staying in sync
        ret, frame = camera.read_frame(crop_osd=False, flush_buffer=True)

        if not ret or frame is None:
            raise HTTPException(status_code=503, detail="Failed to capture frame from camera")

        # Optionally detect and draw faces
        if draw_detections:
            detections = face_detector.detect_faces(frame)
            if detections:
                frame = face_detector.draw_detections(frame, detections, draw_landmarks=True)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        io_buf = io.BytesIO(buffer)

        # Return with cache control headers to prevent browser caching
        return StreamingResponse(
            io_buf,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error capturing snapshot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera/detect")
async def detect_faces_from_camera():
    """
    Capture frame from camera and detect faces.

    Returns:
        Detection results
    """
    try:
        camera = CameraHandler(use_main_stream=False)

        if not camera.connect():
            return CameraFrameResponse(
                success=False,
                message="Failed to connect to camera",
                timestamp=datetime.utcnow()
            )

        # Read frame
        ret, frame = camera.read_frame()
        camera.disconnect()

        if not ret or frame is None:
            return CameraFrameResponse(
                success=False,
                message="Failed to capture frame",
                timestamp=datetime.utcnow()
            )

        # Detect faces
        detections = face_detector.detect_faces(frame)

        return CameraFrameResponse(
            success=True,
            message=f"Detected {len(detections)} face(s)",
            timestamp=datetime.utcnow(),
            faces_detected=len(detections)
        )

    except Exception as e:
        logger.error(f"Error in camera detection: {str(e)}")
        return CameraFrameResponse(
            success=False,
            message=f"Error: {str(e)}",
            timestamp=datetime.utcnow()
        )
