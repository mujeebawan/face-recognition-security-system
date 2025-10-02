"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int


class Landmark(BaseModel):
    """Facial landmark coordinates"""
    x: int
    y: int


class FaceDetectionResult(BaseModel):
    """Single face detection result"""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0, le=1)
    landmarks: Optional[List[Landmark]] = None


class DetectionResponse(BaseModel):
    """Response for face detection endpoint"""
    faces_detected: int
    detections: List[FaceDetectionResult]
    image_width: int
    image_height: int
    processing_time_ms: float


class CameraFrameResponse(BaseModel):
    """Response for camera frame endpoint"""
    success: bool
    message: str
    timestamp: datetime
    faces_detected: Optional[int] = None
