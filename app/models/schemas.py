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


# Alert Schemas
class AlertConfig(BaseModel):
    """Alert configuration"""
    enabled: bool = True
    alert_on_unknown: bool = True
    alert_on_known: bool = False
    min_confidence_unknown: float = 0.5
    cooldown_seconds: int = 60
    webhook_url: Optional[str] = None
    email_recipients: List[str] = []
    save_snapshot: bool = True


class AlertEvent(BaseModel):
    """Alert event data"""
    id: Optional[int] = None
    timestamp: datetime
    event_type: str  # 'unknown_person', 'known_person', 'multiple_unknown'
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    num_faces: int = 1
    snapshot_path: Optional[str] = None

    # Admin acknowledgment
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    notes: Optional[str] = None

    # Guard verification
    guard_verified: bool = False
    guard_action: Optional[str] = None  # 'confirmed', 'false_alarm', 'investigating', 'apprehended', 'escalated'
    guard_verified_by: Optional[str] = None
    guard_verified_at: Optional[datetime] = None
    action_notes: Optional[str] = None

    # Watchlist info (cached from person)
    threat_level: Optional[str] = None  # 'critical', 'high', 'medium', 'low', 'none'
    watchlist_status: Optional[str] = None  # 'most_wanted', 'suspect', 'person_of_interest', 'banned', 'none'

    original_image_url: Optional[str] = None  # URL to original enrolled person image

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    """Response for alert query"""
    success: bool
    total: int
    alerts: List[AlertEvent]


class AlertAcknowledgeRequest(BaseModel):
    """Request to acknowledge alert"""
    alert_id: int
    acknowledged_by: str
    notes: Optional[str] = None


# WebSocket Schemas
class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str  # 'recognition', 'alert', 'status', 'error'
    data: dict
    timestamp: str


# Configuration Schemas
class SystemConfig(BaseModel):
    """System configuration"""
    face_recognition_threshold: float = Field(ge=0.0, le=1.0)
    face_detection_confidence: float = Field(ge=0.0, le=1.0)
    enable_gpu: bool
    frame_skip: int = Field(ge=0, le=10)  # 0 = process all frames
    recognition_frequency: int = Field(ge=1, le=60)


class SystemConfigUpdateRequest(BaseModel):
    """Request to update system configuration"""
    face_recognition_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    face_detection_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    frame_skip: Optional[int] = Field(None, ge=0, le=10)  # 0 = process all frames
    recognition_frequency: Optional[int] = Field(None, ge=1, le=60)


class SystemStatus(BaseModel):
    """System status information"""
    status: str  # 'running', 'error', 'starting'
    uptime_seconds: float
    total_enrolled_persons: int
    total_recognition_logs: int
    active_alerts: int
    camera_connected: bool
    gpu_available: bool
    current_fps: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
