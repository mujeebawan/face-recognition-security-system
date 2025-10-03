"""
Camera handler for Hikvision IP camera RTSP streaming.
Manages video capture, frame processing, and stream connection.
"""

import cv2
import logging
from typing import Optional, Tuple
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)


class CameraHandler:
    """Handle RTSP camera stream from Hikvision IP camera"""

    def __init__(self, use_main_stream: bool = True):
        """
        Initialize camera handler.

        Args:
            use_main_stream: If True, use main stream (high quality), else use sub-stream (lower quality)
        """
        self.stream_url = settings.camera_main_stream if use_main_stream else settings.camera_sub_stream
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame_count = 0

        logger.info(f"Camera handler initialized with {'main' if use_main_stream else 'sub'} stream")

    def connect(self) -> bool:
        """
        Connect to the RTSP camera stream.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to camera at {settings.camera_ip}...")

            # Create VideoCapture with RTSP URL
            self.capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            # Set buffer size to reduce latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Try to read a frame to verify connection
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    self.is_connected = True
                    height, width = frame.shape[:2]
                    logger.info(f"✓ Camera connected successfully - Resolution: {width}x{height}")
                    return True
                else:
                    logger.error("✗ Camera opened but cannot read frames")
                    return False
            else:
                logger.error("✗ Failed to open camera stream")
                return False

        except Exception as e:
            logger.error(f"✗ Camera connection error: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from camera stream"""
        if self.capture is not None:
            self.capture.release()
            self.is_connected = False
            logger.info("Camera disconnected")

    def read_frame(self, crop_osd: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.

        Args:
            crop_osd: If True, crop top region to remove camera OSD text overlay

        Returns:
            Tuple of (success, frame)
        """
        if not self.is_connected or self.capture is None:
            logger.warning("Camera not connected")
            return False, None

        try:
            ret, frame = self.capture.read()
            if ret:
                self.frame_count += 1

                # Crop OSD overlay from top
                if crop_osd and frame is not None:
                    # Crop 65 pixels from top to remove camera OSD text
                    frame = frame[65:, :]

                return True, frame
            else:
                logger.warning("Failed to read frame from camera")
                return False, None
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return False, None

    def get_frame_jpeg(self, quality: int = 85) -> Optional[bytes]:
        """
        Get current frame as JPEG bytes.

        Args:
            quality: JPEG quality (0-100)

        Returns:
            JPEG encoded frame bytes or None
        """
        ret, frame = self.read_frame()
        if ret and frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buffer.tobytes()
        return None

    def test_connection(self) -> dict:
        """
        Test camera connection and return diagnostics.

        Returns:
            Dictionary with connection test results
        """
        result = {
            "camera_ip": settings.camera_ip,
            "stream_url": self.stream_url.replace(settings.camera_password, "***"),
            "connected": False,
            "frame_readable": False,
            "resolution": None,
            "fps": None,
            "error": None
        }

        try:
            if self.connect():
                result["connected"] = True

                # Read test frame
                ret, frame = self.read_frame()
                if ret and frame is not None:
                    result["frame_readable"] = True
                    height, width = frame.shape[:2]
                    result["resolution"] = f"{width}x{height}"

                    # Get FPS if available
                    fps = self.capture.get(cv2.CAP_PROP_FPS)
                    result["fps"] = fps if fps > 0 else "Unknown"
                else:
                    result["error"] = "Cannot read frames from camera"
            else:
                result["error"] = "Failed to connect to camera stream"

        except Exception as e:
            result["error"] = str(e)

        return result

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()
