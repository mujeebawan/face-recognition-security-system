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

            # Use FFMPEG with optimal low-latency settings
            # Note: OpenCV not built with GStreamer, using FFMPEG
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay'

            self.capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            # Aggressive low-latency settings
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer

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

    def read_frame(self, crop_osd: bool = True, flush_buffer: bool = False, max_retries: int = 3) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera with automatic reconnection on failure.

        Args:
            crop_osd: If True, crop top region to remove camera OSD text overlay
            flush_buffer: If True, flush old frames to get the latest frame (reduces latency)
            max_retries: Maximum number of reconnection attempts on frame read failure

        Returns:
            Tuple of (success, frame)
        """
        if not self.is_connected or self.capture is None:
            logger.warning("Camera not connected, attempting to connect...")
            if not self.connect():
                return False, None

        retry_count = 0
        while retry_count < max_retries:
            try:
                # Flush old buffered frames to get the latest frame (reduces delay)
                # Grab and discard buffered frames to get the most recent one
                if flush_buffer:
                    # Grab 2 frames to clear buffer without being too aggressive
                    for _ in range(2):
                        self.capture.grab()

                ret, frame = self.capture.read()
                if ret and frame is not None:
                    self.frame_count += 1

                    # Crop OSD overlay from top
                    if crop_osd:
                        # Crop 65 pixels from top to remove camera OSD text
                        frame = frame[65:, :]

                    return True, frame
                else:
                    # Frame read failed - attempt reconnection
                    logger.warning(f"Frame read failed (attempt {retry_count + 1}/{max_retries}), reconnecting...")
                    self.disconnect()
                    if self.connect():
                        retry_count += 1
                        continue
                    else:
                        return False, None

            except Exception as e:
                logger.error(f"Error reading frame: {str(e)}, reconnecting...")
                self.disconnect()
                if retry_count < max_retries - 1:
                    if self.connect():
                        retry_count += 1
                        continue
                return False, None

        logger.error(f"Failed to read frame after {max_retries} reconnection attempts")
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


class FrameBroadcaster:
    """
    Singleton frame broadcaster for sharing a single RTSP stream across multiple viewers.
    Runs a background thread that continuously reads frames and makes them available to multiple clients.
    """
    _instance = None
    _lock = None

    def __new__(cls, use_main_stream: bool = True):
        """Ensure only one instance exists (singleton pattern)"""
        if cls._instance is None:
            import threading
            cls._lock = threading.Lock()
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, use_main_stream: bool = True):
        """Initialize the broadcaster (only once)"""
        if self._initialized:
            return

        import threading
        import time

        self.camera = CameraHandler(use_main_stream=use_main_stream)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.capture_thread = None
        self.last_frame_time = 0
        self._initialized = True

        logger.info("FrameBroadcaster initialized")

    def start(self):
        """Start the background frame capture thread"""
        if self.running:
            logger.warning("FrameBroadcaster already running")
            return

        import threading

        # Connect to camera first
        if not self.camera.connect():
            logger.error("Failed to connect to camera in broadcaster")
            return False

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("FrameBroadcaster started")
        return True

    def stop(self):
        """Stop the background capture thread"""
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2)
        self.camera.disconnect()
        logger.info("FrameBroadcaster stopped")

    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        import time

        while self.running:
            try:
                ret, frame = self.camera.read_frame(crop_osd=True, flush_buffer=False)
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.last_frame_time = time.time()
                else:
                    # Frame read failed, wait a bit before retry
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.5)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame (safe for multiple viewers to call simultaneously).

        Returns:
            Tuple of (success, frame_copy)
        """
        import time

        with self.frame_lock:
            if self.latest_frame is not None:
                # Check if frame is recent (within last 5 seconds)
                if time.time() - self.last_frame_time < 5.0:
                    return True, self.latest_frame.copy()
                else:
                    logger.warning("Latest frame is stale")
                    return False, None
            else:
                return False, None

    def is_alive(self) -> bool:
        """Check if broadcaster is running and receiving frames"""
        import time
        return self.running and (time.time() - self.last_frame_time < 5.0)
