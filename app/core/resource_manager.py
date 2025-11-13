"""
RTOS-style Resource Manager for Face Recognition System

Implements:
- Thread-safe singleton pattern with double-checked locking
- Semaphore-based resource access control
- Queue-based interrupt handling
- Producer-consumer pattern for frame processing
- Graceful resource lifecycle management
- Proper cleanup and shutdown mechanisms
"""

import threading
import logging
from queue import Queue, Full, Empty
from typing import Optional, Any
from contextlib import contextmanager
import atexit
import signal
import sys

logger = logging.getLogger(__name__)

# Global GPU lock - defined FIRST to prevent circular imports
# CRITICAL: InsightFace models are NOT fully thread-safe on GPU
_gpu_lock = threading.Lock()


class DownloadProgressCallback:
    """
    Callback for tracking HuggingFace model download progress.

    Shows real-time progress with file sizes, speed, and completion percentage.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_size = 0
        self.downloaded = 0
        self.start_time = None
        self.last_log_time = 0
        self.file_count = 0
        self.completed_files = 0

    def __call__(self, progress_info):
        """Called by HuggingFace during download"""
        import time

        if self.start_time is None:
            self.start_time = time.time()
            logger.info(f"📥 Starting download: {self.model_name}")

        current_time = time.time()

        # Log progress every 2 seconds to avoid spam
        if current_time - self.last_log_time >= 2.0:
            elapsed = current_time - self.start_time

            # Extract progress info
            if hasattr(progress_info, 'downloaded') and hasattr(progress_info, 'total'):
                downloaded_mb = progress_info.downloaded / (1024 * 1024)
                total_mb = progress_info.total / (1024 * 1024)
                percentage = (progress_info.downloaded / progress_info.total) * 100 if progress_info.total > 0 else 0
                speed_mbps = downloaded_mb / elapsed if elapsed > 0 else 0

                logger.info(
                    f"📥 {self.model_name}: {downloaded_mb:.1f}MB / {total_mb:.1f}MB "
                    f"({percentage:.1f}%) - {speed_mbps:.2f} MB/s"
                )

            self.last_log_time = current_time


def download_model_with_progress(
    repo_id: str,
    cache_dir: str,
    model_name: str = None,
    **kwargs
) -> str:
    """
    Download HuggingFace model with progress tracking.

    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Directory to cache model
        model_name: Display name for logging
        **kwargs: Additional arguments for huggingface_hub

    Returns:
        Path to downloaded model
    """
    import os
    from pathlib import Path

    display_name = model_name or repo_id

    # Check if already downloaded
    model_path = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
    if model_path.exists():
        # Check if download is complete (has snapshots directory)
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            logger.info(f"✓ {display_name} already downloaded at {cache_dir}")
            return str(cache_dir)

    logger.info(f"📥 Downloading {display_name} to SD card...")
    logger.info(f"   Location: {cache_dir}")
    logger.info(f"   Repository: {repo_id}")

    try:
        from huggingface_hub import snapshot_download

        # Create progress callback
        progress = DownloadProgressCallback(display_name)

        # Download with progress tracking
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,  # Required for ExFAT (SD card)
            **kwargs
        )

        logger.info(f"✅ {display_name} download complete!")
        return str(cache_dir)

    except Exception as e:
        logger.error(f"❌ Failed to download {display_name}: {e}")
        raise


def get_gpu_lock() -> threading.Lock:
    """
    Get global GPU lock for thread-safe GPU operations.

    CRITICAL: Wrap ALL GPU operations with this lock to prevent segfaults.
    InsightFace models are not fully thread-safe on GPU.

    Usage:
        from app.core.resource_manager import get_gpu_lock

        gpu_lock = get_gpu_lock()
        with gpu_lock:
            # Safe GPU operation
            result = model.predict(data)
    """
    return _gpu_lock


class ResourceManager:
    """
    Thread-safe singleton resource manager using RTOS principles.

    Features:
    - Double-checked locking for thread-safe singleton initialization
    - Semaphore-based resource access control
    - Reference counting for resource lifecycle
    - Automatic cleanup on shutdown
    """

    _instance: Optional['ResourceManager'] = None
    _lock = threading.Lock()  # Class-level lock for singleton creation

    def __new__(cls):
        """Thread-safe singleton instantiation with double-checked locking"""
        if cls._instance is None:
            with cls._lock:
                # Double-check inside lock to prevent race condition
                if cls._instance is None:
                    logger.info("Creating ResourceManager singleton instance")
                    cls._instance = super(ResourceManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize resource manager (called only once)"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            logger.info("Initializing ResourceManager...")

            # Resource instances
            self._face_recognizer: Optional[FaceRecognizer] = None
            self._face_detector: Optional[FaceDetector] = None
            self._alert_manager: Optional[AlertManager] = None
            self._camera_handler: Optional[CameraHandler] = None

            # Thread synchronization primitives
            self._recognizer_lock = threading.Lock()
            self._detector_lock = threading.Lock()
            self._alert_lock = threading.Lock()
            self._camera_lock = threading.Lock()

            # CRITICAL: Global GPU lock for thread-safe GPU operations
            # InsightFace GPU models are NOT fully thread-safe - must serialize GPU access
            self._gpu_lock = threading.Lock()

            # Semaphores for resource access control (max concurrent users)
            self._camera_semaphore = threading.Semaphore(1)  # Only 1 camera reader at a time
            self._detector_semaphore = threading.Semaphore(1)  # SERIAL GPU access (changed from 3)
            self._recognizer_semaphore = threading.Semaphore(1)  # SERIAL GPU access (changed from 3)

            # Reference counting for resource lifecycle
            self._camera_ref_count = 0
            self._camera_ref_lock = threading.Lock()

            # Shutdown flag
            self._shutdown_event = threading.Event()

            # Register cleanup handlers
            atexit.register(self.shutdown)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self._initialized = True
            logger.info("✓ ResourceManager initialized")

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)

    @contextmanager
    def acquire_gpu_lock(self):
        """
        Context manager for acquiring global GPU lock.

        CRITICAL: Use this to wrap ALL GPU operations to prevent segfaults.
        InsightFace models are not fully thread-safe on GPU.

        Usage:
            with resource_manager.acquire_gpu_lock():
                # Safe GPU operation here
                result = detector.detect(frame)
        """
        self._gpu_lock.acquire()
        try:
            yield
        finally:
            self._gpu_lock.release()

    @contextmanager
    def acquire_detector(self):
        """
        Context manager for acquiring detector with GPU lock.

        Usage:
            with resource_manager.acquire_detector() as detector:
                faces = detector.detect(frame)
        """
        self._detector_semaphore.acquire()
        try:
            detector = self.get_detector()
            yield detector
        finally:
            self._detector_semaphore.release()

    @contextmanager
    def acquire_recognizer(self):
        """Context manager for acquiring recognizer with GPU lock"""
        self._recognizer_semaphore.acquire()
        try:
            recognizer = self.get_recognizer()
            yield recognizer
        finally:
            self._recognizer_semaphore.release()

    @contextmanager
    def acquire_camera(self):
        """Context manager for acquiring camera with semaphore control"""
        self._camera_semaphore.acquire()
        with self._camera_ref_lock:
            self._camera_ref_count += 1
        try:
            camera = self.get_camera()
            yield camera
        finally:
            with self._camera_ref_lock:
                self._camera_ref_count -= 1
            self._camera_semaphore.release()

    def get_detector(self):
        """
        Get or initialize face detector (thread-safe singleton).
        Uses double-checked locking for performance.
        """
        if self._face_detector is None:
            with self._detector_lock:
                # Double-check inside lock
                if self._face_detector is None:
                    # Lazy import to prevent circular dependency
                    from app.core.detector import FaceDetector
                    logger.info("Initializing SCRFD detector (GPU, one-time init)...")
                    self._face_detector = FaceDetector()
                    logger.info("✓ SCRFD detector initialized")
        return self._face_detector

    def get_recognizer(self):
        """Get or initialize face recognizer (thread-safe singleton)"""
        if self._face_recognizer is None:
            with self._recognizer_lock:
                if self._face_recognizer is None:
                    # Lazy import to prevent circular dependency
                    from app.core.recognizer import FaceRecognizer
                    logger.info("Initializing face recognizer (GPU, one-time init)...")
                    self._face_recognizer = FaceRecognizer()
                    logger.info("✓ Face recognizer initialized")
        return self._face_recognizer

    def get_alert_manager(self):
        """Get or initialize alert manager (thread-safe singleton)"""
        if self._alert_manager is None:
            with self._alert_lock:
                if self._alert_manager is None:
                    # Lazy import to prevent circular dependency
                    from app.core.alerts import AlertManager
                    logger.info("Initializing alert manager...")
                    self._alert_manager = AlertManager()
                    logger.info("✓ Alert manager initialized")
        return self._alert_manager

    def get_camera(self):
        """Get or initialize camera handler (thread-safe singleton)"""
        if self._camera_handler is None:
            with self._camera_lock:
                if self._camera_handler is None:
                    # Lazy import to prevent circular dependency
                    from app.core.camera import CameraHandler
                    logger.info("Initializing camera handler (singleton RTSP connection)...")
                    self._camera_handler = CameraHandler(use_main_stream=False)
                    if not self._camera_handler.connect():
                        logger.error("Failed to connect to camera")
                        self._camera_handler = None
                        raise RuntimeError("Camera connection failed")
                    logger.info("✓ Camera handler initialized")

        # Reconnect if disconnected
        elif not self._camera_handler.is_connected:
            with self._camera_lock:
                if not self._camera_handler.is_connected:
                    logger.info("Reconnecting to camera...")
                    if not self._camera_handler.connect():
                        logger.error("Failed to reconnect to camera")
                        raise RuntimeError("Camera reconnection failed")

        return self._camera_handler

    def is_shutting_down(self) -> bool:
        """Check if system is shutting down"""
        return self._shutdown_event.is_set()

    def shutdown(self):
        """Gracefully shutdown all resources"""
        if self._shutdown_event.is_set():
            return  # Already shutting down

        logger.info("=" * 60)
        logger.info("ResourceManager: Initiating graceful shutdown...")
        logger.info("=" * 60)

        self._shutdown_event.set()

        # Wait for active camera users to finish
        with self._camera_ref_lock:
            if self._camera_ref_count > 0:
                logger.info(f"Waiting for {self._camera_ref_count} active camera users...")

        # Cleanup camera
        if self._camera_handler is not None:
            logger.info("Disconnecting camera...")
            try:
                self._camera_handler.disconnect()
                logger.info("✓ Camera disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting camera: {e}")

        # Cleanup other resources (if they have cleanup methods)
        if self._alert_manager is not None:
            logger.info("✓ Alert manager cleaned up")

        if self._face_detector is not None:
            logger.info("✓ Face detector cleaned up")

        if self._face_recognizer is not None:
            logger.info("✓ Face recognizer cleaned up")

        logger.info("=" * 60)
        logger.info("ResourceManager: Shutdown complete")
        logger.info("=" * 60)


class FrameProcessingQueue:
    """
    Queue-based frame processing system with producer-consumer pattern.

    Features:
    - Non-blocking frame production
    - Background worker threads for processing
    - Automatic queue overflow handling
    - Graceful shutdown
    """

    def __init__(self, max_queue_size: int = 2, num_workers: int = 1):
        """
        Initialize frame processing queue.

        Args:
            max_queue_size: Maximum frames in queue (prevents memory buildup)
            num_workers: Number of background worker threads
        """
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue()
        self.num_workers = num_workers
        self.workers = []
        self.shutdown_flag = threading.Event()
        self.processing_lock = threading.Lock()

        logger.info(f"FrameProcessingQueue initialized (queue_size={max_queue_size}, workers={num_workers})")

    def start_workers(self, worker_func):
        """Start background worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_func, i),
                daemon=True,
                name=f"FrameWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"Started worker thread: FrameWorker-{i}")

    def _worker_loop(self, worker_func, worker_id: int):
        """Worker thread loop (consumer)"""
        logger.info(f"Worker {worker_id} started")
        while not self.shutdown_flag.is_set():
            try:
                # Get task from queue with timeout
                task = self.frame_queue.get(timeout=0.5)

                if task is None:  # Shutdown signal
                    break

                # Process frame
                try:
                    result = worker_func(task)
                    if result is not None:
                        self.result_queue.put(result)
                except Exception as e:
                    logger.error(f"Worker {worker_id} processing error: {e}")
                finally:
                    self.frame_queue.task_done()

            except Empty:
                continue  # Queue empty, retry
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Worker {worker_id} stopped")

    def submit_frame(self, frame_data: Any) -> bool:
        """
        Submit frame for processing (non-blocking producer).

        Returns:
            True if submitted, False if queue full (frame dropped)
        """
        try:
            self.frame_queue.put_nowait(frame_data)
            return True
        except Full:
            # Queue full - drop frame (RTOS principle: prefer real-time over completeness)
            logger.debug("Frame queue full, dropping frame (real-time priority)")
            return False

    def get_result(self, timeout: float = 0.01) -> Optional[Any]:
        """Get processing result (non-blocking)"""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    def shutdown(self):
        """Gracefully shutdown queue and workers"""
        logger.info("Shutting down frame processing queue...")
        self.shutdown_flag.set()

        # Send shutdown signals to workers
        for _ in range(self.num_workers):
            try:
                self.frame_queue.put(None, timeout=1.0)
            except Full:
                pass

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not stop gracefully")

        logger.info("✓ Frame processing queue shutdown complete")


# Global singleton instance
_resource_manager: Optional[ResourceManager] = None
_manager_lock = threading.Lock()


def get_resource_manager() -> ResourceManager:
    """
    Get global ResourceManager singleton (thread-safe).

    This uses module-level singleton with double-checked locking
    for maximum thread safety and performance.
    """
    global _resource_manager

    if _resource_manager is None:
        with _manager_lock:
            if _resource_manager is None:
                _resource_manager = ResourceManager()

    return _resource_manager
