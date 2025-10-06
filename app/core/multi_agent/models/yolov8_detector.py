"""
YOLOv8-Face Detection Model

Ultra-fast face detection using YOLOv8 architecture
"""

import logging
import time
from typing import List, Dict, Optional, Any
import numpy as np
import cv2

from app.core.multi_agent.engine import BaseModel, ModelResult
from app.core.multi_agent.utils.cuda_streams import cuda_stream_manager

logger = logging.getLogger(__name__)


class YOLOv8FaceDetector(BaseModel):
    """
    YOLOv8-Face detection model

    Fast face detection with bounding boxes and confidence scores
    """

    def __init__(self, stream_id: int = 0):
        super().__init__(model_name="YOLOv8-Face", stream_id=stream_id)
        self.model = None
        self.conf_threshold = 0.5

    async def initialize(self):
        """Initialize YOLOv8 model"""
        logger.info(f"Initializing YOLOv8-Face on stream {self.stream_id}...")
        start_time = time.time()

        try:
            # Try to use ultralytics YOLOv8
            try:
                from ultralytics import YOLO

                # For now, use standard YOLOv8 (will download weights automatically)
                # TODO: Switch to YOLOv8-face specific weights when available
                self.model = YOLO('yolov8n.pt')  # Nano model for speed

                # Warm-up
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model.predict(dummy, verbose=False)

                logger.info("✓ Using Ultralytics YOLOv8")

            except ImportError:
                # Fallback: Use Haar Cascade
                logger.warning("Ultralytics not available, using Haar Cascade fallback")
                self.model = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                logger.info("✓ Using Haar Cascade fallback")

            # Register with CUDA stream manager
            cuda_stream_manager.assign_model_to_stream(
                self.model_name,
                self.stream_id
            )

            self.initialized = True
            elapsed = time.time() - start_time
            logger.info(f"✓ YOLOv8-Face initialized in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8-Face: {e}")
            raise

    async def infer(
        self,
        image: np.ndarray,
        database_embeddings: Optional[List[np.ndarray]] = None,
        database_persons: Optional[List[Dict]] = None,
        threshold: float = 0.5,
        **kwargs
    ) -> ModelResult:
        """
        Run YOLOv8 face detection

        Args:
            image: Input image (BGR format)

        Returns:
            ModelResult with detection info
        """
        if not self.initialized:
            raise RuntimeError("YOLOv8-Face not initialized")

        start_time = time.time()

        try:
            if hasattr(self.model, 'predict'):
                # Ultralytics YOLO
                faces = self._detect_yolo(image)
            else:
                # Haar Cascade fallback
                faces = self._detect_haar(image)

            inference_time = (time.time() - start_time) * 1000

            if len(faces) == 0:
                return ModelResult(
                    model_name=self.model_name,
                    person_id=None,
                    person_name=None,
                    confidence=0.0,
                    embedding=None,
                    bbox=None,
                    metadata={'num_faces': 0},
                    inference_time=inference_time
                )

            # Return first (highest confidence) face
            best_face = faces[0]

            # YOLOv8 doesn't do recognition, just detection
            # So we don't have person_id/name
            return ModelResult(
                model_name=self.model_name,
                person_id=None,  # Detection only, no recognition
                person_name=None,
                confidence=best_face['confidence'],
                embedding=None,
                bbox=best_face['bbox'],
                metadata={
                    'num_faces': len(faces),
                    'all_faces': faces
                },
                inference_time=inference_time
            )

        except Exception as e:
            logger.error(f"YOLOv8 inference error: {e}")
            return ModelResult(
                model_name=self.model_name,
                person_id=None,
                person_name=None,
                confidence=0.0,
                embedding=None,
                bbox=None,
                metadata={'error': str(e)},
                inference_time=(time.time() - start_time) * 1000
            )

    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using YOLO"""
        results = self.model.predict(image, verbose=False, conf=self.conf_threshold)

        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                faces.append({
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'confidence': conf
                })

        # Sort by confidence
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        return faces

    def _detect_opencv_dnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV DNN (placeholder)"""
        # TODO: Implement OpenCV DNN detection
        return self._detect_haar(image)

    def _detect_haar(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rect = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        faces = []
        for (x, y, w, h) in faces_rect:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 0.9  # Haar doesn't provide confidence
            })

        return faces

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up YOLOv8-Face...")
        self.model = None
        self.initialized = False
