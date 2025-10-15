"""
Face detection module using SCRFD (InsightFace).
GPU-accelerated face detector compatible with MediaPipe interface.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Data class for face detection results"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # [(x, y), ...]


class FaceDetector:
    """Face detector using SCRFD from InsightFace (GPU-accelerated)"""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize SCRFD face detector.

        Args:
            min_detection_confidence: Minimum confidence threshold (0-1)
        """
        self.min_detection_confidence = min_detection_confidence

        logger.info("Initializing SCRFD face detector (TensorRT)...")

        # Initialize FaceAnalysis with SCRFD detector
        # This uses the 'det_10g' model from buffalo_l pack (10G FLOPs SCRFD)
        # Configure TensorRT with engine caching for optimal Jetson performance
        import os
        tensorrt_options = {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': os.path.join(os.getcwd(), 'data/tensorrt_engines'),
            'trt_fp16_enable': True,  # FP16 for faster inference on Jetson
        }

        self.app = FaceAnalysis(
            name='buffalo_l',  # Uses SCRFD det_10g detector
            providers=[
                ('TensorrtExecutionProvider', tensorrt_options),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        )

        # Prepare with GPU context and detection size
        self.app.prepare(
            ctx_id=0,  # GPU 0
            det_size=(640, 640),
            det_thresh=min_detection_confidence
        )

        logger.info(f"SCRFD detector initialized (GPU, threshold={min_detection_confidence})")

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            List of FaceDetection objects
        """
        if image is None:
            logger.warning("Received None image for detection")
            return []

        # Get image dimensions
        height, width = image.shape[:2]

        # Detect faces using SCRFD (expects BGR)
        faces = self.app.get(image)

        detections = []

        if faces:
            for face in faces:
                # Get bounding box (format: [x1, y1, x2, y2])
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Convert to (x, y, width, height) format
                x = max(0, x1)
                y = max(0, y1)
                w = min(x2 - x1, width - x)
                h = min(y2 - y1, height - y)

                confidence = float(face.det_score)

                # Extract landmarks if available (5 keypoints: eyes, nose, mouth corners)
                landmarks = []
                if hasattr(face, 'kps') and face.kps is not None:
                    kps = face.kps.astype(int)
                    for i in range(kps.shape[0]):
                        landmarks.append((int(kps[i, 0]), int(kps[i, 1])))

                face_det = FaceDetection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    landmarks=landmarks if landmarks else None
                )

                detections.append(face_det)

            logger.debug(f"Detected {len(detections)} face(s)")

        return detections

    def draw_detections(self, image: np.ndarray, detections: List[FaceDetection],
                       draw_landmarks: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on image.

        Args:
            image: Input image
            detections: List of FaceDetection objects
            draw_landmarks: Whether to draw facial landmarks

        Returns:
            Image with drawn detections
        """
        output = image.copy()

        for detection in detections:
            x, y, w, h = detection.bbox
            confidence = detection.confidence

            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

            # Draw confidence score
            label = f"{confidence:.2f}"
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(output, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw landmarks if available
            if draw_landmarks and detection.landmarks:
                for lm_x, lm_y in detection.landmarks:
                    cv2.circle(output, (lm_x, lm_y), 3, (0, 0, 255), -1)  # Red dots

        # Draw face count
        count_text = f"Faces: {len(detections)}"
        cv2.putText(output, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return output

    def crop_face(self, image: np.ndarray, detection: FaceDetection,
                  padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop face region from image with optional padding.

        Args:
            image: Input image
            detection: FaceDetection object
            padding: Padding around face (0.2 = 20% padding)

        Returns:
            Cropped face image or None
        """
        if image is None:
            return None

        x, y, w, h = detection.bbox
        height, width = image.shape[:2]

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(width, x + w + pad_w)
        y2 = min(height, y + h + pad_h)

        face_crop = image[y1:y2, x1:x2]

        return face_crop if face_crop.size > 0 else None

    def __del__(self):
        """Cleanup detector on deletion"""
        # InsightFace handles cleanup automatically
        pass
