"""
NRTC Face AI - Face Detector (Licensed)
GPU-accelerated face detection using SCRFD with license enforcement.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from insightface.app import FaceAnalysis

from ..license.validator import LicenseValidator
from ..utils.exceptions import LicenseError

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Data class for face detection results"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # [(x, y), ...]


class FaceDetector:
    """
    NRTC licensed face detector using SCRFD from InsightFace.
    GPU-accelerated with hardware binding.
    """

    _license_validated = False
    _validator = None

    def __init__(self, min_detection_confidence: float = 0.5, license_path: str = None):
        """
        Initialize NRTC Face Detector.

        Args:
            min_detection_confidence: Minimum confidence threshold (0-1)
            license_path: Path to license file (optional)

        Raises:
            LicenseError: If license validation fails
        """
        # Validate license first
        self._validate_license(license_path)

        self.min_detection_confidence = min_detection_confidence

        logger.info("Initializing NRTC SCRFD face detector (CUDA)...")

        # Initialize FaceAnalysis with SCRFD detector
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=[
                ('CUDAExecutionProvider', cuda_options),
                'CPUExecutionProvider'
            ]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
            det_thresh=min_detection_confidence
        )

        logger.info(f"✓ NRTC Face Detector initialized (GPU, threshold={min_detection_confidence})")

    @classmethod
    def _validate_license(cls, license_path: str = None):
        """Validate license on first use"""
        if not cls._license_validated:
            cls._validator = LicenseValidator(license_path)
            cls._validator.validate()

            # Check if face detection feature is enabled
            if not cls._validator.check_feature('face_detection') and not cls._validator.check_feature('*'):
                raise LicenseError("Face detection feature not enabled in license")

            cls._license_validated = True

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

        # Detect faces using SCRFD
        faces = self.app.get(image)

        detections = []

        if faces:
            height, width = image.shape[:2]

            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                x = max(0, x1)
                y = max(0, y1)
                w = min(x2 - x1, width - x)
                h = min(y2 - y1, height - y)

                confidence = float(face.det_score)

                # Extract landmarks
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

            # Draw landmarks
            if draw_landmarks and detection.landmarks:
                for lm_x, lm_y in detection.landmarks:
                    cv2.circle(output, (lm_x, lm_y), 3, (0, 0, 255), -1)

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
