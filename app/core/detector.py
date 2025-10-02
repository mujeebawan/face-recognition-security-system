"""
Face detection module using MediaPipe.
Detects faces in images and video frames, returns bounding boxes and landmarks.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Data class for face detection results"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # [(x, y), ...]


class FaceDetector:
    """Face detector using MediaPipe Face Detection"""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe face detector.

        Args:
            min_detection_confidence: Minimum confidence threshold (0-1)
        """
        self.min_detection_confidence = min_detection_confidence

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range (better for camera feeds)
            min_detection_confidence=min_detection_confidence
        )

        logger.info(f"Face detector initialized (confidence threshold: {min_detection_confidence})")

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

        # Convert BGR to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width = image.shape[:2]

        # Detect faces
        results = self.detector.process(image_rgb)

        detections = []

        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute pixels
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                confidence = detection.score[0]

                # Extract key landmarks (eyes, nose, mouth, ears)
                landmarks = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        lm_x = int(keypoint.x * width)
                        lm_y = int(keypoint.y * height)
                        landmarks.append((lm_x, lm_y))

                face = FaceDetection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    landmarks=landmarks if landmarks else None
                )

                detections.append(face)

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
        if hasattr(self, 'detector'):
            self.detector.close()
