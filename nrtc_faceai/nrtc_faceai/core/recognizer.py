"""
NRTC Face AI - Face Recognizer (Licensed)
Face recognition using InsightFace ArcFace with license enforcement.
"""

import numpy as np
import cv2
from insightface.app import FaceAnalysis
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from ..license.validator import LicenseValidator
from ..utils.exceptions import LicenseError

logger = logging.getLogger(__name__)


@dataclass
class FaceEmbeddingResult:
    """Result of face embedding extraction"""
    embedding: np.ndarray  # 512-D vector
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[np.ndarray] = None


class FaceRecognizer:
    """
    NRTC licensed face recognizer using InsightFace ArcFace model.
    GPU-accelerated with hardware binding.
    """

    _license_validated = False
    _validator = None

    def __init__(self, license_path: str = None):
        """
        Initialize NRTC Face Recognizer.

        Args:
            license_path: Path to license file (optional)

        Raises:
            LicenseError: If license validation fails
        """
        # Validate license first
        self._validate_license(license_path)

        logger.info("Initializing NRTC InsightFace face recognition...")

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

        self.app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info("✓ NRTC Face recognition model loaded successfully")

    @classmethod
    def _validate_license(cls, license_path: str = None):
        """Validate license on first use"""
        if not cls._license_validated:
            cls._validator = LicenseValidator(license_path)
            cls._validator.validate()

            # Check if face recognition feature is enabled
            if not cls._validator.check_feature('face_recognition') and not cls._validator.check_feature('*'):
                raise LicenseError("Face recognition feature not enabled in license")

            cls._license_validated = True

    def extract_embedding(self, image: np.ndarray) -> Optional[FaceEmbeddingResult]:
        """
        Extract face embedding from image.

        Args:
            image: Input image (BGR format)

        Returns:
            FaceEmbeddingResult or None if no face detected
        """
        if image is None:
            logger.warning("Received None image for embedding extraction")
            return None

        try:
            faces = self.app.get(image)

            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None

            # Use the first detected face
            face = faces[0]

            # Extract bounding box
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            width = x2 - x
            height = y2 - y

            # Extract and normalize embedding
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            result = FaceEmbeddingResult(
                embedding=embedding,
                bbox=(x, y, width, height),
                confidence=float(face.det_score),
                landmarks=face.kps if hasattr(face, 'kps') else None
            )

            logger.debug(f"Extracted embedding: shape={embedding.shape}, confidence={result.confidence:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return None

    def extract_multiple_embeddings(self, image: np.ndarray) -> List[FaceEmbeddingResult]:
        """
        Extract embeddings for all faces in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of FaceEmbeddingResult objects
        """
        if image is None:
            return []

        try:
            faces = self.app.get(image)
            results = []

            for face in faces:
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                width = x2 - x
                height = y2 - y

                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)

                result = FaceEmbeddingResult(
                    embedding=embedding,
                    bbox=(x, y, width, height),
                    confidence=float(face.det_score),
                    landmarks=face.kps if hasattr(face, 'kps') else None
                )

                results.append(result)

            logger.debug(f"Extracted {len(results)} face embedding(s)")
            return results

        except Exception as e:
            logger.error(f"Error extracting multiple embeddings: {str(e)}")
            return []

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)

        similarity = cosine_similarity(emb1, emb2)[0][0]

        return float(similarity)

    def match_face(self, query_embedding: np.ndarray,
                   database_embeddings: List[np.ndarray],
                   threshold: float = 0.4) -> Tuple[int, float]:
        """
        Match query embedding against database of embeddings.

        Args:
            query_embedding: Query face embedding
            database_embeddings: List of database embeddings
            threshold: Similarity threshold

        Returns:
            Tuple of (best_match_index, similarity_score)
            Returns (-1, 0.0) if no match found
        """
        if len(database_embeddings) == 0:
            return -1, 0.0

        # Compute similarities
        similarities = []
        for db_embedding in database_embeddings:
            sim = self.compare_embeddings(query_embedding, db_embedding)
            similarities.append(sim)

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        # Check threshold
        if best_similarity >= threshold:
            logger.info(f"Match found: index={best_idx}, similarity={best_similarity:.3f}")
            return best_idx, best_similarity
        else:
            logger.info(f"No match found (best similarity: {best_similarity:.3f} < threshold: {threshold:.3f})")
            return -1, 0.0

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize numpy embedding to bytes for database storage"""
        return pickle.dumps(embedding)

    @staticmethod
    def deserialize_embedding(data: bytes) -> np.ndarray:
        """Deserialize embedding from database bytes"""
        return pickle.loads(data)

    def draw_embedding_info(self, image: np.ndarray, result: FaceEmbeddingResult,
                           label: str = "") -> np.ndarray:
        """
        Draw bounding box and info on image.

        Args:
            image: Input image
            result: FaceEmbeddingResult
            label: Optional label text

        Returns:
            Image with drawn info
        """
        output = image.copy()
        x, y, w, h = result.bbox

        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw confidence and label
        text = f"{label} {result.confidence:.2f}" if label else f"{result.confidence:.2f}"
        cv2.putText(output, text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw landmarks
        if result.landmarks is not None:
            for landmark in result.landmarks:
                lx, ly = int(landmark[0]), int(landmark[1])
                cv2.circle(output, (lx, ly), 2, (0, 0, 255), -1)

        return output
