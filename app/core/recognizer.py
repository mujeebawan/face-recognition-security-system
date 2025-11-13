"""
Face recognition module using InsightFace (ArcFace).
Extracts face embeddings and performs face matching.
"""

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import threading

from app.config import settings
from app.core.settings_manager import get_setting

logger = logging.getLogger(__name__)


@dataclass
class FaceEmbeddingResult:
    """Result of face embedding extraction"""
    embedding: np.ndarray  # 512-D vector
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[np.ndarray] = None


class FaceRecognizer:
    """Face recognition using InsightFace ArcFace model"""

    def __init__(self):
        """Initialize InsightFace face recognition"""
        logger.info("Initializing InsightFace face recognition...")

        # Thread lock for TensorRT thread-safety
        self._lock = threading.Lock()

        # Initialize FaceAnalysis with buffalo_l model
        # Use TensorRT with FP16 for maximum GPU acceleration

        # TensorRT options with FP16 enabled
        trt_options = {
            'device_id': 0,
            'trt_fp16_enable': True,  # Enable FP16 precision
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': 'data/tensorrt_engines',
            'trt_max_workspace_size': 2147483648,  # 2GB workspace
        }

        # CUDA fallback options
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=[
                ('TensorrtExecutionProvider', trt_options),
                ('CUDAExecutionProvider', cuda_options),
                'CPUExecutionProvider'
            ]
        )

        # Prepare model (downloads if needed)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info("✓ Face recognition model loaded successfully (TensorRT FP16, thread-safe)")

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
            # Thread-safe GPU access - acquire lock before calling TensorRT model
            with self._lock:
                # Detect and extract features
                faces = self.app.get(image)

            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None

            # Use the first detected face (highest confidence)
            face = faces[0]

            # Extract bounding box
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            width = x2 - x
            height = y2 - y

            # Extract embedding (512-D vector)
            embedding = face.embedding

            # Normalize embedding
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
            # Thread-safe GPU access - acquire lock before calling TensorRT model
            with self._lock:
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
        # Reshape for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)

        # Compute cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]

        return float(similarity)

    def match_face(self, query_embedding: np.ndarray,
                   database_embeddings: List[np.ndarray],
                   threshold: float = None,
                   embedding_ids: Optional[List[int]] = None) -> Tuple[int, float, Optional[int]]:
        """
        Match query embedding against database of embeddings.

        Args:
            query_embedding: Query face embedding
            database_embeddings: List of database embeddings
            threshold: Similarity threshold (default from config)
            embedding_ids: Optional list of embedding IDs (parallel to database_embeddings)

        Returns:
            Tuple of (best_match_index, similarity_score, matched_embedding_id)
            Returns (-1, 0.0, None) if no match found
        """
        if threshold is None:
            # Try to get from database settings first, fallback to config
            recognition_threshold = get_setting('recognition_threshold', settings.max_face_distance)
            threshold = recognition_threshold if recognition_threshold <= 1.0 else (1.0 - recognition_threshold)
            logger.debug(f"Using recognition threshold: {threshold:.3f} (from {'database' if recognition_threshold != settings.max_face_distance else 'config'})")

        if len(database_embeddings) == 0:
            return -1, 0.0, None

        # Compute similarities with all database embeddings
        similarities = []
        for db_embedding in database_embeddings:
            sim = self.compare_embeddings(query_embedding, db_embedding)
            similarities.append(sim)

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        # Get the matched embedding ID if provided
        matched_embedding_id = None
        if embedding_ids and len(embedding_ids) > best_idx:
            matched_embedding_id = embedding_ids[best_idx]

        # Check if it meets threshold
        if best_similarity >= threshold:
            logger.info(f"Match found: index={best_idx}, similarity={best_similarity:.3f}, embedding_id={matched_embedding_id}")
            return best_idx, best_similarity, matched_embedding_id
        else:
            logger.info(f"No match found (best similarity: {best_similarity:.3f} < threshold: {threshold:.3f})")
            return -1, 0.0, None

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """
        Serialize numpy embedding to bytes for database storage.

        Args:
            embedding: Numpy array

        Returns:
            Pickled bytes
        """
        return pickle.dumps(embedding)

    @staticmethod
    def deserialize_embedding(data: bytes) -> np.ndarray:
        """
        Deserialize embedding from database bytes.

        Args:
            data: Pickled bytes

        Returns:
            Numpy array
        """
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

        # Draw landmarks if available
        if result.landmarks is not None:
            for landmark in result.landmarks:
                lx, ly = int(landmark[0]), int(landmark[1])
                cv2.circle(output, (lx, ly), 2, (0, 0, 255), -1)

        return output
