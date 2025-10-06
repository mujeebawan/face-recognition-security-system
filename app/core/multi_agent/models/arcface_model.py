"""
ArcFace Model Wrapper for Multi-Agent System

Adapts the existing InsightFace ArcFace model to work with
the parallel inference engine.
"""

import logging
import time
from typing import List, Dict, Optional, Any
import numpy as np

from app.core.recognizer import FaceRecognizer
from app.core.multi_agent.engine import BaseModel, ModelResult
from app.core.multi_agent.utils.cuda_streams import cuda_stream_manager

logger = logging.getLogger(__name__)


class ArcFaceModel(BaseModel):
    """
    ArcFace recognition model wrapper

    Wraps the existing InsightFace ArcFace (buffalo_l) model
    for use in the multi-agent parallel system.
    """

    def __init__(self, stream_id: int = 1):
        """
        Initialize ArcFace model

        Args:
            stream_id: CUDA stream ID for parallel execution
        """
        super().__init__(model_name="ArcFace", stream_id=stream_id)
        self.recognizer: Optional[FaceRecognizer] = None
        self.match_threshold = 0.6

    async def initialize(self):
        """Initialize ArcFace model"""
        logger.info(f"Initializing ArcFace model on stream {self.stream_id}...")
        start_time = time.time()

        try:
            # Initialize existing recognizer
            self.recognizer = FaceRecognizer()

            # Register with CUDA stream manager
            cuda_stream_manager.assign_model_to_stream(
                self.model_name,
                self.stream_id
            )

            self.initialized = True
            elapsed = time.time() - start_time
            logger.info(f"✓ ArcFace initialized in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to initialize ArcFace: {e}")
            raise

    async def infer(
        self,
        image: np.ndarray,
        database_embeddings: Optional[List[np.ndarray]] = None,
        database_persons: Optional[List[Dict]] = None,
        threshold: float = 0.6,
        **kwargs
    ) -> ModelResult:
        """
        Run ArcFace inference

        Args:
            image: Input image (BGR format)
            database_embeddings: Known face embeddings
            database_persons: Person metadata
            threshold: Similarity threshold

        Returns:
            ModelResult with prediction
        """
        if not self.initialized:
            raise RuntimeError("ArcFace model not initialized")

        start_time = time.time()

        try:
            # Extract embedding
            result = self.recognizer.extract_embedding(image)

            if result is None:
                # No face detected
                return ModelResult(
                    model_name=self.model_name,
                    person_id=None,
                    person_name=None,
                    confidence=0.0,
                    embedding=None,
                    bbox=None,
                    metadata={'error': 'no_face_detected'},
                    inference_time=(time.time() - start_time) * 1000
                )

            # Match against database
            person_id = None
            person_name = None
            similarity = 0.0

            if database_embeddings and database_persons:
                match_idx, similarity = self.recognizer.match_face(
                    result.embedding,
                    database_embeddings,
                    threshold=threshold
                )

                if match_idx >= 0:
                    person_id = database_persons[match_idx]['id']
                    person_name = database_persons[match_idx]['name']

            inference_time = (time.time() - start_time) * 1000

            return ModelResult(
                model_name=self.model_name,
                person_id=person_id,
                person_name=person_name,
                confidence=float(similarity),
                embedding=result.embedding,
                bbox=result.bbox,
                metadata={
                    'det_confidence': result.confidence,
                    'embedding_dim': result.embedding.shape[0]
                },
                inference_time=inference_time
            )

        except Exception as e:
            logger.error(f"ArcFace inference error: {e}")
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

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ArcFace model...")
        self.recognizer = None
        self.initialized = False
