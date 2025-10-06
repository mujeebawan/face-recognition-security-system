"""
FaceNet Recognition Model

Google's FaceNet model for face recognition with 128-D embeddings
"""

import logging
import time
from typing import List, Dict, Optional, Any
import numpy as np
import cv2
import torch

from app.core.multi_agent.engine import BaseModel, ModelResult
from app.core.multi_agent.utils.cuda_streams import cuda_stream_manager

logger = logging.getLogger(__name__)


class FaceNetModel(BaseModel):
    """
    FaceNet recognition model

    Uses Google's FaceNet (Inception-ResNet) architecture
    for robust face recognition with 128-D embeddings
    """

    def __init__(self, stream_id: int = 2):
        super().__init__(model_name="FaceNet", stream_id=stream_id)
        self.model = None
        self.device = None
        self.mtcnn = None

    async def initialize(self):
        """Initialize FaceNet model"""
        logger.info(f"Initializing FaceNet on stream {self.stream_id}...")
        start_time = time.time()

        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1

            # Setup device
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # Load detection model (MTCNN)
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                device=self.device,
                post_process=False
            )

            # Load recognition model (pretrained on VGGFace2)
            self.model = InceptionResnetV1(
                pretrained='vggface2',
                device=self.device
            ).eval()

            # Register with CUDA stream manager
            cuda_stream_manager.assign_model_to_stream(
                self.model_name,
                self.stream_id
            )

            self.initialized = True
            elapsed = time.time() - start_time
            logger.info(f"✓ FaceNet initialized in {elapsed:.2f}s (device: {self.device})")

        except ImportError as e:
            logger.error(f"FaceNet dependencies not installed: {e}")
            logger.info("Install with: pip install facenet-pytorch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FaceNet: {e}")
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
        Run FaceNet inference

        Args:
            image: Input image (BGR format)
            database_embeddings: Known face embeddings
            database_persons: Person metadata
            threshold: Similarity threshold

        Returns:
            ModelResult with prediction
        """
        if not self.initialized:
            raise RuntimeError("FaceNet not initialized")

        start_time = time.time()

        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face
            boxes, probs = self.mtcnn.detect(rgb_image)

            if boxes is None or len(boxes) == 0:
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

            # Get first face (highest confidence)
            box = boxes[0]
            prob = probs[0]

            # Crop and align face
            x1, y1, x2, y2 = [int(b) for b in box]
            face = rgb_image[y1:y2, x1:x2]

            # Resize to 160x160
            face = cv2.resize(face, (160, 160))

            # Convert to tensor
            face_tensor = torch.from_numpy(face).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
            face_tensor = face_tensor.unsqueeze(0).to(self.device)

            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy()[0]

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Match against database
            person_id = None
            person_name = None
            similarity = 0.0

            if database_embeddings and database_persons:
                # Compute cosine similarity
                similarities = []
                for db_emb in database_embeddings:
                    # Convert ArcFace (512-D) to comparable format if needed
                    # For now, skip if dimensions don't match
                    if db_emb.shape[0] != embedding.shape[0]:
                        similarities.append(0.0)
                        continue

                    sim = np.dot(embedding, db_emb)
                    similarities.append(float(sim))

                if similarities:
                    best_idx = np.argmax(similarities)
                    similarity = similarities[best_idx]

                    if similarity >= threshold:
                        person_id = database_persons[best_idx]['id']
                        person_name = database_persons[best_idx]['name']

            inference_time = (time.time() - start_time) * 1000

            return ModelResult(
                model_name=self.model_name,
                person_id=person_id,
                person_name=person_name,
                confidence=float(similarity),
                embedding=embedding,
                bbox=(x1, y1, x2-x1, y2-y1),
                metadata={
                    'det_confidence': float(prob),
                    'embedding_dim': embedding.shape[0]
                },
                inference_time=inference_time
            )

        except Exception as e:
            logger.error(f"FaceNet inference error: {e}")
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
        logger.info("Cleaning up FaceNet...")
        self.model = None
        self.mtcnn = None
        self.initialized = False
