"""
CLIP Vision Transformer Model

OpenAI's CLIP for multimodal face understanding
"""

import logging
import time
from typing import List, Dict, Optional, Any
import numpy as np
import cv2
import torch
from PIL import Image

from app.core.multi_agent.engine import BaseModel, ModelResult
from app.core.multi_agent.utils.cuda_streams import cuda_stream_manager

logger = logging.getLogger(__name__)


class CLIPModel(BaseModel):
    """
    CLIP Vision Transformer

    Uses OpenAI's CLIP for robust multimodal face features
    """

    def __init__(self, stream_id: int = 4):
        super().__init__(model_name="CLIP-ViT", stream_id=stream_id)
        self.model = None
        self.processor = None
        self.device = None
        self.face_detector = None

    async def initialize(self):
        """Initialize CLIP model"""
        logger.info(f"Initializing CLIP-ViT on stream {self.stream_id}...")
        start_time = time.time()

        try:
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

            # Setup device
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # Load CLIP model (using ViT-B/32 for balance of speed/accuracy)
            model_name = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = HFCLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            # Face detector for cropping
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Register with CUDA stream manager
            cuda_stream_manager.assign_model_to_stream(
                self.model_name,
                self.stream_id
            )

            self.initialized = True
            elapsed = time.time() - start_time
            logger.info(f"✓ CLIP-ViT initialized in {elapsed:.2f}s (device: {self.device})")

        except ImportError as e:
            logger.error(f"CLIP dependencies not installed: {e}")
            logger.info("Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {e}")
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
        Run CLIP inference

        Args:
            image: Input image (BGR format)
            database_embeddings: Known face embeddings
            database_persons: Person metadata
            threshold: Similarity threshold

        Returns:
            ModelResult with prediction
        """
        if not self.initialized:
            raise RuntimeError("CLIP not initialized")

        start_time = time.time()

        try:
            # Detect face
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
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

            # Get first face
            x, y, w, h = faces[0]
            face_crop = image[y:y+h, x:x+w]

            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_face)

            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract vision features
            with torch.no_grad():
                vision_features = self.model.get_image_features(**inputs)
                embedding = vision_features.cpu().numpy()[0]

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Match against database
            person_id = None
            person_name = None
            similarity = 0.0

            if database_embeddings and database_persons:
                similarities = []
                for db_emb in database_embeddings:
                    # Skip if dimensions don't match
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
                bbox=(x, y, w, h),
                metadata={
                    'embedding_dim': embedding.shape[0],
                    'model_type': 'vision_transformer'
                },
                inference_time=inference_time
            )

        except Exception as e:
            logger.error(f"CLIP inference error: {e}")
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
        logger.info("Cleaning up CLIP...")
        self.model = None
        self.processor = None
        self.face_detector = None
        self.initialized = False
