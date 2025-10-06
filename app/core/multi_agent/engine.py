"""
Parallel Inference Engine for Multi-Agent Face Recognition

This engine orchestrates multiple models running in parallel on GPU using
CUDA streams for maximum throughput and accuracy.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from a single model"""
    model_name: str
    person_id: Optional[int]
    person_name: Optional[str]
    confidence: float
    embedding: Optional[np.ndarray]
    bbox: Optional[Tuple[int, int, int, int]]
    metadata: Dict[str, Any]
    inference_time: float


@dataclass
class AgentResult:
    """Final result from multi-agent system"""
    person_id: Optional[int]
    person_name: Optional[str]
    confidence: float
    trust_score: float  # 0-100
    model_results: List[ModelResult]
    consensus_count: int
    quality_score: float
    liveness_score: float
    is_live: bool
    total_inference_time: float
    metadata: Dict[str, Any]


class BaseModel:
    """Base class for all model wrappers"""

    def __init__(self, model_name: str, stream_id: int = 0):
        self.model_name = model_name
        self.stream_id = stream_id
        self.initialized = False

    async def initialize(self):
        """Initialize model (async for parallel loading)"""
        raise NotImplementedError

    async def infer(self, image: np.ndarray, **kwargs) -> ModelResult:
        """Run inference on image"""
        raise NotImplementedError

    def cleanup(self):
        """Cleanup resources"""
        pass


class ParallelInferenceEngine:
    """
    Multi-Agent Parallel Inference Engine

    Orchestrates multiple models running in parallel using CUDA streams
    and async execution for maximum GPU utilization and accuracy.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parallel inference engine

        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        self.models: Dict[str, BaseModel] = {}
        self.initialized = False
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Performance tracking
        self.stats = {
            'total_inferences': 0,
            'avg_latency': 0.0,
            'avg_trust_score': 0.0,
        }

        logger.info("ParallelInferenceEngine initialized")

    def register_model(self, model: BaseModel):
        """
        Register a model with the engine

        Args:
            model: Model instance to register
        """
        if model.model_name in self.models:
            logger.warning(f"Model {model.model_name} already registered, replacing...")

        self.models[model.model_name] = model
        logger.info(f"Registered model: {model.model_name} on stream {model.stream_id}")

    async def initialize_all_models(self):
        """Initialize all registered models in parallel"""
        logger.info(f"Initializing {len(self.models)} models in parallel...")
        start_time = time.time()

        # Initialize all models concurrently
        init_tasks = [
            model.initialize()
            for model in self.models.values()
        ]

        await asyncio.gather(*init_tasks)

        self.initialized = True
        elapsed = time.time() - start_time
        logger.info(f"✓ All models initialized in {elapsed:.2f}s")

    async def run_parallel_inference(
        self,
        image: np.ndarray,
        database_embeddings: Optional[List[np.ndarray]] = None,
        database_persons: Optional[List[Dict]] = None,
        threshold: float = 0.6
    ) -> AgentResult:
        """
        Run parallel inference across all models

        Args:
            image: Input image (BGR format)
            database_embeddings: List of known face embeddings
            database_persons: List of person metadata
            threshold: Similarity threshold for matching

        Returns:
            AgentResult with fused predictions
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize_all_models() first.")

        start_time = time.time()

        # Run all models in parallel
        inference_tasks = [
            model.infer(
                image,
                database_embeddings=database_embeddings,
                database_persons=database_persons,
                threshold=threshold
            )
            for model in self.models.values()
        ]

        model_results = await asyncio.gather(*inference_tasks)

        # Fuse results
        final_result = self._fuse_results(model_results)
        final_result.total_inference_time = (time.time() - start_time) * 1000  # ms

        # Update stats
        self._update_stats(final_result)

        return final_result

    def _fuse_results(self, model_results: List[ModelResult]) -> AgentResult:
        """
        Fuse results from multiple models using voting

        Args:
            model_results: List of results from each model

        Returns:
            Fused AgentResult
        """
        # Filter out None results (no detection)
        valid_results = [r for r in model_results if r.person_id is not None]

        if not valid_results:
            # No detections from any model
            return AgentResult(
                person_id=None,
                person_name="Unknown",
                confidence=0.0,
                trust_score=0.0,
                model_results=model_results,
                consensus_count=0,
                quality_score=0.0,
                liveness_score=0.0,
                is_live=False,
                total_inference_time=0.0,
                metadata={'reason': 'no_detection'}
            )

        # Voting: Count predictions for each person_id
        votes = {}
        confidence_sums = {}

        for result in valid_results:
            person_id = result.person_id
            if person_id not in votes:
                votes[person_id] = 0
                confidence_sums[person_id] = 0.0

            votes[person_id] += 1
            confidence_sums[person_id] += result.confidence

        # Get winner (most votes)
        winner_id = max(votes.items(), key=lambda x: x[1])[0]
        consensus_count = votes[winner_id]
        avg_confidence = confidence_sums[winner_id] / consensus_count

        # Get person name from first matching result
        person_name = next(
            r.person_name for r in valid_results
            if r.person_id == winner_id
        )

        # Calculate trust score based on consensus
        total_models = len(model_results)
        consensus_ratio = consensus_count / total_models if total_models > 0 else 0
        trust_score = (consensus_ratio * 0.6 + avg_confidence * 0.4) * 100

        # Extract quality and liveness (if available)
        quality_score = 0.8  # TODO: Get from quality model
        liveness_score = 0.9  # TODO: Get from liveness model
        is_live = liveness_score > 0.5

        return AgentResult(
            person_id=winner_id,
            person_name=person_name,
            confidence=avg_confidence,
            trust_score=trust_score,
            model_results=model_results,
            consensus_count=consensus_count,
            quality_score=quality_score,
            liveness_score=liveness_score,
            is_live=is_live,
            total_inference_time=0.0,  # Set by caller
            metadata={
                'total_votes': sum(votes.values()),
                'vote_distribution': votes,
                'consensus_ratio': consensus_ratio
            }
        )

    def _update_stats(self, result: AgentResult):
        """Update performance statistics"""
        self.stats['total_inferences'] += 1

        # Running average
        n = self.stats['total_inferences']
        self.stats['avg_latency'] = (
            self.stats['avg_latency'] * (n - 1) + result.total_inference_time
        ) / n
        self.stats['avg_trust_score'] = (
            self.stats['avg_trust_score'] * (n - 1) + result.trust_score
        ) / n

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'num_models': len(self.models),
            'models': list(self.models.keys())
        }

    def cleanup(self):
        """Cleanup all resources"""
        logger.info("Cleaning up ParallelInferenceEngine...")

        for model in self.models.values():
            model.cleanup()

        self.executor.shutdown(wait=True)
        logger.info("✓ Cleanup complete")
