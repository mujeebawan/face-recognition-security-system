"""
Multi-Agent Parallel Face Recognition System

This module implements a parallel inference engine that runs multiple face
recognition models simultaneously on GPU using CUDA streams for maximum
performance and accuracy.

Architecture:
- Multiple detection models (YOLOv8, RetinaFace)
- Multiple recognition models (ArcFace, FaceNet, AdaFace)
- Transformer models (CLIP, DINOv2, Temporal)
- Quality and liveness assessment
- Fusion layer with voting and confidence scoring
"""

from .engine import ParallelInferenceEngine

__all__ = ['ParallelInferenceEngine']
