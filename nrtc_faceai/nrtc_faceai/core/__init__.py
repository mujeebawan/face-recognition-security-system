"""NRTC Face AI - Core Face Recognition"""
from .detector import FaceDetector, FaceDetection
from .recognizer import FaceRecognizer, FaceEmbeddingResult

__all__ = [
    'FaceDetector',
    'FaceDetection',
    'FaceRecognizer',
    'FaceEmbeddingResult'
]
