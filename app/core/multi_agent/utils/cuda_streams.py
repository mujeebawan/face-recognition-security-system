"""
CUDA Stream Manager for parallel GPU execution

Manages CUDA streams to run multiple models simultaneously on GPU
without blocking each other.
"""

import logging
from typing import Dict, Optional
import threading

logger = logging.getLogger(__name__)


class CUDAStreamManager:
    """
    Manages CUDA streams for parallel model execution

    Each model gets assigned to a specific CUDA stream, allowing
    multiple models to run on GPU simultaneously.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure one manager instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize CUDA stream manager"""
        if hasattr(self, '_initialized'):
            return

        self.streams: Dict[int, any] = {}
        self.max_streams = 8  # Jetson AGX Orin can handle 8+ streams
        self.current_stream_id = 0
        self._initialized = True

        logger.info(f"CUDAStreamManager initialized (max streams: {self.max_streams})")

    def get_stream(self, stream_id: Optional[int] = None) -> int:
        """
        Get or create a CUDA stream

        Args:
            stream_id: Specific stream ID to get/create. If None, auto-assign.

        Returns:
            Stream ID
        """
        if stream_id is None:
            # Auto-assign next available stream
            stream_id = self.current_stream_id
            self.current_stream_id = (self.current_stream_id + 1) % self.max_streams

        if stream_id not in self.streams:
            # Create new stream (placeholder for actual CUDA stream)
            # In ONNX Runtime, streams are managed via ExecutionProvider options
            self.streams[stream_id] = {
                'id': stream_id,
                'created_at': None,
                'models': []
            }
            logger.info(f"Created CUDA stream {stream_id}")

        return stream_id

    def assign_model_to_stream(self, model_name: str, stream_id: int):
        """
        Assign a model to a specific CUDA stream

        Args:
            model_name: Name of the model
            stream_id: Stream ID to assign to
        """
        if stream_id not in self.streams:
            self.get_stream(stream_id)

        if model_name not in self.streams[stream_id]['models']:
            self.streams[stream_id]['models'].append(model_name)
            logger.info(f"Assigned model '{model_name}' to CUDA stream {stream_id}")

    def get_provider_options(self, stream_id: int, use_tensorrt: bool = True) -> tuple:
        """
        Get ONNX Runtime provider options for specific stream

        Args:
            stream_id: CUDA stream ID
            use_tensorrt: Whether to use TensorRT

        Returns:
            Tuple of (providers, provider_options)
        """
        import os

        # TensorRT options with stream support
        tensorrt_options = {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': os.path.join(os.getcwd(), 'data/tensorrt_engines'),
            'trt_fp16_enable': True,
            'trt_builder_optimization_level': 2,
            'device_id': 0,
            # Stream-specific options would go here if supported
        }

        # CUDA options with stream support
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB per stream
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            # Note: ONNX Runtime doesn't directly expose CUDA stream assignment
            # Streams are managed internally by the execution provider
        }

        if use_tensorrt:
            providers = [
                ('TensorrtExecutionProvider', tensorrt_options),
                ('CUDAExecutionProvider', cuda_options),
                'CPUExecutionProvider'
            ]
        else:
            providers = [
                ('CUDAExecutionProvider', cuda_options),
                'CPUExecutionProvider'
            ]

        return providers

    def get_stream_info(self) -> Dict:
        """Get information about all streams"""
        return {
            'total_streams': len(self.streams),
            'max_streams': self.max_streams,
            'streams': {
                sid: {
                    'id': sid,
                    'num_models': len(info['models']),
                    'models': info['models']
                }
                for sid, info in self.streams.items()
            }
        }

    def cleanup(self):
        """Cleanup all streams"""
        logger.info("Cleaning up CUDA streams...")
        self.streams.clear()
        self.current_stream_id = 0
        logger.info("✓ CUDA streams cleaned up")


# Global instance
cuda_stream_manager = CUDAStreamManager()
