"""
NRTC Face AI - Proprietary Face Recognition Library
Copyright (c) 2025 NRTC. All rights reserved.

This software is protected by license agreement.
Unauthorized use, distribution, or reverse engineering is prohibited.
"""

__version__ = "1.0.0"
__author__ = "NRTC"
__license__ = "Proprietary"

# Import core modules
from .core import (
    FaceDetector,
    FaceDetection,
    FaceRecognizer,
    FaceEmbeddingResult
)

from .augmentation import FaceAugmentation

from .license import (
    LicenseValidator,
    HardwareIdentifier
)

from .utils import (
    LicenseError,
    HardwareBindingError,
    LicenseExpiredError,
    InvalidLicenseError
)

__all__ = [
    # Core
    'FaceDetector',
    'FaceDetection',
    'FaceRecognizer',
    'FaceEmbeddingResult',

    # Augmentation
    'FaceAugmentation',

    # License
    'LicenseValidator',
    'HardwareIdentifier',

    # Exceptions
    'LicenseError',
    'HardwareBindingError',
    'LicenseExpiredError',
    'InvalidLicenseError',
]
