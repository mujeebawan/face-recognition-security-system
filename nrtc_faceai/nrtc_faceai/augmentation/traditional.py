"""
NRTC Face AI - Traditional Augmentation (Licensed)
Image augmentation techniques for face images.
"""

import cv2
import numpy as np
from typing import List
import logging

from ..license.validator import LicenseValidator
from ..utils.exceptions import LicenseError

logger = logging.getLogger(__name__)


class FaceAugmentation:
    """NRTC licensed traditional augmentation techniques for face images"""

    _license_validated = False
    _validator = None

    def __init__(self, license_path: str = None):
        """
        Initialize NRTC Face Augmentation.

        Args:
            license_path: Path to license file (optional)

        Raises:
            LicenseError: If license validation fails
        """
        self._validate_license(license_path)

    @classmethod
    def _validate_license(cls, license_path: str = None):
        """Validate license on first use"""
        if not cls._license_validated:
            cls._validator = LicenseValidator(license_path)
            cls._validator.validate()

            if not cls._validator.check_feature('augmentation') and not cls._validator.check_feature('*'):
                raise LicenseError("Augmentation feature not enabled in license")

            cls._license_validated = True

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        """Flip image horizontally"""
        return cv2.flip(image, 1)

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, sigma: float = 10) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast"""
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted

    @staticmethod
    def slight_blur(image: np.ndarray) -> np.ndarray:
        """Apply slight Gaussian blur"""
        return cv2.GaussianBlur(image, (3, 3), 0)

    def generate_variations(self, image: np.ndarray, num_variations: int = 10) -> List[np.ndarray]:
        """
        Generate multiple variations of a face image.

        Args:
            image: Original face image
            num_variations: Number of variations to generate

        Returns:
            List of augmented images
        """
        variations = [image.copy()]

        augmentations = [
            lambda img: self.rotate_image(img, -15),
            lambda img: self.rotate_image(img, -10),
            lambda img: self.rotate_image(img, -5),
            lambda img: self.rotate_image(img, 5),
            lambda img: self.rotate_image(img, 10),
            lambda img: self.rotate_image(img, 15),
            lambda img: self.adjust_brightness(img, 0.7),
            lambda img: self.adjust_brightness(img, 0.85),
            lambda img: self.adjust_brightness(img, 1.15),
            lambda img: self.adjust_brightness(img, 1.3),
            lambda img: self.adjust_contrast(img, 0.8),
            lambda img: self.adjust_contrast(img, 1.2),
            lambda img: self.flip_horizontal(img),
            lambda img: self.slight_blur(img),
            lambda img: self.add_gaussian_noise(img, 5),
            lambda img: self.adjust_brightness(self.rotate_image(img, -10), 0.9),
            lambda img: self.adjust_brightness(self.rotate_image(img, 10), 1.1),
            lambda img: self.adjust_contrast(self.rotate_image(img, -5), 1.1),
            lambda img: self.adjust_contrast(self.rotate_image(img, 5), 0.9),
        ]

        for i, aug_func in enumerate(augmentations):
            if len(variations) >= num_variations:
                break

            try:
                augmented = aug_func(image)
                variations.append(augmented)
            except Exception as e:
                logger.warning(f"Augmentation {i} failed: {str(e)}")

        logger.info(f"Generated {len(variations)} image variations (including original)")

        return variations[:num_variations]
