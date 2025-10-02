"""
Image augmentation module for face recognition.
Generates variations of face images to improve recognition accuracy.
"""

import cv2
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceAugmentation:
    """Traditional augmentation techniques for face images"""

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        return rotated

    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.

        Args:
            image: Input image
            factor: Brightness factor (0.5=darker, 1.5=brighter)

        Returns:
            Brightness-adjusted image
        """
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
        """
        Add Gaussian noise to image.

        Args:
            image: Input image
            sigma: Noise standard deviation

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.

        Args:
            image: Input image
            factor: Contrast factor (1.0=original, >1.0=more contrast)

        Returns:
            Contrast-adjusted image
        """
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
        variations = [image.copy()]  # Original image

        augmentations = [
            # Rotations
            lambda img: self.rotate_image(img, -15),
            lambda img: self.rotate_image(img, -10),
            lambda img: self.rotate_image(img, -5),
            lambda img: self.rotate_image(img, 5),
            lambda img: self.rotate_image(img, 10),
            lambda img: self.rotate_image(img, 15),

            # Brightness
            lambda img: self.adjust_brightness(img, 0.7),
            lambda img: self.adjust_brightness(img, 0.85),
            lambda img: self.adjust_brightness(img, 1.15),
            lambda img: self.adjust_brightness(img, 1.3),

            # Contrast
            lambda img: self.adjust_contrast(img, 0.8),
            lambda img: self.adjust_contrast(img, 1.2),

            # Flip
            lambda img: self.flip_horizontal(img),

            # Blur
            lambda img: self.slight_blur(img),

            # Noise
            lambda img: self.add_gaussian_noise(img, 5),

            # Combined
            lambda img: self.adjust_brightness(self.rotate_image(img, -10), 0.9),
            lambda img: self.adjust_brightness(self.rotate_image(img, 10), 1.1),
            lambda img: self.adjust_contrast(self.rotate_image(img, -5), 1.1),
            lambda img: self.adjust_contrast(self.rotate_image(img, 5), 0.9),
        ]

        # Generate variations
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

    def augment_face_crop(self, face_image: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         num_variations: int = 10) -> List[np.ndarray]:
        """
        Augment a cropped face region.

        Args:
            face_image: Full image
            bbox: Bounding box (x, y, width, height)
            num_variations: Number of variations

        Returns:
            List of augmented face crops
        """
        x, y, w, h = bbox

        # Add padding
        padding = int(min(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(face_image.shape[1], x + w + padding)
        y2 = min(face_image.shape[0], y + h + padding)

        # Crop face
        face_crop = face_image[y1:y2, x1:x2]

        # Generate variations
        return self.generate_variations(face_crop, num_variations)
