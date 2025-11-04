"""
LivePortrait-based face pose augmentation for enrollment.
Generates realistic pose variations (left, right, up, down, frontal) from a single face image.
"""

import os
import sys
import numpy as np
import cv2
import torch
from typing import List, Optional
from pathlib import Path
import logging

# Read LivePortrait path from environment variable
LIVEPORTRAIT_PATH = os.getenv("LIVEPORTRAIT_PATH", "/home/mujeeb/Downloads/LivePortrait")

# Add LivePortrait to path if it exists
if LIVEPORTRAIT_PATH and os.path.exists(LIVEPORTRAIT_PATH):
    if LIVEPORTRAIT_PATH not in sys.path:
        sys.path.insert(0, LIVEPORTRAIT_PATH)
else:
    raise ImportError(
        f"LivePortrait not found at: {LIVEPORTRAIT_PATH}\n"
        "Please install LivePortrait and set LIVEPORTRAIT_PATH in .env file.\n"
        "See README.md section 'LivePortrait Installation' for instructions."
    )

from src.live_portrait_wrapper import LivePortraitWrapper
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

logger = logging.getLogger(__name__)


class LivePortraitAugmentor:
    """
    Uses LivePortrait (CVPR 2024) for 3D-aware face pose generation.
    Generates realistic pose variations by transferring poses from template images.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = True,
        liveportrait_path: str = LIVEPORTRAIT_PATH
    ):
        """
        Initialize LivePortrait augmentor.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            use_fp16: Use half-precision for faster inference
            liveportrait_path: Path to LivePortrait repository
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.liveportrait_path = Path(liveportrait_path)

        # Initialize LivePortrait
        logger.info("Initializing LivePortrait models...")
        self.inference_cfg = InferenceConfig(
            device_id=0,
            flag_use_half_precision=use_fp16,
            flag_force_cpu=(device == "cpu"),
            flag_do_torch_compile=False  # Disable compilation for faster startup
        )

        self.wrapper = LivePortraitWrapper(inference_cfg=self.inference_cfg)

        # Define pose templates (5 diverse angles)
        self.pose_templates = {
            "left_profile": "d9.jpg",      # Strong left turn
            "right_profile": "d8.jpg",     # Strong right turn
            "looking_up": "d12.jpg",       # Looking up
            "looking_down": "d19.jpg",     # Looking down
            "frontal_slight": "d38.jpg"    # Frontal with slight variation
        }

        # Load and cache pose template images
        self.driving_images = {}
        self.driving_kp_info = {}
        self._load_pose_templates()

        logger.info(f"✓ LivePortrait initialized with {len(self.pose_templates)} pose templates")

    def _load_pose_templates(self):
        """Load and preprocess pose template images."""
        driving_dir = self.liveportrait_path / "assets" / "examples" / "driving"

        for pose_name, filename in self.pose_templates.items():
            img_path = driving_dir / filename

            if not img_path.exists():
                logger.warning(f"Pose template not found: {img_path}")
                continue

            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Prepare for LivePortrait (resize to 256x256)
            img_tensor = self.wrapper.prepare_source(img)

            # Extract keypoint information (this defines the pose)
            kp_info = self.wrapper.get_kp_info(img_tensor)

            # Cache both image and keypoints
            self.driving_images[pose_name] = img_tensor
            self.driving_kp_info[pose_name] = kp_info

            logger.debug(f"Loaded pose template: {pose_name} from {filename}")

    def _rotate_keypoints(self, kp: torch.Tensor, angle_type: str, strength: float = 0.5) -> torch.Tensor:
        """
        Rotate keypoints to simulate head movement while preserving identity 100%.

        Args:
            kp: Source keypoints tensor
            angle_type: Type of rotation ('left', 'right', 'up', 'down', 'frontal')
            strength: Rotation strength (0.0 to 1.0)

        Returns:
            Rotated keypoints tensor
        """
        kp_rotated = kp.clone()

        # Adjust keypoints based on angle (preserves facial features exactly)
        if angle_type == 'left_profile' or angle_type == 'left':
            # Turn head left
            kp_rotated[:, :, 0] -= strength * 0.12
            kp_rotated[:, :, 2] -= strength * 0.04

        elif angle_type == 'right_profile' or angle_type == 'right':
            # Turn head right
            kp_rotated[:, :, 0] += strength * 0.12
            kp_rotated[:, :, 2] -= strength * 0.04

        elif angle_type == 'looking_up' or angle_type == 'up':
            # Look up
            kp_rotated[:, :, 1] -= strength * 0.10
            kp_rotated[:, :, 2] += strength * 0.03

        elif angle_type == 'looking_down' or angle_type == 'down':
            # Look down
            kp_rotated[:, :, 1] += strength * 0.10
            kp_rotated[:, :, 2] += strength * 0.03

        elif angle_type == 'frontal_slight' or angle_type == 'frontal':
            # Slight frontal variation
            kp_rotated[:, :, 0] += strength * 0.03

        return kp_rotated

    def generate_face_angles(
        self,
        reference_image: np.ndarray,
        num_variations: int = 5,
        angles: Optional[List[str]] = None,
        **kwargs  # Accept but ignore SD-specific params
    ) -> List[np.ndarray]:
        """
        Generate face variations with different poses using LivePortrait.
        ✨ Identity-Preserving Mode: Uses ONLY your keypoints (no reference faces).

        Args:
            reference_image: Input face image (HxWx3, BGR, uint8)
            num_variations: Number of variations to generate (default: 5)
            angles: Optional list of specific angles to generate
                   (e.g., ['left_profile', 'right_profile', 'looking_up'])
                   If None, uses standard angles
            **kwargs: Ignored (for compatibility with SD augmentors)

        Returns:
            List of augmented face images (numpy arrays, BGR, uint8)
        """
        logger.info(f"Generating {num_variations} pose variations with LivePortrait (identity-preserving)...")

        # Prepare source image
        source_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        source_tensor = self.wrapper.prepare_source(source_rgb)

        # Extract source features and keypoints
        source_feature_3d = self.wrapper.extract_feature_3d(source_tensor)
        source_kp_info = self.wrapper.get_kp_info(source_tensor)

        # Define target angles
        if angles is not None:
            angle_list = angles[:num_variations]
        else:
            # Default angles
            angle_list = ['left_profile', 'right_profile', 'looking_up', 'looking_down', 'frontal_slight'][:num_variations]

        results = []
        for i, angle in enumerate(angle_list):
            try:
                logger.info(f"[{i+1}/{len(angle_list)}] Generating {angle} pose...")

                # Rotate YOUR keypoints directly (100% identity preservation)
                rotated_kp = self._rotate_keypoints(
                    source_kp_info['kp'],
                    angle,
                    strength=0.6  # Moderate rotation for natural look
                )

                # Generate warped image using your rotated keypoints
                ret_dct = self.wrapper.warp_decode(
                    source_feature_3d,
                    source_kp_info['kp'],  # Original as reference
                    rotated_kp  # Your rotated keypoints (not someone else's!)
                )

                # Parse output to numpy array
                output = self.wrapper.parse_output(ret_dct['out'])  # 1xHxWx3, uint8, RGB
                output_img = output[0]  # Remove batch dimension

                # Convert back to BGR for consistency with OpenCV
                output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

                results.append(output_bgr)
                logger.debug(f"✓ Generated pose: {angle}")

            except Exception as e:
                logger.error(f"Failed to generate pose {angle}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: use original image
                results.append(reference_image.copy())

        logger.info(f"✅ Generated {len(results)} pose variations (identity preserved)")
        return results

    def get_available_poses(self) -> List[str]:
        """Get list of available pose template names."""
        return list(self.driving_kp_info.keys())

    def __del__(self):
        """Cleanup GPU memory."""
        try:
            if hasattr(self, 'wrapper'):
                del self.wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


# Test function
def test_liveportrait_augmentor():
    """Test the LivePortrait augmentor with a sample image."""
    import matplotlib.pyplot as plt

    # Create augmentor
    augmentor = LivePortraitAugmentor(device="cuda", use_fp16=True)

    # Load a test image
    test_img_path = "/home/mujeeb/Downloads/LivePortrait/assets/examples/source/s6.jpg"
    if not os.path.exists(test_img_path):
        print(f"Test image not found: {test_img_path}")
        return

    img = cv2.imread(test_img_path)

    # Generate variations
    variations = augmentor.generate_face_angles(img, num_variations=5)

    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Variations
    pose_names = augmentor.get_available_poses()
    for i, (var, pose_name) in enumerate(zip(variations, pose_names), 1):
        axes[i].imshow(cv2.cvtColor(var, cv2.COLOR_BGR2RGB))
        axes[i].set_title(pose_name)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("/home/mujeeb/Downloads/liveportrait_test_results.png", dpi=150)
    print("✓ Results saved to /home/mujeeb/Downloads/liveportrait_test_results.png")


if __name__ == "__main__":
    test_liveportrait_augmentor()
