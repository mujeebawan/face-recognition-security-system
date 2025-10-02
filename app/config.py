"""
Application configuration module.
Loads settings from environment variables using pydantic-settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file"""

    # Camera Configuration
    camera_ip: str = "192.168.1.64"
    camera_username: str = "admin"
    camera_password: str
    camera_rtsp_port: int = 554
    camera_main_stream: str
    camera_sub_stream: str

    # Application Settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    log_level: str = "INFO"

    # Database Configuration
    database_url: str = "sqlite:///./face_recognition.db"

    # Face Recognition Settings
    face_detection_confidence: float = 0.5
    face_recognition_threshold: float = 0.6
    max_face_distance: float = 0.6

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Model Paths
    face_detection_model: str = "data/models/face_detection"
    face_recognition_model: str = "data/models/face_recognition"

    # Image Storage
    images_dir: str = "data/images"
    embeddings_dir: str = "data/embeddings"

    # Performance Settings
    enable_gpu: bool = True
    use_tensorrt: bool = False
    frame_skip: int = 2
    max_workers: int = 4

    # Augmentation Settings
    enable_augmentation: bool = True
    augmentation_count: int = 5
    use_diffusion: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures settings are loaded only once.
    """
    return Settings()


# Convenience instance for importing
settings = get_settings()
