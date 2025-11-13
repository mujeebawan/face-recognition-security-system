"""
Settings Manager - Runtime configuration from database
Provides dynamic settings that can be updated without restart
"""

from sqlalchemy.orm import Session
from typing import Any, Dict, Optional
import logging
from datetime import datetime

from app.models.database import SystemConfiguration
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Singleton settings manager that caches and dynamically loads settings from database.
    Updates are reflected immediately without requiring service restart.
    """

    _instance = None
    _cache: Dict[str, Any] = {}
    _last_refresh: Optional[datetime] = None
    _cache_ttl_seconds = 10  # Cache for 10 seconds

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._defaults = self._get_defaults()
        # Don't refresh on init - will be lazy loaded on first access

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Default settings as fallback"""
        return {
            "recognition_threshold": 0.35,
            "detection_confidence": 0.5,
            "max_faces_per_frame": 10,
            "alert_cooldown_seconds": 60,
            "alert_on_unknown": True,
            "alert_on_known": True,
            "alert_save_snapshot": True,
            "camera_rtsp_url": "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
            "frame_skip": 5,
            "enable_gpu": True
        }

    def _parse_value(self, value_str: str, data_type: str) -> Any:
        """Parse string value based on data type"""
        if data_type == "float":
            return float(value_str)
        elif data_type == "int":
            return int(value_str)
        elif data_type == "bool":
            return value_str.lower() in ('true', '1', 'yes')
        else:
            return value_str

    def refresh_settings(self, force: bool = False):
        """
        Refresh settings from database.
        Uses cache to avoid excessive DB queries.
        """
        now = datetime.utcnow()

        # Check if cache is still valid
        if not force and self._last_refresh:
            elapsed = (now - self._last_refresh).total_seconds()
            if elapsed < self._cache_ttl_seconds:
                return

        try:
            db = SessionLocal()
            settings = db.query(SystemConfiguration).all()

            # Update cache
            new_cache = self._defaults.copy()
            for setting in settings:
                try:
                    new_cache[setting.config_key] = self._parse_value(
                        setting.config_value,
                        setting.data_type
                    )
                except Exception as e:
                    logger.error(f"Error parsing setting {setting.config_key}: {e}")

            self._cache = new_cache
            self._last_refresh = now
            logger.info(f"Settings refreshed: {len(settings)} loaded from database")

        except Exception as e:
            logger.error(f"Failed to refresh settings from database: {e}")
            # Use defaults on error
            if not self._cache:
                self._cache = self._defaults.copy()
        finally:
            db.close()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key.
        Returns default if key not found.
        """
        # Lazy load on first access or auto-refresh if cache is stale
        if not self._cache or not self._last_refresh:
            self.refresh_settings(force=True)
        else:
            self.refresh_settings()

        return self._cache.get(key, default if default is not None else self._defaults.get(key))

    def get_all(self) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        # Lazy load on first access
        if not self._cache or not self._last_refresh:
            self.refresh_settings(force=True)
        else:
            self.refresh_settings()
        return self._cache.copy()

    def get_active_vs_db_comparison(self) -> Dict[str, Any]:
        """
        Compare active cached settings with current database values.
        Useful for debugging and verification.
        """
        try:
            db = SessionLocal()
            db_settings = db.query(SystemConfiguration).all()

            comparison = {
                "active_settings": self._cache.copy(),
                "database_settings": {},
                "mismatches": [],
                "last_cache_refresh": self._last_refresh.isoformat() if self._last_refresh else None
            }

            for setting in db_settings:
                db_value = self._parse_value(setting.config_value, setting.data_type)
                comparison["database_settings"][setting.config_key] = db_value

                # Check for mismatch
                active_value = self._cache.get(setting.config_key)
                if active_value != db_value:
                    comparison["mismatches"].append({
                        "key": setting.config_key,
                        "active_value": active_value,
                        "db_value": db_value
                    })

            return comparison

        except Exception as e:
            logger.error(f"Error comparing settings: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    def force_reload(self):
        """Force immediate reload of settings from database"""
        self.refresh_settings(force=True)
        logger.info("Settings force reloaded")


# Global singleton instance
settings_manager = SettingsManager()


def get_setting(key: str, default: Any = None) -> Any:
    """Convenience function to get a setting"""
    return settings_manager.get(key, default)


def reload_settings():
    """Convenience function to force reload settings"""
    settings_manager.force_reload()
