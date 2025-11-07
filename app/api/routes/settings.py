"""
Settings API Routes
Manage system configuration and runtime settings.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
from pydantic import BaseModel, Field
import json
from datetime import datetime

from app.models.database import SystemConfiguration
from app.config import get_settings
from app.core.database import get_db
from app.core.auth import get_current_user

router = APIRouter(prefix="/api/settings", tags=["settings"])


# Pydantic schemas for request/response
class SettingUpdate(BaseModel):
    """Schema for updating a single setting"""
    key: str
    value: Any
    data_type: str = Field(..., pattern="^(float|int|bool|string|json)$")
    description: str = ""


class BulkSettingsUpdate(BaseModel):
    """Schema for bulk settings update"""
    settings: Dict[str, Any]


class SettingResponse(BaseModel):
    """Schema for setting response"""
    key: str
    value: Any
    data_type: str
    description: str
    updated_at: datetime


# Default settings configuration
DEFAULT_SETTINGS = {
    "recognition_threshold": {
        "value": 0.35,
        "type": "float",
        "description": "Face recognition matching threshold (0.0-1.0). Lower = stricter matching."
    },
    "alert_cooldown_seconds": {
        "value": 60,
        "type": "int",
        "description": "Cooldown period between alerts for the same person (seconds)"
    },
    "alert_on_unknown": {
        "value": True,
        "type": "bool",
        "description": "Generate alerts for unknown persons"
    },
    "alert_on_known": {
        "value": True,
        "type": "bool",
        "description": "Generate alerts for known persons"
    },
    "alert_save_snapshot": {
        "value": True,
        "type": "bool",
        "description": "Save snapshot images when alerts are triggered"
    },
    "camera_rtsp_url": {
        "value": "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
        "type": "string",
        "description": "RTSP URL for IP camera main stream"
    },
    "detection_confidence": {
        "value": 0.5,
        "type": "float",
        "description": "Minimum confidence for face detection (0.0-1.0)"
    },
    "max_faces_per_frame": {
        "value": 10,
        "type": "int",
        "description": "Maximum number of faces to process per frame"
    },
    "frame_skip": {
        "value": 5,
        "type": "int",
        "description": "Process every Nth frame for recognition (1 = every frame)"
    },
    "enable_gpu": {
        "value": True,
        "type": "bool",
        "description": "Enable GPU acceleration for face detection"
    }
}


def get_setting_from_db(db: Session, key: str) -> SystemConfiguration:
    """Get a setting from database"""
    return db.query(SystemConfiguration).filter(
        SystemConfiguration.config_key == key
    ).first()


def create_or_update_setting(db: Session, key: str, value: Any, data_type: str, description: str = ""):
    """Create or update a setting in database"""
    setting = get_setting_from_db(db, key)

    # Convert value to string for storage
    if data_type == "json":
        value_str = json.dumps(value)
    elif data_type == "bool":
        value_str = str(value).lower()
    else:
        value_str = str(value)

    if setting:
        # Update existing
        setting.config_value = value_str
        setting.data_type = data_type
        setting.description = description
        setting.updated_at = datetime.utcnow()
    else:
        # Create new
        setting = SystemConfiguration(
            config_key=key,
            config_value=value_str,
            data_type=data_type,
            description=description
        )
        db.add(setting)

    db.commit()
    db.refresh(setting)
    return setting


def parse_setting_value(setting: SystemConfiguration) -> Any:
    """Parse setting value from string based on data_type"""
    if setting.data_type == "float":
        return float(setting.config_value)
    elif setting.data_type == "int":
        return int(setting.config_value)
    elif setting.data_type == "bool":
        return setting.config_value.lower() in ('true', '1', 'yes')
    elif setting.data_type == "json":
        return json.loads(setting.config_value)
    else:
        return setting.config_value


@router.get("/")
async def get_all_settings(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all system settings.
    Returns settings from database, with defaults for missing values.
    """
    settings_dict = {}

    # Get all settings from database
    db_settings = db.query(SystemConfiguration).all()
    db_settings_map = {s.config_key: s for s in db_settings}

    # Merge with defaults
    for key, default_config in DEFAULT_SETTINGS.items():
        if key in db_settings_map:
            # Use database value
            setting = db_settings_map[key]
            settings_dict[key] = {
                "value": parse_setting_value(setting),
                "type": setting.data_type,
                "description": setting.description,
                "updated_at": setting.updated_at.isoformat()
            }
        else:
            # Use default value
            settings_dict[key] = {
                "value": default_config["value"],
                "type": default_config["type"],
                "description": default_config["description"],
                "updated_at": None
            }

    return {
        "success": True,
        "settings": settings_dict
    }


@router.get("/{key}")
async def get_setting(
    key: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific setting by key"""
    setting = get_setting_from_db(db, key)

    if setting:
        return {
            "success": True,
            "setting": {
                "key": setting.config_key,
                "value": parse_setting_value(setting),
                "type": setting.data_type,
                "description": setting.description,
                "updated_at": setting.updated_at.isoformat()
            }
        }
    elif key in DEFAULT_SETTINGS:
        # Return default
        default = DEFAULT_SETTINGS[key]
        return {
            "success": True,
            "setting": {
                "key": key,
                "value": default["value"],
                "type": default["type"],
                "description": default["description"],
                "updated_at": None
            }
        }
    else:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")


@router.put("/{key}")
async def update_setting(
    key: str,
    update: SettingUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a specific setting"""
    # Validate the key matches
    if update.key != key:
        raise HTTPException(status_code=400, detail="Key mismatch")

    # Validate the setting exists in defaults
    if key not in DEFAULT_SETTINGS:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not recognized")

    # Validate value type
    expected_type = DEFAULT_SETTINGS[key]["type"]
    if update.data_type != expected_type:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data type. Expected '{expected_type}', got '{update.data_type}'"
        )

    # Additional validation for specific settings
    if key == "recognition_threshold":
        if not (0.0 <= float(update.value) <= 1.0):
            raise HTTPException(status_code=400, detail="Recognition threshold must be between 0.0 and 1.0")
    elif key == "detection_confidence":
        if not (0.0 <= float(update.value) <= 1.0):
            raise HTTPException(status_code=400, detail="Detection confidence must be between 0.0 and 1.0")
    elif key == "alert_cooldown_seconds":
        if int(update.value) < 0:
            raise HTTPException(status_code=400, detail="Alert cooldown must be non-negative")
    elif key == "max_faces_per_frame":
        if int(update.value) < 1 or int(update.value) > 50:
            raise HTTPException(status_code=400, detail="Max faces per frame must be between 1 and 50")
    elif key == "frame_skip":
        if int(update.value) < 1:
            raise HTTPException(status_code=400, detail="Frame skip must be at least 1")

    # Update in database
    setting = create_or_update_setting(
        db, key, update.value, update.data_type,
        update.description or DEFAULT_SETTINGS[key]["description"]
    )

    return {
        "success": True,
        "message": f"Setting '{key}' updated successfully",
        "setting": {
            "key": setting.config_key,
            "value": parse_setting_value(setting),
            "type": setting.data_type,
            "description": setting.description,
            "updated_at": setting.updated_at.isoformat()
        }
    }


@router.post("/bulk")
async def bulk_update_settings(
    updates: BulkSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update multiple settings at once"""
    updated_settings = []
    errors = []

    for key, value in updates.settings.items():
        if key not in DEFAULT_SETTINGS:
            errors.append(f"Unknown setting: {key}")
            continue

        try:
            expected_type = DEFAULT_SETTINGS[key]["type"]
            description = DEFAULT_SETTINGS[key]["description"]

            # Validate and update
            setting = create_or_update_setting(db, key, value, expected_type, description)
            updated_settings.append({
                "key": setting.config_key,
                "value": parse_setting_value(setting),
                "type": setting.data_type
            })
        except Exception as e:
            errors.append(f"Error updating {key}: {str(e)}")

    return {
        "success": len(errors) == 0,
        "updated_count": len(updated_settings),
        "updated_settings": updated_settings,
        "errors": errors if errors else None
    }


@router.post("/reset")
async def reset_to_defaults(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Reset all settings to default values"""
    reset_settings = []

    for key, config in DEFAULT_SETTINGS.items():
        setting = create_or_update_setting(
            db, key, config["value"], config["type"], config["description"]
        )
        reset_settings.append({
            "key": setting.config_key,
            "value": parse_setting_value(setting)
        })

    return {
        "success": True,
        "message": "All settings reset to defaults",
        "reset_count": len(reset_settings),
        "settings": reset_settings
    }


@router.post("/initialize")
async def initialize_defaults(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Initialize default settings if they don't exist.
    Does not overwrite existing settings.
    """
    initialized = []

    for key, config in DEFAULT_SETTINGS.items():
        existing = get_setting_from_db(db, key)
        if not existing:
            setting = create_or_update_setting(
                db, key, config["value"], config["type"], config["description"]
            )
            initialized.append(key)

    return {
        "success": True,
        "message": f"Initialized {len(initialized)} default settings",
        "initialized": initialized
    }


@router.get("/export/backup")
async def export_settings(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Export all settings as JSON backup"""
    settings = db.query(SystemConfiguration).all()

    backup = {
        "export_date": datetime.utcnow().isoformat(),
        "settings_count": len(settings),
        "settings": [
            {
                "key": s.config_key,
                "value": parse_setting_value(s),
                "type": s.data_type,
                "description": s.description
            }
            for s in settings
        ]
    }

    return backup


@router.post("/import/backup")
async def import_settings(
    backup: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Import settings from JSON backup"""
    if "settings" not in backup:
        raise HTTPException(status_code=400, detail="Invalid backup format")

    imported = []
    errors = []

    for setting_data in backup["settings"]:
        try:
            key = setting_data["key"]
            value = setting_data["value"]
            data_type = setting_data["type"]
            description = setting_data.get("description", "")

            create_or_update_setting(db, key, value, data_type, description)
            imported.append(key)
        except Exception as e:
            errors.append(f"Error importing {setting_data.get('key', 'unknown')}: {str(e)}")

    return {
        "success": len(errors) == 0,
        "imported_count": len(imported),
        "imported": imported,
        "errors": errors if errors else None
    }
