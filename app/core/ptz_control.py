"""
PTZ Control Module for Hikvision Camera
Handles Pan, Tilt, and Zoom operations via ISAPI
"""

import requests
from requests.auth import HTTPDigestAuth
import logging
from typing import Optional, Tuple
import time

from app.config import settings

logger = logging.getLogger(__name__)


class PTZController:
    """
    PTZ Controller for Hikvision IP Camera
    Supports zoom, pan, tilt, presets, and patrols
    """

    def __init__(self, camera_ip: str = None, username: str = None, password: str = None):
        """
        Initialize PTZ Controller

        Args:
            camera_ip: Camera IP address (default from settings)
            username: Camera username (default from settings)
            password: Camera password (default from settings)
        """
        self.camera_ip = camera_ip or settings.camera_ip
        self.username = username or settings.camera_username
        self.password = password or settings.camera_password
        self.channel = 1  # Default camera channel
        self.base_url = f"http://{self.camera_ip}/ISAPI/PTZCtrl/channels/{self.channel}"

        # Track current zoom level (0-100%)
        self.current_zoom_level = 0  # Start at 0%

    def _send_ptz_command(self, command: str, params: dict) -> Tuple[bool, str]:
        """
        Send PTZ command to camera

        Args:
            command: PTZ command (continuous, momentary, etc.)
            params: Command parameters

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            url = f"{self.base_url}/{command}"

            # Build XML payload
            xml_payload = '<?xml version="1.0" encoding="UTF-8"?>'
            xml_payload += '<PTZData>'

            for key, value in params.items():
                xml_payload += f'<{key}>{value}</{key}>'

            xml_payload += '</PTZData>'

            response = requests.put(
                url,
                data=xml_payload,
                auth=HTTPDigestAuth(self.username, self.password),
                headers={'Content-Type': 'application/xml'},
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"PTZ command '{command}' executed successfully")
                return True, "Command executed successfully"
            else:
                error_msg = f"PTZ command failed: HTTP {response.status_code}"
                logger.error(error_msg)
                return False, error_msg

        except requests.exceptions.Timeout:
            error_msg = "PTZ command timeout"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"PTZ command error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def zoom_in(self, speed: int = 50, duration: float = 0.1) -> Tuple[bool, str]:
        """
        Zoom in (continuous)

        Args:
            speed: Zoom speed (1-100)
            duration: How long to zoom (used for percentage calculation)

        Returns:
            Tuple of (success: bool, message: str)
        """
        speed = max(1, min(100, speed))  # Clamp between 1-100
        params = {'zoom': speed}

        # Increment zoom level (approximate based on speed)
        # Higher speed = faster zoom = more percentage increase
        increment = (speed / 100) * 5  # Max 5% per command at full speed
        self.current_zoom_level = min(100, self.current_zoom_level + increment)

        return self._send_ptz_command('continuous', params)

    def zoom_out(self, speed: int = 50, duration: float = 0.1) -> Tuple[bool, str]:
        """
        Zoom out (continuous)

        Args:
            speed: Zoom speed (1-100)
            duration: How long to zoom (used for percentage calculation)

        Returns:
            Tuple of (success: bool, message: str)
        """
        speed = max(1, min(100, speed))  # Clamp between 1-100
        params = {'zoom': -speed}  # Negative for zoom out

        # Decrement zoom level (approximate based on speed)
        decrement = (speed / 100) * 5  # Max 5% per command at full speed
        self.current_zoom_level = max(0, self.current_zoom_level - decrement)

        return self._send_ptz_command('continuous', params)

    def zoom_stop(self) -> Tuple[bool, str]:
        """
        Stop zoom movement

        Returns:
            Tuple of (success: bool, message: str)
        """
        params = {'zoom': 0}
        return self._send_ptz_command('continuous', params)

    def pan_left(self, speed: int = 50) -> Tuple[bool, str]:
        """Pan camera left"""
        speed = max(1, min(100, speed))
        params = {'pan': -speed}
        return self._send_ptz_command('continuous', params)

    def pan_right(self, speed: int = 50) -> Tuple[bool, str]:
        """Pan camera right"""
        speed = max(1, min(100, speed))
        params = {'pan': speed}
        return self._send_ptz_command('continuous', params)

    def tilt_up(self, speed: int = 50) -> Tuple[bool, str]:
        """Tilt camera up"""
        speed = max(1, min(100, speed))
        params = {'tilt': speed}
        return self._send_ptz_command('continuous', params)

    def tilt_down(self, speed: int = 50) -> Tuple[bool, str]:
        """Tilt camera down"""
        speed = max(1, min(100, speed))
        params = {'tilt': -speed}
        return self._send_ptz_command('continuous', params)

    def stop_all(self) -> Tuple[bool, str]:
        """Stop all PTZ movements"""
        params = {'pan': 0, 'tilt': 0, 'zoom': 0}
        return self._send_ptz_command('continuous', params)

    def goto_preset(self, preset_id: int) -> Tuple[bool, str]:
        """
        Go to a saved preset position

        Args:
            preset_id: Preset ID (1-255)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not 1 <= preset_id <= 255:
            return False, "Preset ID must be between 1 and 255"

        try:
            url = f"{self.base_url}/presets/{preset_id}/goto"

            response = requests.put(
                url,
                auth=HTTPDigestAuth(self.username, self.password),
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"Moved to preset {preset_id}")
                return True, f"Moved to preset {preset_id}"
            else:
                error_msg = f"Failed to goto preset: HTTP {response.status_code}"
                logger.error(error_msg)
                return False, error_msg

        except Exception as e:
            error_msg = f"Preset goto error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def save_preset(self, preset_id: int, name: str = "") -> Tuple[bool, str]:
        """
        Save current position as preset

        Args:
            preset_id: Preset ID (1-255)
            name: Optional preset name

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not 1 <= preset_id <= 255:
            return False, "Preset ID must be between 1 and 255"

        try:
            url = f"{self.base_url}/presets/{preset_id}"

            xml_payload = '<?xml version="1.0" encoding="UTF-8"?>'
            xml_payload += '<PTZPreset>'
            xml_payload += f'<id>{preset_id}</id>'
            if name:
                xml_payload += f'<presetName>{name}</presetName>'
            xml_payload += '</PTZPreset>'

            response = requests.put(
                url,
                data=xml_payload,
                auth=HTTPDigestAuth(self.username, self.password),
                headers={'Content-Type': 'application/xml'},
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"Saved preset {preset_id}: {name}")
                return True, f"Preset {preset_id} saved"
            else:
                error_msg = f"Failed to save preset: HTTP {response.status_code}"
                logger.error(error_msg)
                return False, error_msg

        except Exception as e:
            error_msg = f"Preset save error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def get_zoom_level(self) -> int:
        """
        Get current zoom level

        Returns:
            Zoom level as percentage (0-100)
        """
        return int(self.current_zoom_level)

    def set_zoom_level(self, level: int):
        """
        Set current zoom level (for manual override or reset)

        Args:
            level: Zoom level percentage (0-100)
        """
        self.current_zoom_level = max(0, min(100, level))
        logger.info(f"Zoom level manually set to {self.current_zoom_level}%")

    def get_ptz_status(self) -> Optional[dict]:
        """
        Get current PTZ status

        Returns:
            Dict with PTZ status or None on error
        """
        try:
            url = f"{self.base_url}/status"

            response = requests.get(
                url,
                auth=HTTPDigestAuth(self.username, self.password),
                timeout=5
            )

            if response.status_code == 200:
                # Parse XML response (simplified)
                return {
                    "success": True,
                    "status": "online",
                    "zoom_level": self.get_zoom_level()
                }
            else:
                logger.error(f"Failed to get PTZ status: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"PTZ status error: {str(e)}")
            return None


# Global PTZ controller instance
_ptz_controller = None


def get_ptz_controller() -> PTZController:
    """Get or create PTZ controller singleton"""
    global _ptz_controller
    if _ptz_controller is None:
        _ptz_controller = PTZController()
    return _ptz_controller
