"""
Hardware binding utilities for Jetson devices.
Extracts hardware identifiers for license validation.
"""

import os
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HardwareIdentifier:
    """Extract and validate hardware identifiers from Jetson devices"""

    @staticmethod
    def get_jetson_serial() -> Optional[str]:
        """
        Get Jetson device serial number.

        Returns:
            Serial number or None if not found
        """
        serial_paths = [
            '/sys/firmware/devicetree/base/serial-number',
            '/proc/device-tree/serial-number',
            '/sys/devices/soc0/soc_uid'
        ]

        for path in serial_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        serial = f.read().strip()
                        # Remove null bytes
                        serial = serial.replace('\x00', '').strip()
                        if serial:
                            logger.info(f"Found Jetson serial: {serial[:8]}...")
                            return serial
            except Exception as e:
                logger.debug(f"Failed to read {path}: {e}")
                continue

        logger.warning("Could not find Jetson serial number")
        return None

    @staticmethod
    def get_jetson_model() -> Optional[str]:
        """
        Get Jetson device model name.

        Returns:
            Model name or None if not found
        """
        model_paths = [
            '/sys/firmware/devicetree/base/model',
            '/proc/device-tree/model'
        ]

        for path in model_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        model = f.read().strip()
                        model = model.replace('\x00', '').strip()
                        if model:
                            logger.info(f"Found Jetson model: {model}")
                            return model
            except Exception as e:
                logger.debug(f"Failed to read {path}: {e}")
                continue

        return None

    @staticmethod
    def get_mac_address() -> Optional[str]:
        """
        Get primary network interface MAC address.

        Returns:
            MAC address or None
        """
        try:
            # Try to get eth0 MAC address
            mac_path = '/sys/class/net/eth0/address'
            if os.path.exists(mac_path):
                with open(mac_path, 'r') as f:
                    mac = f.read().strip()
                    return mac

            # Try wlan0 as fallback
            mac_path = '/sys/class/net/wlan0/address'
            if os.path.exists(mac_path):
                with open(mac_path, 'r') as f:
                    mac = f.read().strip()
                    return mac
        except Exception as e:
            logger.debug(f"Failed to get MAC address: {e}")

        return None

    @staticmethod
    def generate_hardware_id() -> str:
        """
        Generate a unique hardware ID for this device.
        Combines serial number, model, and MAC address.

        Returns:
            SHA256 hash of hardware identifiers
        """
        serial = HardwareIdentifier.get_jetson_serial()
        model = HardwareIdentifier.get_jetson_model()
        mac = HardwareIdentifier.get_mac_address()

        # Combine all available identifiers
        hw_string = f"{serial or 'unknown'}-{model or 'unknown'}-{mac or 'unknown'}"

        # Generate SHA256 hash
        hw_id = hashlib.sha256(hw_string.encode()).hexdigest()

        logger.info(f"Generated hardware ID: {hw_id[:16]}...")
        return hw_id

    @staticmethod
    def validate_hardware_id(expected_id: str) -> bool:
        """
        Validate that current hardware matches expected ID.

        Args:
            expected_id: Expected hardware ID hash

        Returns:
            True if hardware matches, False otherwise
        """
        current_id = HardwareIdentifier.generate_hardware_id()
        matches = current_id == expected_id

        if matches:
            logger.info("✓ Hardware binding validation passed")
        else:
            logger.error("✗ Hardware binding validation failed")
            logger.debug(f"Expected: {expected_id[:16]}...")
            logger.debug(f"Current:  {current_id[:16]}...")

        return matches
