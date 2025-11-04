"""
License validation module for NRTC Face AI.
Validates license files with hardware binding and expiration checks.
"""

import os
import json
import hmac
import hashlib
from datetime import datetime
from typing import Dict, Optional
import logging

from .hardware import HardwareIdentifier
from ..utils.exceptions import (
    LicenseError,
    LicenseExpiredError,
    InvalidLicenseError,
    HardwareBindingError
)

logger = logging.getLogger(__name__)


class LicenseValidator:
    """Validates NRTC Face AI license"""

    # Secret key for license signature (in production, this would be more secure)
    LICENSE_SECRET = b'nrtc-faceai-2025-jetson-agx-orin-secure-key'

    def __init__(self, license_path: str = None):
        """
        Initialize license validator.

        Args:
            license_path: Path to license file (default: ./nrtc_license.json)
        """
        if license_path is None:
            # Look for license in multiple locations
            possible_paths = [
                os.path.join(os.getcwd(), 'nrtc_license.json'),
                os.path.join(os.path.expanduser('~'), '.nrtc', 'license.json'),
                '/etc/nrtc/license.json',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    license_path = path
                    break

        if not license_path or not os.path.exists(license_path):
            raise LicenseError(
                f"License file not found. Please contact NRTC for licensing. "
                f"Expected location: {license_path or possible_paths[0]}"
            )

        self.license_path = license_path
        self.license_data = None

    def load_license(self) -> Dict:
        """
        Load license file from disk.

        Returns:
            License data dictionary

        Raises:
            InvalidLicenseError: If license file is corrupted
        """
        try:
            with open(self.license_path, 'r') as f:
                self.license_data = json.load(f)

            logger.info(f"Loaded license from: {self.license_path}")
            return self.license_data

        except json.JSONDecodeError as e:
            raise InvalidLicenseError(f"License file is corrupted: {e}")
        except Exception as e:
            raise InvalidLicenseError(f"Failed to load license: {e}")

    def verify_signature(self) -> bool:
        """
        Verify license signature to prevent tampering.

        Returns:
            True if signature is valid

        Raises:
            InvalidLicenseError: If signature is invalid
        """
        if not self.license_data:
            self.load_license()

        signature = self.license_data.get('signature', '')
        license_content = {
            'licensee': self.license_data.get('licensee'),
            'organization': self.license_data.get('organization'),
            'hardware_id': self.license_data.get('hardware_id'),
            'issue_date': self.license_data.get('issue_date'),
            'expiry_date': self.license_data.get('expiry_date'),
            'license_type': self.license_data.get('license_type'),
            'features': self.license_data.get('features', []),
        }

        # Compute expected signature
        content_str = json.dumps(license_content, sort_keys=True)
        expected_signature = hmac.new(
            self.LICENSE_SECRET,
            content_str.encode(),
            hashlib.sha256
        ).hexdigest()

        if signature != expected_signature:
            raise InvalidLicenseError("License signature is invalid. License may have been tampered with.")

        logger.info("✓ License signature verified")
        return True

    def check_expiration(self) -> bool:
        """
        Check if license has expired.

        Returns:
            True if license is still valid

        Raises:
            LicenseExpiredError: If license has expired
        """
        if not self.license_data:
            self.load_license()

        expiry_str = self.license_data.get('expiry_date')
        if not expiry_str:
            raise InvalidLicenseError("License missing expiry date")

        try:
            expiry_date = datetime.fromisoformat(expiry_str)
            now = datetime.now()

            if now > expiry_date:
                raise LicenseExpiredError(
                    f"License expired on {expiry_date.strftime('%Y-%m-%d')}. "
                    f"Please contact NRTC to renew your license."
                )

            days_remaining = (expiry_date - now).days
            logger.info(f"✓ License valid ({days_remaining} days remaining)")

            if days_remaining < 30:
                logger.warning(f"⚠️  License expires soon ({days_remaining} days)!")

            return True

        except ValueError as e:
            raise InvalidLicenseError(f"Invalid expiry date format: {e}")

    def check_hardware_binding(self) -> bool:
        """
        Verify that license is bound to this hardware.

        Returns:
            True if hardware matches

        Raises:
            HardwareBindingError: If hardware doesn't match
        """
        if not self.license_data:
            self.load_license()

        expected_hw_id = self.license_data.get('hardware_id')
        if not expected_hw_id:
            raise InvalidLicenseError("License missing hardware ID")

        # Allow wildcard for development licenses
        if expected_hw_id == '*':
            logger.warning("⚠️  License allows any hardware (development mode)")
            return True

        current_hw_id = HardwareIdentifier.generate_hardware_id()

        if current_hw_id != expected_hw_id:
            raise HardwareBindingError(
                f"License is not valid for this device. "
                f"This license is bound to a different Jetson device. "
                f"Current device ID: {current_hw_id[:16]}..."
            )

        logger.info("✓ Hardware binding verified")
        return True

    def check_feature(self, feature: str) -> bool:
        """
        Check if a feature is enabled in the license.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is enabled
        """
        if not self.license_data:
            self.load_license()

        features = self.license_data.get('features', [])

        # Allow wildcard for full licenses
        if '*' in features or 'all' in features:
            return True

        return feature in features

    def validate(self) -> Dict:
        """
        Perform full license validation.

        Returns:
            License data if valid

        Raises:
            LicenseError: If validation fails
        """
        logger.info("=" * 60)
        logger.info("NRTC Face AI - License Validation")
        logger.info("=" * 60)

        # Load license
        self.load_license()

        # Verify signature
        self.verify_signature()

        # Check expiration
        self.check_expiration()

        # Check hardware binding
        self.check_hardware_binding()

        # Log license info
        logger.info(f"Licensee: {self.license_data.get('licensee')}")
        logger.info(f"Organization: {self.license_data.get('organization')}")
        logger.info(f"License Type: {self.license_data.get('license_type')}")
        logger.info(f"Features: {', '.join(self.license_data.get('features', []))}")
        logger.info("=" * 60)
        logger.info("✓ License validation successful")
        logger.info("=" * 60)

        return self.license_data

    @staticmethod
    def generate_license(licensee: str, organization: str, hardware_id: str,
                        expiry_date: str, license_type: str = 'commercial',
                        features: list = None) -> Dict:
        """
        Generate a new license file (for internal NRTC use only).

        Args:
            licensee: License holder name
            organization: Organization name
            hardware_id: Hardware ID or '*' for any device
            expiry_date: Expiry date (ISO format: YYYY-MM-DD)
            license_type: License type (commercial, development, trial)
            features: List of enabled features or ['*'] for all

        Returns:
            License dictionary
        """
        if features is None:
            features = ['*']  # All features by default

        issue_date = datetime.now().isoformat()

        license_content = {
            'licensee': licensee,
            'organization': organization,
            'hardware_id': hardware_id,
            'issue_date': issue_date,
            'expiry_date': expiry_date,
            'license_type': license_type,
            'features': features,
        }

        # Generate signature
        content_str = json.dumps(license_content, sort_keys=True)
        signature = hmac.new(
            LicenseValidator.LICENSE_SECRET,
            content_str.encode(),
            hashlib.sha256
        ).hexdigest()

        license_content['signature'] = signature

        return license_content

    @staticmethod
    def save_license(license_data: Dict, output_path: str):
        """
        Save license to file.

        Args:
            license_data: License dictionary
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(license_data, f, indent=2)

        logger.info(f"License saved to: {output_path}")
