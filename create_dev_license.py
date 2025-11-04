#!/usr/bin/env python3
"""
Quick script to create a development license for this Jetson device
"""

import sys
import os
from datetime import datetime, timedelta

# Add nrtc_faceai to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nrtc_faceai'))

from nrtc_faceai.license import LicenseValidator, HardwareIdentifier

def create_dev_license():
    """Create development license for current device"""
    print("=" * 70)
    print("Creating Development License for Current Jetson Device")
    print("=" * 70)
    print()

    # Get hardware ID
    hw_id = HardwareIdentifier.generate_hardware_id()
    serial = HardwareIdentifier.get_jetson_serial()
    model = HardwareIdentifier.get_jetson_model()

    print(f"Jetson Serial:  {serial or 'Not found'}")
    print(f"Jetson Model:   {model or 'Not found'}")
    print(f"Hardware ID:    {hw_id[:16]}...")
    print()

    # License details
    licensee = "Development"
    organization = "NRTC"
    # Use '*' for development to work on any device
    hardware_id = "*"  # Wildcard - works on any device
    expiry_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    license_type = "development"
    features = ["*"]  # All features

    print("Generating license...")
    print(f"  Licensee:      {licensee}")
    print(f"  Organization:  {organization}")
    print(f"  Hardware ID:   {hardware_id} (wildcard - any device)")
    print(f"  Expiry Date:   {expiry_date}")
    print(f"  License Type:  {license_type}")
    print(f"  Features:      {', '.join(features)}")
    print()

    # Generate license
    license_data = LicenseValidator.generate_license(
        licensee=licensee,
        organization=organization,
        hardware_id=hardware_id,
        expiry_date=expiry_date,
        license_type=license_type,
        features=features
    )

    # Save in current directory
    output_file = "nrtc_license.json"
    LicenseValidator.save_license(license_data, output_file)

    print(f"✓ Development license created: {output_file}")
    print()
    print("This license will work on any Jetson device for 1 year.")
    print("For production use, generate a hardware-bound license.")
    print("=" * 70)


if __name__ == "__main__":
    create_dev_license()
