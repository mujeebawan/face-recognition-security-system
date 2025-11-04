#!/usr/bin/env python3
"""
NRTC Face AI - License Generation Tool
For internal NRTC use only to generate customer licenses.
"""

import sys
import argparse
from datetime import datetime, timedelta
from nrtc_faceai.license import LicenseValidator, HardwareIdentifier


def generate_license_interactive():
    """Interactive license generation"""
    print("=" * 70)
    print("NRTC Face AI - License Generation Tool")
    print("=" * 70)
    print()

    # Get customer information
    licensee = input("Licensee Name: ").strip()
    organization = input("Organization: ").strip()

    # Hardware ID options
    print("\nHardware ID Options:")
    print("1. Current Device (Auto-detect)")
    print("2. Specific Device (Enter Hardware ID)")
    print("3. Any Device (Development License - '*')")

    hw_choice = input("\nSelect option (1-3): ").strip()

    if hw_choice == "1":
        hardware_id = HardwareIdentifier.generate_hardware_id()
        print(f"\nDetected Hardware ID: {hardware_id[:16]}...")
    elif hw_choice == "2":
        hardware_id = input("Enter Hardware ID: ").strip()
    elif hw_choice == "3":
        hardware_id = "*"
        print("\n⚠️  WARNING: This license will work on ANY device (development only)")
    else:
        print("Invalid choice")
        return

    # License type
    print("\nLicense Types:")
    print("1. Commercial")
    print("2. Development")
    print("3. Trial")

    type_choice = input("\nSelect type (1-3): ").strip()
    license_types = {"1": "commercial", "2": "development", "3": "trial"}
    license_type = license_types.get(type_choice, "commercial")

    # Expiry date
    print("\nExpiry Date:")
    print("1. 1 Year")
    print("2. 6 Months")
    print("3. 3 Months")
    print("4. 1 Month (Trial)")
    print("5. Custom")

    expiry_choice = input("\nSelect expiry (1-5): ").strip()

    expiry_options = {
        "1": 365,
        "2": 180,
        "3": 90,
        "4": 30
    }

    if expiry_choice in expiry_options:
        days = expiry_options[expiry_choice]
        expiry_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    elif expiry_choice == "5":
        expiry_date = input("Enter expiry date (YYYY-MM-DD): ").strip()
    else:
        expiry_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")

    # Features
    print("\nFeatures:")
    print("1. All Features (*)")
    print("2. Select Specific Features")

    feature_choice = input("\nSelect option (1-2): ").strip()

    if feature_choice == "2":
        features = []
        print("\nAvailable features:")
        print("- face_detection")
        print("- face_recognition")
        print("- augmentation")
        print("\nEnter features (comma-separated):")
        features_input = input().strip()
        features = [f.strip() for f in features_input.split(",") if f.strip()]
    else:
        features = ["*"]

    # Generate license
    print("\n" + "=" * 70)
    print("Generating License...")
    print("=" * 70)

    license_data = LicenseValidator.generate_license(
        licensee=licensee,
        organization=organization,
        hardware_id=hardware_id,
        expiry_date=expiry_date,
        license_type=license_type,
        features=features
    )

    # Save license
    output_file = f"nrtc_license_{licensee.replace(' ', '_').lower()}.json"
    LicenseValidator.save_license(license_data, output_file)

    print(f"\n✓ License generated successfully!")
    print(f"  File: {output_file}")
    print(f"  Licensee: {licensee}")
    print(f"  Organization: {organization}")
    print(f"  Hardware ID: {hardware_id[:16] if hardware_id != '*' else '*'}...")
    print(f"  License Type: {license_type}")
    print(f"  Expiry Date: {expiry_date}")
    print(f"  Features: {', '.join(features)}")
    print("\n" + "=" * 70)


def get_hardware_id():
    """Display current device hardware ID"""
    print("=" * 70)
    print("NRTC Face AI - Hardware ID Detection")
    print("=" * 70)
    print()

    hw_id = HardwareIdentifier.generate_hardware_id()
    serial = HardwareIdentifier.get_jetson_serial()
    model = HardwareIdentifier.get_jetson_model()
    mac = HardwareIdentifier.get_mac_address()

    print(f"Jetson Serial:  {serial or 'Not found'}")
    print(f"Jetson Model:   {model or 'Not found'}")
    print(f"MAC Address:    {mac or 'Not found'}")
    print()
    print(f"Hardware ID:    {hw_id}")
    print()
    print("Use this Hardware ID when requesting a license from NRTC.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="NRTC Face AI License Generation Tool"
    )

    parser.add_argument(
        "--get-hardware-id",
        action="store_true",
        help="Display current device hardware ID"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new license (interactive mode)"
    )

    args = parser.parse_args()

    if args.get_hardware_id:
        get_hardware_id()
    elif args.generate:
        generate_license_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
