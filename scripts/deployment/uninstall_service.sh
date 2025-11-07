#!/bin/bash
# Uninstall Face Recognition System systemd service

set -e

echo "================================================"
echo "Face Recognition System - Service Uninstallation"
echo "================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Stop service if running
echo "Stopping service..."
systemctl stop face-recognition.service 2>/dev/null || true

# Disable service
echo "Disabling service..."
systemctl disable face-recognition.service 2>/dev/null || true

# Remove service file
echo "Removing service file..."
rm -f /etc/systemd/system/face-recognition.service

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload
systemctl reset-failed

echo ""
echo "✅ Service uninstalled successfully!"
echo ""
echo "Note: Log files in /var/log/face-recognition/ were not deleted."
echo "To remove logs, run: sudo rm -rf /var/log/face-recognition/"
echo ""
