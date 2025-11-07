#!/bin/bash
# Install Face Recognition System as a systemd service

set -e

echo "================================================"
echo "Face Recognition System - Service Installation"
echo "================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Create log directory
echo "Creating log directory..."
mkdir -p /var/log/face-recognition
chown mujeeb:mujeeb /var/log/face-recognition

# Copy service file
echo "Installing systemd service file..."
cp face-recognition.service /etc/systemd/system/

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service
echo "Enabling service to start on boot..."
systemctl enable face-recognition.service

echo ""
echo "✅ Service installed successfully!"
echo ""
echo "Available commands:"
echo "  sudo systemctl start face-recognition    # Start the service"
echo "  sudo systemctl stop face-recognition     # Stop the service"
echo "  sudo systemctl restart face-recognition  # Restart the service"
echo "  sudo systemctl status face-recognition   # Check service status"
echo "  sudo journalctl -u face-recognition -f  # View live logs"
echo ""
echo "To start the service now, run:"
echo "  sudo systemctl start face-recognition"
echo ""
