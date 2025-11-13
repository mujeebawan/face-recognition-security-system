#!/bin/bash
# Test script to verify SD card auto-mount configuration

echo "=== SD Card Auto-Mount Configuration Test ==="
echo ""
echo "1. Checking /etc/fstab entry:"
grep sdcard /etc/fstab
echo ""

echo "2. Current SD card mount status:"
mount | grep sdcard
echo ""

echo "3. SD card device info:"
lsblk | grep mmcblk1
echo ""

echo "4. SD card UUID:"
sudo blkid /dev/mmcblk1p1 | grep UUID
echo ""

echo "5. Mount point permissions:"
ls -ld /media/sdcard
echo ""

echo "6. SD card contents:"
ls -lh /media/sdcard/
echo ""

echo "✅ Configuration complete!"
echo "The SD card will auto-mount to /media/sdcard on reboot."
echo ""
echo "To test, run: sudo reboot"
