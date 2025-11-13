#!/bin/bash
# Mount SD card script
sudo mkdir -p /media/sdcard
sudo mount /dev/mmcblk1p1 /media/sdcard
sudo chown -R mujeeb:mujeeb /media/sdcard
df -h /media/sdcard
