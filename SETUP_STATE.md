# Setup State Reference (for Claude context)

## SD Card Setup ✅
- Device: `/dev/mmcblk1p1` mounted at `/media/sdcard` (ExFAT, 64GB)
- Mount: `sudo mount.exfat-fuse /dev/mmcblk1p1 /media/sdcard`
- Data directory: `/media/sdcard/data/` (185MB - face images and embeddings)
- Space available: **52GB free**

## Storage Strategy ✅ IMPLEMENTED
- **SD Card** (`/media/sdcard/huggingface_cache/`):
  - Stable Diffusion 1.5 (~4GB) - **lazy loaded**
  - ControlNet Depth (~1.5GB) - **lazy loaded**
  - IP-Adapter (~3.7GB) - **lazy loaded**
  - MiDaS Annotators (~200MB) - **lazy loaded**

- **Jetson** (`/home/mujeeb/Downloads/LivePortrait/pretrained_weights/`):
  - LivePortrait models (2GB already downloaded ✓)
  - InsightFace models (ArcFace, SCRFD) - in code directory
  - Application code

## Lazy Loading Implementation ✅
**Models download automatically on first use!**

### How It Works:
1. User enrolls a person via web UI
2. If SD/ControlNet augmentation is enabled:
   - System checks `/media/sdcard/huggingface_cache/` for models
   - If not found, downloads with **real-time progress** shown in logs
   - Models are cached for future use
   - Only downloads what's needed (no upfront bulk downloads)

### Progress Tracking:
- Download progress logged every 2 seconds
- Shows: `Downloaded MB / Total MB (percentage%) - Speed MB/s`
- Example: `📥 Stable Diffusion 1.5: 1234.5MB / 4096.0MB (30.1%) - 15.23 MB/s`

### Modified Files:
- `app/core/resource_manager.py` - Added `download_model_with_progress()` helper
- `app/core/controlnet_augmentation.py` - SD card paths + lazy download
- `app/core/generative_augmentation.py` - SD card paths + lazy download

## Current Status
- ✅ SD card mounted and clean (52GB free)
- ✅ LivePortrait 2GB on Jetson (complete)
- ✅ Lazy loading implemented with progress tracking
- ✅ No manual downloads needed - automatic on first enrollment!

## Testing
To test lazy loading:
```bash
# 1. Ensure SD card is mounted
mount | grep sdcard

# 2. Start server (watch logs for download progress)
cd ~/Downloads/face-recognition-security-system
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Enroll person with SD augmentation enabled
# → Models will auto-download with progress shown in terminal

# 4. Check downloaded models
du -sh /media/sdcard/huggingface_cache/models--*
```

## Manual Check/Resume (if needed)
```bash
# Check what's downloaded
du -sh /media/sdcard/huggingface_cache/models--*

# Check SD card space
df -h /media/sdcard

# Verify data directory intact
ls -lh /media/sdcard/data/
```
