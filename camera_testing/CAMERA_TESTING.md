# Camera Stream Testing Tools

Two standalone tools for testing the Hikvision camera stream without running the full server.

## 📋 Scripts

### 1. `test_camera_stream.py` - **GUI Version (Recommended)**

Full-featured GUI application with:
- ✅ **Stream selection** - Choose main stream (high quality) or sub stream (lower quality)
- ✅ **Frame skipping** - Actually skips frames (0-10) to reduce processing load
- ✅ **PTZ zoom controls** - Zoom in/out with adjustable speed
- ✅ **Real-time statistics** - FPS, processed frames, skipped frames
- ✅ **Keyboard shortcuts** - `+` to zoom in, `-` to zoom out

**Usage:**
```bash
# Run with GUI
python3 test_camera_stream.py

# Or make executable and run
chmod +x test_camera_stream.py
./test_camera_stream.py
```

**Interface:**
- **Left Panel**: Controls for stream selection, frame skipping, and zoom
- **Right Panel**: Live video display
- **Stream Selection**: Choose between main (1920x1080) or sub (640x480) stream
- **Frame Skipping**:
  - 0 = Process all frames
  - 1 = Skip 1 frame, process every 2nd
  - 2 = Skip 2 frames, process every 3rd
  - etc.
- **Zoom Control**:
  - Adjust speed slider (10-100)
  - Hold zoom buttons to continuously zoom
  - Or use keyboard: `+` (zoom in), `-` (zoom out)

### 2. `test_camera_simple.py` - **CLI Version (Lightweight)**

Simple command-line tool for quick testing.

**Usage:**
```bash
# Test main stream for 30 seconds
python3 test_camera_simple.py

# Test sub stream with frame skipping
python3 test_camera_simple.py --stream sub --skip 2

# Test for 60 seconds
python3 test_camera_simple.py --duration 60

# Full options
python3 test_camera_simple.py --stream sub --skip 1 --duration 120
```

**Options:**
- `--stream {main,sub}` - Stream to test (default: main)
- `--skip N` - Skip N frames (default: 0)
- `--duration N` - Test duration in seconds (default: 30)

**During test:**
- Press `Q` to quit early
- Real-time FPS and frame statistics displayed

## 🎯 Use Cases

### Testing Stream Quality
```bash
# Compare main vs sub stream quality
python3 test_camera_simple.py --stream main --duration 10
python3 test_camera_simple.py --stream sub --duration 10
```

### Testing Frame Skipping Performance
```bash
# Test different skip values to find optimal performance
python3 test_camera_simple.py --skip 0  # All frames
python3 test_camera_simple.py --skip 1  # Every 2nd frame
python3 test_camera_simple.py --skip 2  # Every 3rd frame
```

### Testing PTZ Zoom
```bash
# Use GUI version
python3 test_camera_stream.py

# Then use zoom controls in the interface
```

## 📊 Understanding Frame Skipping

Frame skipping reduces CPU/GPU load by processing fewer frames:

| Skip Value | Frames Processed | Example Use Case |
|------------|------------------|------------------|
| 0 | All frames | Full quality, high CPU |
| 1 | Every 2nd frame | 50% load reduction |
| 2 | Every 3rd frame | 66% load reduction |
| 3 | Every 4th frame | 75% load reduction |

**When to use:**
- **Skip 0**: Maximum quality, real-time detection needed
- **Skip 1-2**: Balanced performance/quality
- **Skip 3+**: Low power mode, reduced detection accuracy

## 🔧 Camera Configuration

Both scripts are pre-configured for your camera:
- **IP**: 192.168.1.64
- **Username**: admin
- **Main Stream**: `rtsp://admin:****@192.168.1.64:554/Streaming/Channels/101`
- **Sub Stream**: `rtsp://admin:****@192.168.1.64:554/Streaming/Channels/102`

To change camera settings, edit the scripts directly:
```python
# In test_camera_stream.py or test_camera_simple.py
CAMERA_IP = "192.168.1.64"
USERNAME = "admin"
PASSWORD = "your_password"
```

## 🚀 Quick Start

**For GUI testing (recommended):**
```bash
python3 test_camera_stream.py
```

**For quick CLI test:**
```bash
# Test main stream
python3 test_camera_simple.py

# Test sub stream with frame skip
python3 test_camera_simple.py --stream sub --skip 2
```

## 🐛 Troubleshooting

### "Failed to connect to camera"
1. Check camera is powered on
2. Verify network connectivity: `ping 192.168.1.64`
3. Test RTSP manually:
   ```bash
   ffmpeg -i "rtsp://admin:Mujeeb@321@192.168.1.64:554/Streaming/Channels/101" -frames:v 1 test.jpg
   ```

### Low FPS
1. Try sub stream instead of main stream
2. Increase frame skip value
3. Check network bandwidth
4. Close other applications using GPU

### Zoom doesn't work
1. Verify camera supports PTZ (not all cameras do)
2. Check camera IP is correct
3. Verify username/password
4. Test ISAPI endpoint:
   ```bash
   curl -u admin:Mujeeb@321 "http://192.168.1.64/ISAPI/PTZCtrl/channels/1/status"
   ```

### GUI doesn't display
1. Make sure X11 is available (if SSH, use `ssh -X`)
2. Try the CLI version instead: `test_camera_simple.py`
3. Check tkinter is installed: `python3 -c "import tkinter"`

## 📝 Examples

**Test main stream quality:**
```bash
python3 test_camera_simple.py --stream main --duration 60
```

**Test sub stream with aggressive frame skipping:**
```bash
python3 test_camera_simple.py --stream sub --skip 3 --duration 120
```

**Interactive GUI testing with zoom:**
```bash
python3 test_camera_stream.py
# Then:
# 1. Select stream type
# 2. Adjust frame skip slider
# 3. Click "Start Stream"
# 4. Use zoom controls to test PTZ
```

## 🎨 GUI Features

### Stream Selection
- **Main Stream**: 1920x1080 (or camera's max resolution)
  - Better quality
  - Higher bandwidth
  - More CPU/GPU load
- **Sub Stream**: 640x480 (typical)
  - Lower quality
  - Lower bandwidth
  - Less CPU/GPU load

### Frame Skipping (Actually Works!)
The GUI slider (0-10) controls how many frames are **actually skipped**:
- Value shows in real-time
- Statistics show processed vs skipped frames
- You can adjust while stream is running (restart stream to apply)

### Zoom Controls
- **Speed slider**: 10-100 (higher = faster zoom)
- **Zoom In/Out buttons**: Hold to continuously zoom
- **Keyboard shortcuts**:
  - `+` or `=` to zoom in
  - `-` or `_` to zoom out
  - Release to stop
- Status shows current zoom action

### Statistics Panel
- **FPS**: Current frames per second
- **Frames**: Total frames processed
- **Skipped**: Total frames skipped

## 💡 Tips

1. **Start with sub stream** - Lower load, faster feedback
2. **Test frame skipping** - Find optimal balance for your use case
3. **Monitor FPS** - Should be 15-25 FPS for smooth viewing
4. **Use CLI for benchmarking** - Simpler, more consistent results
5. **Use GUI for interactive testing** - Better for PTZ and visual testing
