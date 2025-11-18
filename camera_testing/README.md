# Camera Testing Tools

Standalone tools for testing camera streams without running the main server.

## 🚀 Quick Start

### GUI Version (Full Features)
```bash
python3 test_camera_stream.py
```

Features:
- Stream selection (main/sub)
- Frame skipping slider
- PTZ zoom controls
- Real-time statistics

### CLI Version (Quick Test)
```bash
# Test main stream
python3 test_camera_simple.py

# Test sub stream with frame skip
python3 test_camera_simple.py --stream sub --skip 2
```

## 📁 Files

- **test_camera_stream.py** - GUI testing tool (recommended)
- **test_camera_simple.py** - CLI testing tool (lightweight)
- **CAMERA_TESTING.md** - Full documentation

## 📖 Documentation

See [CAMERA_TESTING.md](CAMERA_TESTING.md) for complete documentation, examples, and troubleshooting.
