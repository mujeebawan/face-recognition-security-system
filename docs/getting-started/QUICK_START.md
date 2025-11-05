# Quick Start Guide

## 🚀 Starting the Server

```bash
./start_server.sh
```

The script will:
- ✅ Automatically stop any running instances
- ✅ Start fresh server on port 8000
- ✅ Load AI models (takes ~10 seconds)
- ✅ Show you the access URLs with your IP

**Output example:**
```
✅ Server is running successfully!

📍 Access the system:
   • Admin Panel:    http://192.168.0.117:8000/admin
   • Live Stream:    http://192.168.0.117:8000/live
   • Dashboard:      http://192.168.0.117:8000/dashboard
   • API Docs:       http://192.168.0.117:8000/docs
```

---

## 🛑 Stopping the Server

```bash
./stop_server.sh
```

**To stop and clean logs:**
```bash
./stop_server.sh --clean
```

---

## 📋 Useful Commands

| Command | Description |
|---------|-------------|
| `./start_server.sh` | Start the server |
| `./stop_server.sh` | Stop the server |
| `./stop_server.sh --clean` | Stop server + delete logs |
| `tail -f server.log` | Watch server logs in real-time |
| `pgrep -f uvicorn` | Check if server is running |

---

## 🎯 What You Can Do

### 1. Admin Panel (`/admin`)
- Enroll new persons with multi-angle camera capture
- View smooth camera preview (15-20 FPS)
- Manage enrolled persons database
- Configure alert thresholds

### 2. Live Stream (`/live`)
- Real-time face detection (green boxes)
- Face recognition with names
- Live alert notifications

### 3. Dashboard (`/dashboard`)
- Recognition event logs
- Alert history
- System statistics

### 4. API Documentation (`/docs`)
- Interactive API explorer
- Test endpoints directly
- See request/response schemas

---

## 🔧 Troubleshooting

### Server won't start?
```bash
# Check the logs
tail -50 server.log

# Force clean restart
./stop_server.sh --clean
./start_server.sh
```

### Port already in use?
```bash
# The start script automatically handles this
# But if needed, manually kill:
pkill -9 -f "uvicorn.*app.main:app"
```

### Can't access from browser?
- Make sure you're on the same network (192.168.0.x)
- Check firewall: `sudo ufw status`
- Verify server is running: `pgrep -f uvicorn`

---

## 📊 System Requirements

- **Hardware**: NVIDIA Jetson AGX Orin (64GB)
- **JetPack**: 6.1
- **Python**: 3.10+
- **GPU Memory**: ~8GB during operation
- **Network**: Local network access for camera

---

## ⚡ Performance

- **Face Detection**: 15-20 FPS (SCRFD + TensorRT)
- **Recognition**: ~300ms per face (ArcFace)
- **Accuracy**: 95%+ recognition accuracy
- **Camera Stream**: Smooth MJPEG (no frame skipping)

---

**Need more details?** See [README.md](README.md) for complete documentation.
