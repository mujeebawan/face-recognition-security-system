# Server Management Scripts

This directory contains scripts for managing the Face Recognition Security System server.

## Available Scripts

### `./start_server.sh`
Starts the FastAPI server with proper initialization and health checking.

**Features:**
- Automatically kills any existing server processes
- Creates logs directory if needed
- Waits up to 30 seconds for server to fully initialize
- Shows progress updates every 5 seconds
- Validates server is bound to port 8000 before reporting success
- Saves PID to `logs/server.pid` for easy management
- Displays access URLs with local IP address

**Usage:**
```bash
./start_server.sh
```

**Startup Time:**
The server typically takes 15-20 seconds to fully start due to loading ML models (InsightFace, TensorRT engines, etc.).

### `./stop_server.sh`
Gracefully stops the server and optionally cleans up log files.

**Features:**
- Uses saved PID file for clean shutdown
- Attempts graceful shutdown (SIGTERM) first
- Falls back to force kill (SIGKILL) if needed
- Cleans up any orphaned processes
- Removes PID file after shutdown

**Usage:**
```bash
# Stop server
./stop_server.sh

# Stop and clean up logs
./stop_server.sh --clean
```

### `./status_server.sh`
Displays current server status, resource usage, and recent logs.

**Features:**
- Shows PID and process status
- Displays memory and CPU usage
- Checks if port 8000 is bound
- Shows access URLs
- Displays last 5 log lines
- Detects stale PID files

**Usage:**
```bash
./status_server.sh
```

**Example Output:**
```
🔍 Face Recognition System - Server Status

✅ Server is running
   • PID: 6853
   • Memory/CPU:  7.9 22.0
   • Port 8000: Bound ✓

📍 Access URLs:
   • Admin Panel:    http://192.168.0.117:8000/admin
   • Live Stream:    http://192.168.0.117:8000/live
   • Dashboard:      http://192.168.0.117:8000/dashboard

📄 Recent logs (last 5 lines):
   [last 5 log lines...]
```

## Directory Structure

```
face-recognition-security-system/
├── start_server.sh       # Start script
├── stop_server.sh        # Stop script
├── status_server.sh      # Status checker
├── logs/
│   ├── server.log        # Application logs
│   └── server.pid        # Current server PID
└── scripts/
    └── deployment/       # Systemd service files
```

## Log Files

### `logs/server.log`
Main application log file containing:
- Server startup messages
- Model loading progress
- HTTP requests
- Face detection/recognition events
- Error messages

**View live logs:**
```bash
tail -f logs/server.log
```

### `logs/server.pid`
Contains the PID of the running server process. Used by stop and status scripts.

## Environment Variables

The start script sets these environment variables:
- `LD_LIBRARY_PATH`: CUDA library paths for GPU acceleration
- `OMP_NUM_THREADS`: OpenMP thread count (prevents hangs)
- `OPENBLAS_NUM_THREADS`: OpenBLAS thread count
- `MKL_NUM_THREADS`: MKL thread count
- `OMP_WAIT_POLICY`: Set to PASSIVE to prevent busy-waiting
- `HF_HOME`: HuggingFace cache directory on SD card

## Troubleshooting

### Server Won't Start

1. Check logs:
   ```bash
   tail -50 logs/server.log
   ```

2. Verify port 8000 is available:
   ```bash
   lsof -ti :8000
   ```

3. Check for Python errors:
   ```bash
   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Server Starts But Times Out

If the script says "failed to bind to port after 30 seconds" but the server is actually starting:
- This may indicate ML models are taking longer than 30 seconds to load
- Check `logs/server.log` to see if the server eventually starts
- The server process may still be running - use `./status_server.sh` to check

### Port Already in Use

```bash
# Find and kill process using port 8000
lsof -ti :8000 | xargs kill -9

# Or use the stop script
./stop_server.sh
```

### Stale PID File

If `status_server.sh` reports a stale PID file:
```bash
rm logs/server.pid
./start_server.sh
```

## Development vs Production

### Development (Manual Scripts)
Use `start_server.sh`, `stop_server.sh`, and `status_server.sh` for:
- Development and testing
- Manual server management
- Quick restarts
- Debugging

### Production (Systemd Service)
Use systemd service for:
- Automatic startup on boot
- Process monitoring and auto-restart
- Better resource management
- Centralized logging

See `scripts/deployment/` for systemd service files.

## Quick Reference

```bash
# Start server
./start_server.sh

# Check status
./status_server.sh

# View logs
tail -f logs/server.log

# Stop server
./stop_server.sh

# Restart server
./stop_server.sh && ./start_server.sh

# Stop and clean logs
./stop_server.sh --clean
```
