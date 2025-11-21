#!/bin/bash
# Face Recognition System - Server Start Script
# This script automatically kills old processes and starts the server fresh

# Set CUDA library path for PyTorch + Stable Diffusion + ONNX Runtime GPU
export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Fix OpenBLAS/OpenMP threading conflict (prevents hang)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE

# Use SD card for HuggingFace model cache
export HF_HOME=/media/sdcard/huggingface_cache

echo "🚀 Face Recognition System - Starting..."
echo ""

# Kill any existing processes
echo "🔄 Stopping any existing server processes..."
pkill -9 -f "uvicorn.*app.main:app" 2>/dev/null
sleep 2

OLD_PIDS=$(lsof -ti :8000 2>/dev/null)
if [ ! -z "$OLD_PIDS" ]; then
    kill -9 $OLD_PIDS 2>/dev/null
    sleep 1
    echo "✓ Old processes cleaned up"
else
    echo "✓ No existing processes found"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server
echo ""
echo "🎬 Starting FastAPI server..."
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/server.log 2>&1 &
SERVER_PID=$!

# Save PID to file
echo $SERVER_PID > logs/server.pid

echo "✓ Server started with PID: $SERVER_PID"
echo ""
echo "⏳ Waiting for server to bind to port (loading ML models, ~15-20 seconds)..."

# Wait up to 30 seconds for server to bind to port
MAX_RETRIES=30
RETRY_COUNT=0
SERVER_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if lsof -ti :8000 > /dev/null 2>&1; then
        SERVER_READY=true
        break
    fi

    # Check if process is still alive
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server process died unexpectedly. Check logs/server.log:"
        echo ""
        tail -30 logs/server.log
        exit 1
    fi

    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))

    # Show progress every 5 seconds
    if [ $((RETRY_COUNT % 5)) -eq 0 ]; then
        echo "   Still waiting... ($RETRY_COUNT seconds)"
    fi
done

# Check if server is responding
if [ "$SERVER_READY" = true ]; then
    # Get all IP addresses
    WIFI_IP=$(ip -4 addr show wlP1p1s0 2>/dev/null | grep inet | awk '{print $2}' | cut -d/ -f1)
    ETH_IP=$(ip -4 addr show eno1 2>/dev/null | grep inet | awk '{print $2}' | cut -d/ -f1)

    echo "✅ Server is running successfully!"
    echo ""
    echo "📍 Access the system:"
    echo ""
    echo "🖥️  From this Jetson (local access):"
    echo "   • Dashboard:      http://localhost:8000/dashboard"
    echo "   • Live Stream:    http://localhost:8000/live"
    echo "   • Admin Panel:    http://localhost:8000/admin"
    echo "   • API Docs:       http://localhost:8000/docs"
    echo ""
    echo "🌐 From other devices on network:"
    if [ ! -z "$WIFI_IP" ]; then
        echo "   WiFi Network (192.168.0.x):"
        echo "   • Dashboard:      http://${WIFI_IP}:8000/dashboard"
        echo "   • Live Stream:    http://${WIFI_IP}:8000/live"
        echo "   • Admin Panel:    http://${WIFI_IP}:8000/admin"
        echo ""
    fi
    if [ ! -z "$ETH_IP" ]; then
        echo "   Ethernet Network (192.168.1.x - Camera Network):"
        echo "   • Dashboard:      http://${ETH_IP}:8000/dashboard"
        echo "   • Live Stream:    http://${ETH_IP}:8000/live"
        echo "   • Admin Panel:    http://${ETH_IP}:8000/admin"
        echo ""
    fi
    echo "📋 To stop server: ./stop_server.sh"
    echo "📄 View logs: tail -f logs/server.log"
    echo "📄 Server PID: $SERVER_PID (saved to logs/server.pid)"
    echo ""
else
    echo "❌ Server failed to bind to port 8000 after 30 seconds."
    echo "   The process may still be initializing. Check logs/server.log for details:"
    echo ""
    tail -30 logs/server.log
    echo ""
    echo "💡 Tip: The server process (PID $SERVER_PID) may still be loading."
    echo "   Monitor with: tail -f logs/server.log"
    exit 1
fi
