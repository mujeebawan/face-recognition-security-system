#!/bin/bash
# Face Recognition System - Server Start Script
# This script automatically kills old processes and starts the server fresh

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

# Start the server
echo ""
echo "🎬 Starting FastAPI server..."
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
SERVER_PID=$!

echo "✓ Server started with PID: $SERVER_PID"
echo ""
echo "⏳ Waiting for models to load (this takes ~10 seconds)..."
sleep 10

# Check if server is responding
if lsof -ti :8000 > /dev/null 2>&1; then
    # Get local IP address
    LOCAL_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -1)

    echo "✅ Server is running successfully!"
    echo ""
    echo "📍 Access the system:"
    if [ ! -z "$LOCAL_IP" ]; then
        echo "   • Admin Panel:    http://${LOCAL_IP}:8000/admin"
        echo "   • Live Stream:    http://${LOCAL_IP}:8000/live"
        echo "   • Dashboard:      http://${LOCAL_IP}:8000/dashboard"
        echo "   • API Docs:       http://${LOCAL_IP}:8000/docs"
    else
        echo "   • Admin Panel:    http://localhost:8000/admin"
        echo "   • Live Stream:    http://localhost:8000/live"
        echo "   • Dashboard:      http://localhost:8000/dashboard"
        echo "   • API Docs:       http://localhost:8000/docs"
    fi
    echo ""
    echo "📋 To stop server: ./stop_server.sh"
    echo "📄 View logs: tail -f server.log"
    echo ""
else
    echo "❌ Server failed to start. Check server.log for errors:"
    echo ""
    tail -20 server.log
    exit 1
fi
