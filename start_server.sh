#!/bin/bash
# Face Recognition System - Server Start Script
# This script automatically kills old processes and starts the server fresh

echo "🚀 Face Recognition System - Starting..."
echo ""

# Kill any existing processes on port 8000
OLD_PIDS=$(lsof -ti :8000 2>/dev/null)
if [ ! -z "$OLD_PIDS" ]; then
    echo "🔄 Found existing server processes, cleaning up..."
    kill -9 $OLD_PIDS 2>/dev/null
    sleep 1
    echo "✓ Old processes killed"
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
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Server is running successfully!"
    echo ""
    echo "📍 Access the system:"
    echo "   • Admin Panel:    http://localhost:8000/admin"
    echo "   • Live Stream:    http://localhost:8000/live"
    echo "   • Dashboard:      http://localhost:8000/dashboard"
    echo "   • API Docs:       http://localhost:8000/docs"
    echo ""
    echo "📋 Server PID: $SERVER_PID (to stop: kill $SERVER_PID)"
    echo "📄 Logs: tail -f server.log"
else
    echo "❌ Server failed to start. Check server.log for errors."
    exit 1
fi
