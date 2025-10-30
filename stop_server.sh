#!/bin/bash
# Face Recognition System - Server Stop Script

echo "🛑 Stopping Face Recognition System..."
echo ""

# Kill uvicorn processes
UVICORN_PIDS=$(pgrep -f "uvicorn.*app.main:app")
if [ ! -z "$UVICORN_PIDS" ]; then
    echo "📋 Found server processes: $UVICORN_PIDS"
    pkill -9 -f "uvicorn.*app.main:app" 2>/dev/null
    sleep 1
fi

# Kill processes on port 8000
PORT_PIDS=$(lsof -ti :8000 2>/dev/null)
if [ ! -z "$PORT_PIDS" ]; then
    echo "📋 Cleaning up port 8000 processes: $PORT_PIDS"
    kill -9 $PORT_PIDS 2>/dev/null
    sleep 1
fi

# Verify everything is stopped
if lsof -ti :8000 > /dev/null 2>&1 || pgrep -f "uvicorn.*app.main:app" > /dev/null 2>&1; then
    echo "⚠️  Some processes may still be running"
else
    echo "✅ Server stopped successfully"
fi

# Clean up log file if requested
if [ "$1" == "--clean" ]; then
    rm -f server.log nohup.out
    echo "✅ Cleaned up log files"
fi

echo ""
echo "💡 To start server: ./start_server.sh"
