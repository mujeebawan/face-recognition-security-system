#!/bin/bash
# Face Recognition System - Server Stop Script

echo "🛑 Stopping Face Recognition System..."
echo ""

# Check PID file first
if [ -f logs/server.pid ]; then
    SAVED_PID=$(cat logs/server.pid)
    if kill -0 $SAVED_PID 2>/dev/null; then
        echo "📋 Stopping server (PID: $SAVED_PID)"
        kill -15 $SAVED_PID 2>/dev/null
        sleep 2
        # Force kill if still running
        if kill -0 $SAVED_PID 2>/dev/null; then
            kill -9 $SAVED_PID 2>/dev/null
        fi
    fi
    rm -f logs/server.pid
fi

# Kill any remaining uvicorn processes
UVICORN_PIDS=$(pgrep -f "uvicorn.*app.main:app")
if [ ! -z "$UVICORN_PIDS" ]; then
    echo "📋 Found additional server processes: $UVICORN_PIDS"
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

# Clean up log files if requested
if [ "$1" == "--clean" ]; then
    rm -f logs/server.log logs/server.pid nohup.out
    echo "✅ Cleaned up log files"
fi

echo ""
echo "💡 To start server: ./start_server.sh"
