#!/bin/bash
# Face Recognition System - Server Stop Script

echo "🛑 Stopping Face Recognition System..."

PIDS=$(lsof -ti :8000 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "ℹ️  No server processes found on port 8000"
else
    echo "Found PIDs: $PIDS"
    kill -9 $PIDS 2>/dev/null
    sleep 1
    echo "✓ Server stopped successfully"
fi

# Clean up log file if requested
if [ "$1" == "--clean" ]; then
    rm -f server.log
    echo "✓ Cleaned up server.log"
fi

echo ""
echo "Server stopped. To start again: ./start_server.sh"
