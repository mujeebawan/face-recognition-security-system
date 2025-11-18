#!/bin/bash
# Face Recognition System - Server Status Script

echo "🔍 Face Recognition System - Server Status"
echo ""

# Check if PID file exists
if [ -f logs/server.pid ]; then
    SAVED_PID=$(cat logs/server.pid)
    if kill -0 $SAVED_PID 2>/dev/null; then
        echo "✅ Server is running"
        echo "   • PID: $SAVED_PID"

        # Get memory and CPU usage
        MEM_CPU=$(ps -p $SAVED_PID -o %mem,%cpu --no-headers)
        echo "   • Memory/CPU: $MEM_CPU"

        # Check if port is bound
        if lsof -ti :8000 > /dev/null 2>&1; then
            echo "   • Port 8000: Bound ✓"

            # Get local IP
            LOCAL_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -1)
            if [ ! -z "$LOCAL_IP" ]; then
                echo ""
                echo "📍 Access URLs:"
                echo "   • Admin Panel:    http://${LOCAL_IP}:8000/admin"
                echo "   • Live Stream:    http://${LOCAL_IP}:8000/live"
                echo "   • Dashboard:      http://${LOCAL_IP}:8000/dashboard"
            fi
        else
            echo "   • Port 8000: Not bound (server may be starting...)"
        fi

        # Show last few log lines
        if [ -f logs/server.log ]; then
            echo ""
            echo "📄 Recent logs (last 5 lines):"
            tail -5 logs/server.log | sed 's/^/   /'
        fi
    else
        echo "⚠️  PID file exists but process is not running"
        echo "   • Stale PID: $SAVED_PID"
        echo "   • Run ./start_server.sh to start"
    fi
else
    # No PID file, check if server is running anyway
    UVICORN_PIDS=$(pgrep -f "uvicorn.*app.main:app")
    if [ ! -z "$UVICORN_PIDS" ]; then
        echo "⚠️  Server is running but no PID file found"
        echo "   • PIDs: $UVICORN_PIDS"
        echo "   • Run ./stop_server.sh to clean up"
    else
        echo "❌ Server is not running"
        echo "   • Run ./start_server.sh to start"
    fi
fi

echo ""
echo "💡 Commands:"
echo "   • Start server:  ./start_server.sh"
echo "   • Stop server:   ./stop_server.sh"
echo "   • View logs:     tail -f logs/server.log"
echo ""
