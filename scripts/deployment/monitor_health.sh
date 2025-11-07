#!/bin/bash
# Monitor Face Recognition System health and restart if unhealthy

set -e

# Configuration
HEALTH_URL="http://localhost:8000/health"
MAX_FAILURES=3
FAILURE_COUNT=0
CHECK_INTERVAL=30  # seconds

echo "================================================"
echo "Face Recognition System - Health Monitor"
echo "================================================"
echo "Checking health endpoint: $HEALTH_URL"
echo "Max failures before restart: $MAX_FAILURES"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "================================================"
echo ""

while true; do
    # Get current timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check health endpoint
    HTTP_CODE=$(curl -s -o /tmp/health_response.json -w "%{http_code}" $HEALTH_URL 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        # Parse response
        STATUS=$(cat /tmp/health_response.json | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

        if [ "$STATUS" = "healthy" ]; then
            echo "[$TIMESTAMP] ✅ System healthy"
            FAILURE_COUNT=0
        else
            echo "[$TIMESTAMP] ⚠️  System degraded: $STATUS"
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
        fi
    else
        echo "[$TIMESTAMP] ❌ Health check failed (HTTP $HTTP_CODE)"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi

    # Check if we need to restart
    if [ $FAILURE_COUNT -ge $MAX_FAILURES ]; then
        echo "[$TIMESTAMP] 🚨 Max failures reached! Restarting service..."

        if systemctl is-active --quiet face-recognition.service; then
            sudo systemctl restart face-recognition.service
            echo "[$TIMESTAMP] 🔄 Service restarted via systemd"
        else
            echo "[$TIMESTAMP] ⚠️  Systemd service not running, cannot restart"
        fi

        FAILURE_COUNT=0
        sleep 60  # Wait longer after restart
    else
        sleep $CHECK_INTERVAL
    fi
done
