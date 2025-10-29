"""
Quick test script for Alert API endpoints.
Run this after starting the server to verify alert functionality.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_alert_endpoints():
    """Test alert API endpoints"""

    print("=" * 60)
    print("Testing Alert System API")
    print("=" * 60)

    # Test 1: Get alert configuration
    print("\n1. Testing GET /api/alerts/config...")
    try:
        response = requests.get(f"{BASE_URL}/api/alerts/config")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            config = response.json()
            print(f"   ✅ Alert config retrieved")
            print(f"   - Enabled: {config['config']['enabled']}")
            print(f"   - Alert on unknown: {config['config']['alert_on_unknown']}")
            print(f"   - Cooldown: {config['config']['cooldown_seconds']}s")
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Get active alerts
    print("\n2. Testing GET /api/alerts/active...")
    try:
        response = requests.get(f"{BASE_URL}/api/alerts/active")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Active alerts retrieved")
            print(f"   - Total active alerts: {data['total']}")
            if data['total'] > 0:
                for alert in data['alerts'][:3]:  # Show first 3
                    print(f"     • {alert['event_type']} at {alert['timestamp']}")
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Get recent alerts (last 24 hours)
    print("\n3. Testing GET /api/alerts/recent?hours=24...")
    try:
        response = requests.get(f"{BASE_URL}/api/alerts/recent?hours=24")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Recent alerts retrieved")
            print(f"   - Total alerts (24h): {data['total']}")
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 4: Get alert statistics
    print("\n4. Testing GET /api/alerts/statistics?hours=24...")
    try:
        response = requests.get(f"{BASE_URL}/api/alerts/statistics?hours=24")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ Statistics retrieved")
            print(f"   - Total alerts: {stats['total_alerts']}")
            print(f"   - Unknown persons: {stats['unknown_persons']}")
            print(f"   - Known persons: {stats['known_persons']}")
            print(f"   - Unacknowledged: {stats['unacknowledged']}")
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Alert API Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    print("\n⚠️  Make sure the server is running:")
    print("   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000\n")

    input("Press Enter to start tests...")
    test_alert_endpoints()
