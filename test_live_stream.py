#!/usr/bin/env python3
"""
Test the live stream endpoint
"""
import requests
import time

url = "http://localhost:8000/api/stream/live"

print("Testing live stream endpoint...")
print(f"URL: {url}")

try:
    response = requests.get(url, stream=True, timeout=10)

    if response.status_code == 200:
        print(f"✓ Stream endpoint responding (status: {response.status_code})")
        print(f"✓ Content-Type: {response.headers.get('content-type')}")

        # Read a few frames
        frame_count = 0
        start_time = time.time()

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                frame_count += 1
                if frame_count >= 5 or (time.time() - start_time) > 5:
                    break

        print(f"✓ Received {frame_count} chunks in {time.time() - start_time:.1f}s")
        print("✓ Live stream is working!")

    else:
        print(f"✗ Stream returned status code: {response.status_code}")

except requests.exceptions.Timeout:
    print("✗ Request timed out - camera may not be connecting")
except Exception as e:
    print(f"✗ Error: {e}")
