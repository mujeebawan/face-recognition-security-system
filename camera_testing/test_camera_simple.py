#!/usr/bin/env python3
"""
Simple Camera Stream Tester (CLI version)
Quick test of camera streams without GUI
"""

import cv2
import sys
import argparse
import time


def test_stream(stream_url, frame_skip=0, duration=30):
    """
    Test camera stream

    Args:
        stream_url: RTSP URL
        frame_skip: Number of frames to skip (0 = process all)
        duration: How long to run (seconds)
    """
    print(f"\n{'='*60}")
    print(f"Testing Camera Stream")
    print(f"{'='*60}")
    print(f"Stream URL: {stream_url}")
    print(f"Frame skip: {frame_skip} (process every {frame_skip+1}th frame)")
    print(f"Duration: {duration}s")
    print(f"{'='*60}\n")

    # Open stream
    print("Connecting to camera...")
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Failed to connect to camera")
        return False

    print("✓ Connected!")
    print("\nPress 'q' to quit early\n")

    # Get stream info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream resolution: {width}x{height}")

    start_time = time.time()
    frame_count = 0
    skipped_count = 0
    total_frames = 0
    fps_start = time.time()
    fps_frames = 0

    try:
        while True:
            # Check duration
            if time.time() - start_time > duration:
                print("\nTest duration completed")
                break

            ret, frame = cap.read()

            if not ret:
                print("❌ Failed to read frame")
                break

            total_frames += 1

            # Frame skipping
            if frame_skip > 0:
                if total_frames % (frame_skip + 1) != 0:
                    skipped_count += 1
                    continue

            frame_count += 1
            fps_frames += 1

            # Calculate FPS every second
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                print(f"\rFPS: {fps:5.1f} | Processed: {frame_count:6d} | Skipped: {skipped_count:6d} | Total: {total_frames:6d}", end='')
                fps_start = time.time()
                fps_frames = 0

            # Show frame
            cv2.imshow('Camera Stream Test (Press Q to quit)', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nUser quit")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        print(f"Duration: {total_time:.1f}s")
        print(f"Total frames read: {total_frames}")
        print(f"Frames processed: {frame_count}")
        print(f"Frames skipped: {skipped_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"{'='*60}\n")

    return True


def main():
    """Main entry point"""

    # Camera config
    CAMERA_IP = "192.168.1.64"
    USERNAME = "admin"
    PASSWORD = "Mujeeb@321"

    MAIN_STREAM = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/101"
    SUB_STREAM = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/102"

    parser = argparse.ArgumentParser(description='Camera Stream Tester')
    parser.add_argument(
        '--stream',
        choices=['main', 'sub'],
        default='main',
        help='Stream to test (main or sub)'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Frame skip: 0=all frames, 1=every 2nd, 2=every 3rd, etc.'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Test duration in seconds'
    )

    args = parser.parse_args()

    # Select stream
    stream_url = MAIN_STREAM if args.stream == 'main' else SUB_STREAM

    # Run test
    success = test_stream(stream_url, args.skip, args.duration)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
