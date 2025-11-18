#!/usr/bin/env python3
"""
Camera Stream Testing Tool
- Select main/sub stream
- Frame skipping option (actually works)
- PTZ zoom controls
"""

import cv2
import tkinter as tk
from tkinter import ttk
import threading
import time
from PIL import Image, ImageTk
import requests
from requests.auth import HTTPDigestAuth
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PTZController:
    """Simple PTZ controller for zoom"""

    def __init__(self, camera_ip: str, username: str, password: str):
        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.channel = 1
        self.base_url = f"http://{camera_ip}/ISAPI/PTZCtrl/channels/{self.channel}"

    def _send_command(self, params: dict) -> bool:
        """Send PTZ command"""
        try:
            url = f"{self.base_url}/continuous"

            xml_payload = '<?xml version="1.0" encoding="UTF-8"?><PTZData>'
            for key, value in params.items():
                xml_payload += f'<{key}>{value}</{key}>'
            xml_payload += '</PTZData>'

            response = requests.put(
                url,
                data=xml_payload,
                auth=HTTPDigestAuth(self.username, self.password),
                headers={'Content-Type': 'application/xml'},
                timeout=3
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"PTZ error: {e}")
            return False

    def zoom_in(self, speed: int = 50):
        """Zoom in"""
        return self._send_command({'zoom': speed})

    def zoom_out(self, speed: int = 50):
        """Zoom out"""
        return self._send_command({'zoom': -speed})

    def zoom_stop(self):
        """Stop zoom"""
        return self._send_command({'zoom': 0})


class CameraStreamTester:
    """Camera stream testing application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Camera Stream Tester")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e293b')

        # Make window resizable
        self.root.minsize(1000, 700)

        # Camera config
        self.camera_ip = "192.168.1.64"
        self.username = "admin"
        self.password = "Mujeeb@321"

        # Stream URLs
        self.main_stream = f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554/Streaming/Channels/101"
        self.sub_stream = f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554/Streaming/Channels/102"

        # PTZ controller
        self.ptz = PTZController(self.camera_ip, self.username, self.password)

        # State
        self.cap = None
        self.running = False
        self.current_stream = None
        self.frame_skip = 0
        self.frame_counter = 0
        self.last_skip_value = 0
        self.restart_requested = False

        # Build UI
        self.build_ui()

        # Keyboard shortcuts
        self.root.bind('+', lambda e: self.zoom_in_start())
        self.root.bind('-', lambda e: self.zoom_out_start())
        self.root.bind('<KeyRelease-plus>', lambda e: self.zoom_stop())
        self.root.bind('<KeyRelease-minus>', lambda e: self.zoom_stop())

    def build_ui(self):
        """Build the user interface"""

        # Header
        header = tk.Frame(self.root, bg='#124877', height=80)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="Camera Stream Tester",
            font=('Arial', 24, 'bold'),
            bg='#124877',
            fg='#cc9933'
        )
        title.pack(pady=20)

        # Main container
        main = tk.Frame(self.root, bg='#1e293b')
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel - Controls (with scrollbar)
        left_container = tk.Frame(main, bg='#0f172a', width=350)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_container.pack_propagate(False)

        # Create canvas for scrolling
        canvas = tk.Canvas(left_container, bg='#0f172a', highlightthickness=0, width=350)
        scrollbar = tk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        left_panel = tk.Frame(canvas, bg='#0f172a')

        left_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=left_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Stream Selection
        stream_frame = tk.LabelFrame(
            left_panel,
            text="Stream Selection",
            font=('Arial', 12, 'bold'),
            bg='#0f172a',
            fg='#cc9933',
            padx=20,
            pady=15
        )
        stream_frame.pack(fill=tk.X, padx=10, pady=10)

        self.stream_var = tk.StringVar(value="main")
        self.stream_var.trace('w', self.on_stream_change)

        tk.Radiobutton(
            stream_frame,
            text="Main Stream (High Quality)",
            variable=self.stream_var,
            value="main",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#e2e8f0',
            selectcolor='#334155',
            activebackground='#0f172a',
            activeforeground='#cc9933'
        ).pack(anchor=tk.W, pady=5)

        tk.Radiobutton(
            stream_frame,
            text="Sub Stream (Lower Quality)",
            variable=self.stream_var,
            value="sub",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#e2e8f0',
            selectcolor='#334155',
            activebackground='#0f172a',
            activeforeground='#cc9933'
        ).pack(anchor=tk.W, pady=5)

        # Stream change notification
        self.stream_change_label = tk.Label(
            stream_frame,
            text="",
            font=('Arial', 9, 'italic'),
            bg='#0f172a',
            fg='#f59e0b'
        )
        self.stream_change_label.pack(pady=5)

        # Frame Skipping
        skip_frame = tk.LabelFrame(
            left_panel,
            text="Frame Skipping",
            font=('Arial', 12, 'bold'),
            bg='#0f172a',
            fg='#cc9933',
            padx=20,
            pady=15
        )
        skip_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            skip_frame,
            text="Skip every N frames:",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#e2e8f0'
        ).pack(anchor=tk.W)

        self.skip_var = tk.IntVar(value=0)
        skip_scale = tk.Scale(
            skip_frame,
            from_=0,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.skip_var,
            font=('Arial', 9),
            bg='#0f172a',
            fg='#cc9933',
            troughcolor='#334155',
            activebackground='#124877',
            highlightthickness=0
        )
        skip_scale.pack(fill=tk.X, pady=5)

        self.skip_label = tk.Label(
            skip_frame,
            text="Process all frames",
            font=('Arial', 9, 'italic'),
            bg='#0f172a',
            fg='#cbd5e1'
        )
        self.skip_label.pack()

        skip_scale.config(command=self.update_skip_label)

        # PTZ Zoom Controls
        zoom_frame = tk.LabelFrame(
            left_panel,
            text="PTZ Zoom Control",
            font=('Arial', 12, 'bold'),
            bg='#0f172a',
            fg='#cc9933',
            padx=20,
            pady=15
        )
        zoom_frame.pack(fill=tk.X, padx=10, pady=10)

        # Zoom speed
        tk.Label(
            zoom_frame,
            text="Zoom Speed:",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#e2e8f0'
        ).pack(anchor=tk.W)

        self.zoom_speed_var = tk.IntVar(value=70)
        zoom_speed_scale = tk.Scale(
            zoom_frame,
            from_=10,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.zoom_speed_var,
            font=('Arial', 9),
            bg='#0f172a',
            fg='#cc9933',
            troughcolor='#334155',
            activebackground='#124877',
            highlightthickness=0
        )
        zoom_speed_scale.pack(fill=tk.X, pady=5)

        # Zoom buttons
        btn_frame = tk.Frame(zoom_frame, bg='#0f172a')
        btn_frame.pack(fill=tk.X, pady=10)

        self.zoom_in_btn = tk.Button(
            btn_frame,
            text="➕ Zoom In",
            font=('Arial', 11, 'bold'),
            bg='#124877',
            fg='white',
            activebackground='#0F518B',
            relief=tk.FLAT,
            cursor='hand2',
            height=2
        )
        self.zoom_in_btn.pack(fill=tk.X, pady=5)
        self.zoom_in_btn.bind('<ButtonPress-1>', lambda e: self.zoom_in_start())
        self.zoom_in_btn.bind('<ButtonRelease-1>', lambda e: self.zoom_stop())

        self.zoom_out_btn = tk.Button(
            btn_frame,
            text="➖ Zoom Out",
            font=('Arial', 11, 'bold'),
            bg='#124877',
            fg='white',
            activebackground='#0F518B',
            relief=tk.FLAT,
            cursor='hand2',
            height=2
        )
        self.zoom_out_btn.pack(fill=tk.X, pady=5)
        self.zoom_out_btn.bind('<ButtonPress-1>', lambda e: self.zoom_out_start())
        self.zoom_out_btn.bind('<ButtonRelease-1>', lambda e: self.zoom_stop())

        # Status
        self.zoom_status = tk.Label(
            zoom_frame,
            text="Ready",
            font=('Arial', 9),
            bg='#0f172a',
            fg='#cbd5e1'
        )
        self.zoom_status.pack(pady=5)

        # Stream Controls
        control_frame = tk.Frame(left_panel, bg='#0f172a')
        control_frame.pack(fill=tk.X, padx=10, pady=20)

        self.start_btn = tk.Button(
            control_frame,
            text="▶ Start Stream",
            font=('Arial', 12, 'bold'),
            bg='#10b981',
            fg='white',
            activebackground='#059669',
            relief=tk.FLAT,
            cursor='hand2',
            command=self.start_stream,
            height=2
        )
        self.start_btn.pack(fill=tk.X, pady=5)

        self.stop_btn = tk.Button(
            control_frame,
            text="⏹ Stop Stream",
            font=('Arial', 12, 'bold'),
            bg='#ef4444',
            fg='white',
            activebackground='#dc2626',
            relief=tk.FLAT,
            cursor='hand2',
            command=self.stop_stream,
            state=tk.DISABLED,
            height=2
        )
        self.stop_btn.pack(fill=tk.X, pady=5)

        # Stats
        stats_frame = tk.LabelFrame(
            left_panel,
            text="Statistics",
            font=('Arial', 12, 'bold'),
            bg='#0f172a',
            fg='#cc9933',
            padx=20,
            pady=15
        )
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        self.stream_info_label = tk.Label(
            stats_frame,
            text="Stream: --",
            font=('Arial', 10, 'bold'),
            bg='#0f172a',
            fg='#cc9933'
        )
        self.stream_info_label.pack(anchor=tk.W, pady=(0, 5))

        self.fps_label = tk.Label(
            stats_frame,
            text="Display FPS: --",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#e2e8f0'
        )
        self.fps_label.pack(anchor=tk.W)

        self.frames_label = tk.Label(
            stats_frame,
            text="Processed: 0",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#e2e8f0'
        )
        self.frames_label.pack(anchor=tk.W)

        self.skipped_label = tk.Label(
            stats_frame,
            text="Skipped: 0",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#e2e8f0'
        )
        self.skipped_label.pack(anchor=tk.W)

        self.skip_rate_label = tk.Label(
            stats_frame,
            text="Skip Rate: 0%",
            font=('Arial', 10),
            bg='#0f172a',
            fg='#cbd5e1'
        )
        self.skip_rate_label.pack(anchor=tk.W)

        # Right panel - Video display
        right_panel = tk.Frame(main, bg='#000000')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(right_panel, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(
            right_panel,
            text="Select stream and click Start",
            font=('Arial', 14),
            bg='#000000',
            fg='#cbd5e1'
        )
        self.status_label.pack(pady=20)

    def update_skip_label(self, val):
        """Update frame skip label"""
        val = int(float(val))
        self.frame_skip = val
        if val == 0:
            self.skip_label.config(text="Process all frames")
        else:
            self.skip_label.config(text=f"Skip {val} frame(s), process every {val+1}th")

        # Show change notification if streaming
        if self.running and val != self.last_skip_value:
            self.skip_label.config(text=f"✓ Updated: Skip {val}, process every {val+1}th" if val > 0 else "✓ Updated: Process all frames", fg='#10b981')
            # Reset color after 2 seconds
            self.root.after(2000, lambda: self.skip_label.config(fg='#cbd5e1'))
            self.last_skip_value = val

    def on_stream_change(self, *args):
        """Handle stream selection change"""
        if self.running:
            self.stream_change_label.config(text="⚠ Restart to apply")
            # Auto-restart after 1 second
            self.root.after(1000, self.auto_restart_stream)
        else:
            self.stream_change_label.config(text="")

    def auto_restart_stream(self):
        """Auto-restart stream when settings change"""
        if self.running:
            logger.info("Auto-restarting stream due to setting change")
            self.stop_stream()
            self.root.after(500, self.start_stream)

    def zoom_in_start(self):
        """Start zooming in"""
        speed = self.zoom_speed_var.get()
        self.zoom_status.config(text=f"⏫ Zooming IN (speed: {speed})")
        self.ptz.zoom_in(speed)

        # Continue zooming while button held
        def continuous_zoom():
            if self.zoom_in_btn.winfo_containing(*self.root.winfo_pointerxy()) == self.zoom_in_btn:
                self.ptz.zoom_in(speed)
                self.root.after(200, continuous_zoom)

        self.root.after(200, continuous_zoom)

    def zoom_out_start(self):
        """Start zooming out"""
        speed = self.zoom_speed_var.get()
        self.zoom_status.config(text=f"⏬ Zooming OUT (speed: {speed})")
        self.ptz.zoom_out(speed)

        # Continue zooming while button held
        def continuous_zoom():
            if self.zoom_out_btn.winfo_containing(*self.root.winfo_pointerxy()) == self.zoom_out_btn:
                self.ptz.zoom_out(speed)
                self.root.after(200, continuous_zoom)

        self.root.after(200, continuous_zoom)

    def zoom_stop(self):
        """Stop zoom"""
        self.ptz.zoom_stop()
        self.zoom_status.config(text="⏹ Stopped")
        self.root.after(1500, lambda: self.zoom_status.config(text="Ready"))

    def start_stream(self):
        """Start video stream"""
        if self.running:
            return

        # Get selected stream
        stream_type = self.stream_var.get()
        self.current_stream = self.main_stream if stream_type == "main" else self.sub_stream
        self.current_stream_name = "Main Stream" if stream_type == "main" else "Sub Stream"

        logger.info(f"Starting {stream_type} stream: {self.current_stream}")

        # Update stream info
        self.root.after(0, lambda: self.stream_info_label.config(text=f"Stream: {self.current_stream_name}"))

        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Connecting to camera...")

        # Start stream thread
        self.running = True
        self.frame_counter = 0
        self.skipped_counter = 0
        self.total_frames = 0

        threading.Thread(target=self.stream_loop, daemon=True).start()

    def stop_stream(self):
        """Stop video stream"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stream stopped")
        self.video_label.config(image='')

    def stream_loop(self):
        """Main streaming loop"""
        try:
            # Open stream
            self.cap = cv2.VideoCapture(self.current_stream)

            # Set buffer size to 1 for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.root.after(0, lambda: self.status_label.config(text="Failed to connect to camera"))
                self.running = False
                return

            self.root.after(0, lambda: self.status_label.config(text=""))

            fps_start_time = time.time()
            fps_frame_count = 0

            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    logger.error("Failed to read frame")
                    break

                self.total_frames += 1

                # Frame skipping logic (reads value dynamically for real-time changes)
                skip = self.skip_var.get()
                if skip > 0:
                    # Use a counter that resets per skip cycle for immediate response
                    if (self.total_frames - 1) % (skip + 1) != 0:
                        self.skipped_counter += 1
                        continue

                # Process frame
                self.frame_counter += 1
                fps_frame_count += 1

                # Calculate FPS
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    fps = fps_frame_count / elapsed
                    self.root.after(0, lambda f=fps: self.fps_label.config(text=f"Display FPS: {f:.1f}"))
                    fps_start_time = time.time()
                    fps_frame_count = 0

                # Update stats
                self.root.after(0, lambda: self.frames_label.config(text=f"Processed: {self.frame_counter}"))
                self.root.after(0, lambda: self.skipped_label.config(text=f"Skipped: {self.skipped_counter}"))

                # Calculate skip rate
                if self.total_frames > 0:
                    skip_rate = (self.skipped_counter / self.total_frames) * 100
                    self.root.after(0, lambda r=skip_rate: self.skip_rate_label.config(text=f"Skip Rate: {r:.1f}%"))

                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to fit window
                height, width = frame_rgb.shape[:2]
                max_width = 850
                max_height = 700

                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)

                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

                # Convert to PhotoImage
                img = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image=img)

                # Update display
                self.root.after(0, lambda p=photo: self.update_frame(p))

            # Cleanup
            if self.cap:
                self.cap.release()

        except Exception as e:
            logger.error(f"Stream error: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {e}"))
        finally:
            self.running = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def update_frame(self, photo):
        """Update video frame"""
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Keep reference

    def on_closing(self):
        """Handle window close"""
        self.stop_stream()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CameraStreamTester(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
