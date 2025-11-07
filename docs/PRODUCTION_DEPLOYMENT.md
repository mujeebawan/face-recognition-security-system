# Production Deployment Guide

Complete guide for deploying the Face Recognition Security System in a production environment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Systemd Service Setup](#systemd-service-setup)
5. [Health Monitoring](#health-monitoring)
6. [Maintenance](#maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **OS**: Ubuntu 20.04+ (tested on Jetson AGX Orin)
- **Python**: 3.8+
- **CUDA**: 11.4+ (for GPU acceleration)
- **System packages**:
  ```bash
  sudo apt update
  sudo apt install -y python3-pip python3-dev build-essential
  sudo apt install -y libgstreamer1.0-dev libopencv-dev
  sudo apt install -y psutil  # For health monitoring
  ```

### Python Dependencies

Install all required Python packages:

```bash
cd /home/mujeeb/Downloads/face-recognition-security-system
pip3 install -r requirements.txt
```

---

## System Requirements

### Minimum Hardware

- **CPU**: 4 cores
- **RAM**: 4GB (8GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Storage**: 20GB free space for logs and snapshots

### Network

- Port 8000 must be accessible for the web interface
- Camera must be accessible on the network
- Stable internet for system updates

---

## Installation Steps

### 1. Clone/Download the Repository

```bash
cd /home/mujeeb/Downloads
# If using git:
git pull origin master
```

### 2. Configure Environment

Edit `app/config.py` or create `.env` file:

```bash
# Camera configuration
CAMERA_IP=192.168.1.64
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_password

# Database
DATABASE_URL=sqlite:///./data/database.db

# Server
APP_HOST=0.0.0.0
APP_PORT=8000

# GPU
ENABLE_GPU=true
```

### 3. Initialize Database

```bash
python3 scripts/migration/init_db.py
```

### 4. Test the Application

Start the server manually to verify everything works:

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/health` to verify the server is running.

---

## Systemd Service Setup

### 1. Install as System Service

```bash
cd scripts/deployment
sudo ./install_service.sh
```

This script will:
- Create log directory: `/var/log/face-recognition/`
- Copy service file to `/etc/systemd/system/`
- Enable service to start on boot
- Configure automatic restart on failure

### 2. Start the Service

```bash
sudo systemctl start face-recognition
```

### 3. Check Service Status

```bash
sudo systemctl status face-recognition
```

### 4. View Logs

```bash
# Real-time logs
sudo journalctl -u face-recognition -f

# Last 100 lines
sudo journalctl -u face-recognition -n 100

# Application logs
tail -f /var/log/face-recognition/app.log
tail -f /var/log/face-recognition/error.log
```

### Service Configuration

The systemd service includes:

- **Auto-restart**: Restarts on failure with 10-second delay
- **Resource limits**:
  - Memory: 4GB limit (3GB soft limit)
  - CPU: 200% (2 cores)
- **Timeout**: 60s start timeout, 30s stop timeout
- **Security**: No new privileges, private /tmp
- **Workers**: 2 Uvicorn workers for high availability

Edit the service file to adjust:

```bash
sudo nano /etc/systemd/system/face-recognition.service
sudo systemctl daemon-reload
sudo systemctl restart face-recognition
```

---

## Health Monitoring

### Manual Health Check

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2025-11-07T14:30:00",
  "uptime_seconds": 3600,
  "database": {
    "status": "connected",
    "url": "data/database.db"
  },
  "resources": {
    "memory_used_mb": 2048,
    "memory_percent": 25.5,
    "cpu_percent": 15.2,
    "disk_percent": 45.0
  },
  "config": {
    "camera_configured": true,
    "gpu_enabled": true,
    "debug_mode": false
  }
}
```

### Automated Health Monitoring

Run the health monitor script:

```bash
cd scripts/deployment
./monitor_health.sh
```

This script will:
- Check health endpoint every 30 seconds
- Track consecutive failures
- Auto-restart service after 3 consecutive failures
- Log all health check results

### Run Monitor as Background Service

Create a systemd service for the monitor:

```bash
sudo nano /etc/systemd/system/face-recognition-monitor.service
```

```ini
[Unit]
Description=Face Recognition Health Monitor
After=face-recognition.service
Requires=face-recognition.service

[Service]
Type=simple
User=mujeeb
WorkingDirectory=/home/mujeeb/Downloads/face-recognition-security-system/scripts/deployment
ExecStart=/bin/bash /home/mujeeb/Downloads/face-recognition-security-system/scripts/deployment/monitor_health.sh
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable face-recognition-monitor
sudo systemctl start face-recognition-monitor
```

---

## Maintenance

### Backup Database

```bash
# Create backup directory
mkdir -p ~/backups/face-recognition

# Backup database
cp data/database.db ~/backups/face-recognition/database-$(date +%Y%m%d-%H%M%S).db

# Backup snapshots
tar -czf ~/backups/face-recognition/snapshots-$(date +%Y%m%d).tar.gz data/alert_snapshots/
```

### Rotate Logs

Add to `/etc/logrotate.d/face-recognition`:

```
/var/log/face-recognition/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 mujeeb mujeeb
    postrotate
        systemctl reload face-recognition >/dev/null 2>&1 || true
    endscript
}
```

### Update Application

```bash
# Stop service
sudo systemctl stop face-recognition

# Update code
cd /home/mujeeb/Downloads/face-recognition-security-system
git pull origin master  # or copy new files

# Backup database
cp data/database.db data/database.db.backup

# Run migrations if needed
python3 scripts/migration/migrate.py

# Restart service
sudo systemctl start face-recognition
sudo systemctl status face-recognition
```

### Clean Old Snapshots

```bash
# Delete snapshots older than 30 days
find data/alert_snapshots/ -name "*.jpg" -mtime +30 -delete

# Or create a cron job
crontab -e
# Add: 0 2 * * * find /home/mujeeb/Downloads/face-recognition-security-system/data/alert_snapshots/ -name "*.jpg" -mtime +30 -delete
```

---

## Troubleshooting

### Service Won't Start

1. Check logs:
   ```bash
   sudo journalctl -u face-recognition -n 50
   ```

2. Verify Python path:
   ```bash
   which python3
   python3 --version
   ```

3. Test manually:
   ```bash
   cd /home/mujeeb/Downloads/face-recognition-security-system
   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### High Memory Usage

1. Check current usage:
   ```bash
   curl http://localhost:8000/health | grep memory
   ```

2. Reduce workers in service file:
   ```bash
   sudo nano /etc/systemd/system/face-recognition.service
   # Change: --workers 2 to --workers 1
   ```

3. Restart service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart face-recognition
   ```

### Camera Connection Issues

1. Test camera directly:
   ```bash
   ffmpeg -i rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101 -frames:v 1 test.jpg
   ```

2. Check camera settings in config
3. Verify network connectivity
4. Check firewall rules

### Database Locked

1. Check for zombie processes:
   ```bash
   ps aux | grep python3
   ```

2. Stop all instances:
   ```bash
   sudo systemctl stop face-recognition
   pkill -f "uvicorn app.main:app"
   ```

3. Restart:
   ```bash
   sudo systemctl start face-recognition
   ```

### Port Already in Use

1. Find process using port 8000:
   ```bash
   sudo lsof -i :8000
   ```

2. Kill the process:
   ```bash
   kill <PID>
   ```

3. Or change port in service file

---

## Monitoring Best Practices

### Regular Checks

1. **Daily**: Check service status
   ```bash
   sudo systemctl status face-recognition
   ```

2. **Weekly**: Review logs for errors
   ```bash
   sudo journalctl -u face-recognition --since "1 week ago" | grep ERROR
   ```

3. **Monthly**: Review disk usage and clean old snapshots
   ```bash
   du -sh data/alert_snapshots/
   ```

### Performance Monitoring

Monitor these metrics:
- Memory usage (should stay under 3GB)
- CPU usage (should average under 50%)
- Disk space (keep 10GB+ free)
- Alert response time (check via health endpoint)

### Security Checks

1. Keep system updated:
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. Review user access logs
3. Check for failed login attempts
4. Verify camera authentication is secure

---

## Uninstall

To remove the service:

```bash
cd scripts/deployment
sudo ./uninstall_service.sh
```

This will:
- Stop the service
- Disable auto-start
- Remove service files
- Keep logs and data intact

To completely remove everything:

```bash
sudo rm -rf /var/log/face-recognition/
rm -rf /home/mujeeb/Downloads/face-recognition-security-system
```

---

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u face-recognition -n 100`
2. Test health endpoint: `curl http://localhost:8000/health`
3. Review documentation in `docs/` directory
4. Check GitHub issues: https://github.com/yourusername/face-recognition-security-system/issues

---

**Last Updated**: 2025-11-07
**Version**: 1.0.0
