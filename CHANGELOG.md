# Changelog

All notable changes to the Face Recognition Security System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.6.0] - 2025-11-13

### Added - Camera Zoom Control & Dynamic Settings Management

#### Camera Zoom Control
- **PTZ Control Module**: Added Hikvision ISAPI camera control integration
- **Zoom API Endpoints**:
  - `POST /api/ptz/zoom/in?speed=<1-100>` - Continuous zoom in
  - `POST /api/ptz/zoom/out?speed=<1-100>` - Continuous zoom out
  - `POST /api/ptz/zoom/stop` - Stop zoom movement
- **Live Stream Integration**:
  - Added zoom controls to live stream viewer page
  - Adjustable speed slider (10-100)
  - Real-time status display
  - Keyboard shortcuts: `+`/`=` (zoom in), `-`/`_` (zoom out), `Space` (stop)
- **No Authentication Required**: Zoom controls work without login for ease of use
- **Detailed Logging**: All PTZ commands logged with emoji indicators for easy debugging

#### Dynamic Settings Management
- **Settings Manager Module**: Real-time settings loaded from database
- **10-Second Cache**: Optimized performance with automatic cache refresh
- **Lazy Loading**: Database queries only when settings are accessed
- **System Control Panel** (Settings Page):
  - Live GPU status monitoring (utilization, memory, temperature)
  - System memory and disk usage display
  - Settings sync status (active vs database comparison)
  - Camera connectivity indicator
- **Test Endpoints**:
  - `POST /api/system/test/camera` - Test camera connection with current RTSP URL
  - `POST /api/system/test/gpu` - Test GPU availability and perform inference
  - `POST /api/system/reload-settings` - Force immediate settings reload
  - `GET /api/system/status` - Comprehensive system status with active vs DB settings

#### Resource Management
- **Resource Manager**: Memory monitoring and efficient model loading
- **Graceful Degradation**: Automatic model unloading under memory pressure

### Technical Details
- **Camera Model**: Hikvision DS-2CD7A47EWD-XZS (fixed mount with motorized zoom)
- **Communication**: HTTP Digest Auth with XML payloads via ISAPI
- **Settings Storage**: SQLite database with lazy-load singleton pattern
- **New Files**:
  - `app/core/ptz_control.py` - PTZ controller
  - `app/api/routes/ptz.py` - PTZ API endpoints
  - `app/core/settings_manager.py` - Dynamic settings loader
  - `app/api/routes/system_control.py` - System monitoring endpoints
  - `app/core/resource_manager.py` - Resource management utilities

### Fixed
- Settings page now actually applies changes (previously only saved to DB)
- Server startup freeze issue with settings manager (switched to lazy loading)

---

## [1.5.0] - 2025-11-07

### Added - Watchlist System & Production Hardening

#### Watchlist/Criminal Detection System
- **Threat Levels**: Added support for critical, high, medium, low, and none threat levels
- **Watchlist Status**: Added most_wanted, suspect, person_of_interest, banned status
- **Threat-Based Color Coding**:
  - Red for critical threats
  - Orange for high threats
  - Yellow for medium threats
  - Blue for low threats
  - Green for known safe persons
  - Gray for unknown persons
- **Color Legend Box**: Added visual legend on dashboard showing all threat level colors
- **Cached Threat Data**: Cached threat level and watchlist status in alerts for fast filtering

#### Simplified Alert Workflow
- **Full-Screen Alert Popup**: Automatic popup modal for watchlist person detections on dashboard
- **Side-by-Side Image Comparison**: Shows enrolled photo vs captured photo for guard verification
- **One-Click Guard Actions**:
  - Confirmed Match - Person identity verified
  - False Alarm - Wrong person detected
  - Investigating - Further verification needed
  - Apprehended - Person detained/arrested
  - Escalated - Requires supervisor/backup
- **Alert Queue System**: Sequential display of multiple alerts without overlap
- **Removed Redundant Acknowledgment**: Simplified to use only guard verification actions

#### Timezone Fix
- **PKT Timezone Support**: Proper UTC to PKT (Pakistan Standard Time, UTC+5) conversion
- **Fixed Timestamp Display**: Corrected time display across dashboard and alerts page

#### Enhanced Alert Filtering
- **Guard Verification Filters**: Filter by verified, unverified, confirmed, false_alarm, apprehended, escalated
- **Threat Level Filters**: Filter by critical, high, medium, low threat levels
- **Watchlist Status Filters**: Filter by most_wanted, suspect, POI, banned status
- **Name Search**: Text search for filtering alerts by person name
- **Clear Filters Button**: One-click reset of all filter selections

#### Production Hardening
- **Enhanced Health Check Endpoint** (`/health`):
  - Database connectivity test with live query
  - System resource monitoring (CPU, memory, disk usage)
  - Uptime tracking since server start
  - Configuration status reporting
  - HTTP 200 with JSON response for monitoring tools

- **Systemd Service Integration**:
  - Auto-restart on failure with 10-second delay
  - Resource limits (Memory: 4GB, CPU: 200%)
  - Security hardening (NoNewPrivileges, PrivateTmp)
  - Proper logging to `/var/log/face-recognition/`
  - Timeout configurations (60s start, 30s stop)
  - 2 Uvicorn workers for high availability

- **Deployment Scripts**:
  - `install_service.sh` - Install as systemd service
  - `uninstall_service.sh` - Remove systemd service
  - `monitor_health.sh` - Continuous health monitoring with auto-restart

- **Global Exception Handler**: Catches all unhandled exceptions with proper logging
- **Comprehensive Production Guide**: Complete deployment guide in `docs/PRODUCTION_DEPLOYMENT.md`

#### Enhanced User Management
- **Last Login Tracking**: Automatically updates user's last login timestamp
- **Update User Endpoint**: `PUT /api/auth/users/{user_id}` to update email, full_name, role
- **Deactivate User**: `PUT /api/auth/users/{user_id}/deactivate` to disable user access
- **Activate User**: `PUT /api/auth/users/{user_id}/activate` to re-enable user access
- **Comprehensive Activity Logging**: All user actions logged via Python logging system
- **Role-Based Access Control**: Admin/operator/viewer roles with proper permissions

### Changed

- **Alert Card Display**: Removed acknowledgment badges, now shows guard verification status
- **Statistics Display**: Changed "Unacknowledged" to "Not Verified" in alerts page
- **Image Endpoint Security**: Removed authentication requirement from snapshot endpoints for dashboard
- **Alert Filtering Logic**: Updated to use guard_verified and guard_action instead of acknowledged
- **Color Scheme Philosophy**: Changed from known/unknown to threat-based color coding

### Removed

- **Acknowledgment Modal**: Removed redundant acknowledge modal HTML from alerts page
- **Acknowledge Button**: Removed "Acknowledge" action button from alert cards
- **Acknowledge All Button**: Removed bulk acknowledgment functionality
- **Acknowledgment Functions**: Removed `openAcknowledgeModal()`, `submitAcknowledge()`, `acknowledgeAll()` functions
- **Audio Beep**: Removed alert audio beep functionality (not needed for local deployment)

### Fixed

- **Timezone Display**: Fixed UTC to PKT conversion - was showing 5 hours behind
- **Multiple Alert Popups**: Fixed issue where second alert wouldn't show without hard refresh
- **Image Loading**: Fixed "Loading..." issue by removing auth requirement from image endpoints
- **Filter ID Mismatch**: Fixed `clearFilters()` function using wrong filter ID
- **Startup Time Tracking**: Added global `startup_time` variable for uptime calculation

### Technical Details

#### Files Created
- `scripts/deployment/face-recognition.service` - Systemd service file with resource limits
- `scripts/deployment/install_service.sh` - Service installation script
- `scripts/deployment/uninstall_service.sh` - Service removal script
- `scripts/deployment/monitor_health.sh` - Health monitoring with auto-restart
- `docs/PRODUCTION_DEPLOYMENT.md` - Complete production deployment guide (500+ lines)

#### Files Modified
- `app/static/dashboard.html` - Alert popup, queue system, threat-based color coding, color legend
- `app/static/alerts.html` - Removed acknowledgment system, updated filters, guard status filters
- `app/api/routes/alerts.py` - Removed auth from image endpoints, guard verification tracking
- `app/main.py` - Enhanced health check, global exception handler, startup time tracking
- `app/api/routes/auth.py` - Last login tracking, update/deactivate/activate user endpoints
- `app/models/schemas.py` - Updated AlertEvent schema with threat level and watchlist fields
- `CURRENT_STATUS.md` - Added Milestone 5, updated progress to 85%, marked as production-ready

#### API Endpoints Added
- `PUT /api/auth/users/{user_id}` - Update user information (admin only)
- `PUT /api/auth/users/{user_id}/deactivate` - Deactivate user (admin only)
- `PUT /api/auth/users/{user_id}/activate` - Activate user (admin only)

#### API Endpoints Enhanced
- `GET /health` - Now includes database test, system resources, uptime tracking

---

## [1.4.0] - 2025-11-05

### Added - Reports & Analytics Dashboard

- Analytics API with 5 comprehensive endpoints
- Reports dashboard with Chart.js visualizations
- Time-series charts for alerts and recognition
- Person statistics with confidence tracking
- CSV export functionality

---

## [1.3.0] - 2025-11-04

### Added - Alert Management System

- Comprehensive alert management page
- Advanced filtering and search
- Alert acknowledgment system
- Bulk operations
- Image authentication fixes

---

## [1.2.0] - 2025-10-29

### Added - GPU Acceleration & Cleanup

- SCRFD GPU detection with TensorRT
- JetPack 6.1 upgrade
- Project reorganization
- Documentation overhaul
- Removed multi-agent system

---

## [1.1.0] - 2025-10-22

### Added - Core System

- FastAPI application with SQLite
- Hikvision IP camera integration
- InsightFace face recognition
- Person enrollment and recognition
- Admin panel and dashboard
- WebSocket real-time notifications

---

## Key Features Summary

### Current Capabilities
- ✅ Real-time face detection and recognition (~15-20 FPS)
- ✅ Watchlist/criminal detection with threat levels
- ✅ One-click guard verification workflow
- ✅ Production-ready deployment with systemd
- ✅ Health monitoring and auto-restart
- ✅ Comprehensive reports and analytics
- ✅ Role-based user management
- ✅ Alert filtering and search
- ✅ CSV data export

### Production Readiness
- ✅ Systemd service with auto-restart
- ✅ Health check endpoint
- ✅ Resource limits and security hardening
- ✅ Logging and monitoring
- ✅ Exception handling
- ✅ Complete deployment documentation

---

**Contributors**: Mujeeb and Claude (Anthropic)
**License**: [Specify License]
**Last Updated**: November 7, 2025
