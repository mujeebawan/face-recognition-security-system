# UI Screenshots Guide
**Date:** November 18, 2025
**Purpose:** Document what screenshots are needed for complete project documentation

---

## 📸 Screenshots to Capture

### Setup Instructions
1. **Access the system**: http://192.168.0.117:8000
2. **Login** with admin credentials
3. **Capture screenshots** of each page listed below
4. **Save** to the specified location
5. **Name format**: `{page_name}_{description}.png`

---

## 📋 Required Screenshots

### 1. **Login Page**
**URL:** `http://192.168.0.117:8000/`
**Save to:** `docs/screenshots/01_login_page.png`
**What to show:**
- Login form
- Clean interface
- System branding

---

### 2. **Dashboard (Main View)**
**URL:** `http://192.168.0.117:8000/dashboard`
**Save to:** `docs/screenshots/02_dashboard_main.png`
**What to show:**
- Live camera feed
- Real-time statistics
- Alert notifications
- System status

**Additional:**
- `docs/screenshots/02_dashboard_alert_popup.png` - When an alert appears
- `docs/screenshots/02_dashboard_watchlist_detection.png` - When watchlist person detected

---

### 3. **Admin Panel**
**URL:** `http://192.168.0.117:8000/admin`
**Save to:** `docs/screenshots/03_admin_panel.png`
**What to show:**
- Person enrollment form
- List of enrolled persons
- Person management interface

**Additional:**
- `docs/screenshots/03_admin_enrollment_options.png` - Show augmentation options (ControlNet, SD, LivePortrait checkboxes)
- `docs/screenshots/03_admin_person_details.png` - Click on a person to show details

---

### 4. **Alerts Management**
**URL:** `http://192.168.0.117:8000/alerts`
**Save to:** `docs/screenshots/04_alerts_page.png`
**What to show:**
- Alert list with filters
- Alert statistics
- Filter options (threat level, watchlist status, etc.)

**Additional:**
- `docs/screenshots/04_alerts_filters.png` - Show the filter panel expanded
- `docs/screenshots/04_alerts_detail_modal.png` - Click on an alert to show details

---

### 5. **Reports & Analytics**
**URL:** `http://192.168.0.117:8000/reports`
**Save to:** `docs/screenshots/05_reports_analytics.png`
**What to show:**
- Summary statistics cards
- Charts (Alerts Over Time, Success Rate, etc.)
- Time period selectors

---

### 6. **Live Stream**
**URL:** `http://192.168.0.117:8000/live`
**Save to:** `docs/screenshots/06_live_stream.png`
**What to show:**
- Live camera feed
- Stream controls
- Face detection boxes (if any faces visible)

---

### 7. **System Settings**
**URL:** `http://192.168.0.117:8000/settings`
**Save to:** `docs/screenshots/07_settings_page.png`
**What to show:**
- Settings interface
- Configuration options
- Camera settings, recognition thresholds, etc.

---

## 🎬 Special Screenshots

### 8. **Face Detection in Action**
**Save to:** `docs/screenshots/08_face_detection_demo.png`
**What to show:**
- Live feed with face bounding boxes
- Confidence scores visible
- Person name labels (if recognized)

### 9. **Enrollment Progress**
**Save to:** `docs/screenshots/09_enrollment_progress.png`
**What to show:**
- Progress bar during enrollment
- Shows the 9-stage progress (mentioned in code)

### 10. **Augmentation Results**
**Save to:** `docs/screenshots/10_augmentation_results.png`
**What to show:**
- Generated images from ControlNet/SD/LivePortrait
- Side-by-side comparison of original + generated angles

---

## 📊 Architecture Diagrams (Optional)

### 11. **System Architecture**
**Save to:** `docs/images/architecture_diagram.png`
**What to create:**
- Block diagram showing: Camera → GStreamer → SCRFD → ArcFace → Database
- Optional: Use draw.io or similar tool

### 12. **Recognition Flow**
**Save to:** `docs/images/recognition_flow.png`
**What to show:**
- Flowchart: Image Input → Face Detection → Embedding Extraction → Similarity Matching → Result

---

## ✅ After Capturing Screenshots

1. **Verify** all images are clear and readable
2. **Add** them to git:
   ```bash
   git add docs/screenshots/*.png
   git add docs/images/*.png
   ```

3. **Update** README.md to reference the screenshots
4. **Commit**:
   ```bash
   git add .
   git commit -m "docs: Add UI screenshots and visual documentation"
   git push origin master
   ```

---

## 📝 Notes

- **Resolution**: Capture at 1920x1080 or 1280x720
- **Format**: PNG (better quality for UI screenshots)
- **File size**: Try to keep under 500KB each (use compression if needed)
- **Privacy**: Don't include real faces or sensitive CNIC numbers in screenshots
- **Annotations**: You can add arrows/labels using tools like Snagit or Greenshot (optional)

---

**Status:** ⏳ Screenshots pending - please capture when convenient
**Priority:** Medium (improves documentation quality significantly)
