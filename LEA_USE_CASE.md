# Law Enforcement Agency (LEA) Use Case - Wanted Persons Detection

**Created**: October 3, 2025
**Project Purpose**: Detection of wanted persons at airports, toll plazas, and public areas

---

## 🎯 Primary Use Case

### Target Users
- **Law Enforcement Agencies (LEAs)**
- Airport security
- Toll plaza security
- Public area surveillance

### Core Functionality
**Detect wanted persons from NADRA database and alert authorities in real-time**

---

## 📋 System Requirements (LEA Perspective)

### 1. Wanted Persons Database

**Data Source**: NADRA (National Database and Registration Authority)
- **One photo per person** (NADRA official photo)
- **CNIC** (National ID Card Number) - Unique identifier
- **Name** - Person's full name
- **Case details** - Why they're wanted (optional metadata)

**Admin Operations**:
- ✅ **Add wanted person** - Upload NADRA photo, CNIC, name, case details
- ✅ **Remove wanted person** - Clear from system when case resolved
- ✅ **Update wanted person** - Modify details or add better photos
- ✅ **Search/filter** - Find specific wanted persons in database

### 2. Detection & Alert System

**Real-time Detection**:
- System continuously monitors camera feed
- Matches faces against wanted persons database
- Triggers alert when wanted person detected

**Alert Requirements**:
- ⚠️ **Instant notification** - Alert appears within 1-2 seconds of detection
- 📸 **Capture evidence** - Take photo of detected person
- 📊 **Alert details** - Show CNIC, name, confidence score, timestamp
- 🔔 **Multiple channels** - Dashboard, webhook, email (future)

### 3. Image Storage Optimization

**Critical Requirement**: Don't store too many images unnecessarily

**Current Strategy**:
- ✅ **Cooldown period**: 60 seconds between alerts for same person
  - Prevents spam when person is moving/standing in frame
  - Only ONE snapshot per person per minute

**Storage Logic**:
```
Person detected → Check last alert time
  - If < 60 seconds ago: Skip (no new alert, no new photo)
  - If ≥ 60 seconds ago: Create alert + take snapshot
```

**File Naming**: `alert_{alert_id}_{timestamp}.jpg`
- Example: `alert_42_20251003_143025.jpg`
- Stored in: `data/alert_snapshots/`

**Storage Management** (Future Phase):
- Auto-delete snapshots older than 30/60/90 days
- Compress old images
- Archive to external storage

### 4. Speed & Performance

**Detection Speed Requirements**:
- ✅ **Fast detection**: MediaPipe (5-10ms per frame)
- ✅ **Recognition frequency**: Every 15 frames (~1 second at 15 FPS)
- ✅ **Frame skipping**: Process every 2nd frame to maintain speed
- ✅ **Caching**: Person info cached in memory (no DB query per frame)

**Current Performance**:
- Live stream: 10-15 FPS
- Detection latency: <100ms
- Recognition latency: 300-400ms (CPU-only)
- Alert delivery: <100ms via WebSocket

**For High-Traffic Areas** (Airport/Toll Plaza):
- Multiple cameras supported (separate instances)
- Scalable to thousands of wanted persons
- Future: GPU acceleration for faster recognition

---

## 🏗️ Current System Status (LEA Perspective)

### ✅ What's Already Built

1. **Person Enrollment System**
   - Upload NADRA photo via API or web interface
   - Store CNIC as unique identifier
   - Multiple photos per person for better accuracy
   - Auto-augmentation for training variations

2. **Real-time Detection**
   - Live MJPEG stream with face detection overlay
   - Continuous matching against database
   - Known vs Unknown person classification

3. **Alert System**
   - Database-backed alerts (SQLite)
   - Snapshot capture on detection
   - 60-second cooldown to prevent spam
   - WebSocket real-time delivery to dashboard
   - Alert acknowledgment system

4. **Dashboard**
   - Live video stream
   - Real-time alert feed (no refresh needed)
   - Statistics (24-hour alerts, known/unknown counts)
   - Connection status indicators

### ⚠️ Current Configuration Issue

**IMPORTANT**: System is currently configured **BACKWARDS** for LEA use!

**Current Config** (app/core/alerts.py:39-40):
```python
"alert_on_unknown": False,  # Disabled
"alert_on_known": True,     # Enabled - ALERTS ON WANTED PERSONS ✅
```

This is actually **CORRECT** for LEA use case:
- "Known persons" = Wanted persons in database
- Alert triggers when wanted person detected ✅

**Terminology Clarification for LEA**:
- **"Known person"** = Person in database = **WANTED PERSON** ⚠️
- **"Unknown person"** = Not in database = **Regular citizen** (ignore)

---

## 📊 Dashboard Interpretation (LEA Context)

### Current Stats Labels:

| Dashboard Label | LEA Meaning |
|-----------------|-------------|
| **Total Alerts (24h)** | Total wanted person detections in last 24 hours |
| **Known Person Alerts** | Wanted persons detected (THIS IS WHAT MATTERS) ⚠️ |
| **Unknown Person Alerts** | Random people detected (should be 0 with current config) |
| **Live Viewers** | Number of security officers monitoring dashboard |

### Alert Display:

**When wanted person detected**:
```
✓ KNOWN PERSON           12:30:45 PM
Name: Abdul Rasheed
Confidence: 87.3%
```

**LEA Translation**:
- ✓ KNOWN PERSON = ⚠️ WANTED PERSON DETECTED
- Name = Name from NADRA database
- Confidence = Match accuracy (higher = more certain)

---

## 🚀 What Needs to Be Built (LEA Requirements)

### Phase A: Admin Interface for Wanted Persons

**Priority: HIGH**

**Features Needed**:
1. **Web-based admin panel** (`/admin` route)
   - Login/authentication for LEA officers
   - Add wanted person form:
     - Upload NADRA photo
     - Enter CNIC (auto-validation)
     - Enter name
     - Enter case details/notes
   - View all wanted persons (searchable table)
   - Remove wanted person (case cleared)
   - Update wanted person details

2. **API Endpoints** (partially exists):
   - ✅ `POST /api/enroll` - Add person (already exists)
   - ✅ `GET /api/persons` - List all persons (already exists)
   - ✅ `DELETE /api/persons/{id}` - Remove person (already exists)
   - ❌ `PUT /api/persons/{id}` - Update person (NEEDS BUILDING)
   - ❌ `GET /api/persons/{cnic}` - Search by CNIC (NEEDS BUILDING)

3. **Better Terminology**:
   - Rename "Person" → "Wanted Person"
   - Rename "Enrollment" → "Add to Watch List"
   - Rename "Known Person Alert" → "Wanted Person Detected"

### Phase B: Enhanced Alert System

**Priority: MEDIUM**

**Features Needed**:
1. **Alert priority levels**:
   - High priority (dangerous criminals)
   - Medium priority (theft, fraud)
   - Low priority (minor offenses)

2. **Audio alerts**:
   - Play sound when high-priority person detected
   - Different sounds for different priority levels

3. **Alert acknowledgment workflow**:
   - Officer marks alert as "Reviewed"
   - Officer marks alert as "Action Taken" (arrested, etc.)
   - Add notes to alert

4. **Multi-camera support**:
   - Multiple camera feeds on single dashboard
   - Alerts show which camera detected person

### Phase C: Reporting & Analytics

**Priority: MEDIUM**

**Features Needed**:
1. **Detection reports**:
   - Daily/weekly/monthly detection summaries
   - Most frequently detected persons
   - Peak detection times
   - Detection locations (if multiple cameras)

2. **Export capabilities**:
   - Export alert history to PDF/Excel
   - Include snapshots in report
   - Filter by date range, person, camera

3. **Statistics dashboard**:
   - Total wanted persons in database
   - Total detections this month
   - Average detection time
   - False positive rate (if known)

### Phase D: Image Storage Management

**Priority: LOW (Future)**

**Features Needed**:
1. **Automatic cleanup**:
   - Delete snapshots older than configured period (30/60/90 days)
   - Keep only first and last snapshot per person per day
   - Archive to cloud storage (AWS S3, etc.)

2. **Storage quotas**:
   - Alert when storage reaches 80% capacity
   - Automatic compression of old images
   - Prioritize high-priority alerts for retention

3. **Snapshot review**:
   - View all snapshots for specific person
   - Compare snapshots over time
   - Mark snapshots for evidence preservation

---

## 🔧 Current System Configuration (LEA Optimized)

### Alert Settings (app/core/alerts.py)

```python
config = {
    "enabled": True,
    "alert_on_unknown": False,      # Don't alert on random people
    "alert_on_known": True,         # DO alert on wanted persons ✅
    "min_confidence_unknown": 0.5,
    "cooldown_seconds": 60,         # Only one alert per minute per person
    "webhook_url": None,            # Future: integrate with LEA system
    "email_recipients": [],         # Future: email to duty officer
    "save_snapshot": True,          # Always save evidence photo ✅
}
```

### Recognition Settings (app/config.py)

```python
face_recognition_threshold = 0.6    # 60% similarity to match
```

**Threshold Tuning for LEA**:
- **Lower threshold (0.5-0.55)**: More detections, more false positives
- **Higher threshold (0.65-0.7)**: Fewer false positives, might miss some
- **Recommended**: 0.6 (current) - balanced approach
- **Can be adjusted** based on field testing results

---

## 🎯 Deployment Scenarios

### Scenario 1: Airport Security Checkpoint

**Setup**:
- 2-4 cameras at immigration/security checkpoints
- Central monitoring room with dashboard
- Database of 1000-5000 wanted persons

**Workflow**:
1. Passenger walks through checkpoint
2. Camera captures face
3. System matches against wanted persons database
4. If match: Alert appears on officer's dashboard
5. Officer reviews alert, verifies identity
6. Officer takes appropriate action (detain, verify further)

**Performance Requirements**:
- Fast detection (< 2 seconds)
- High accuracy (low false positives)
- Reliable alerting (no missed alerts)

### Scenario 2: Toll Plaza

**Setup**:
- 1-2 cameras per toll booth
- Central dashboard for security office
- Database of 500-2000 wanted persons

**Workflow**:
1. Vehicle approaches toll booth
2. Driver's face captured by camera
3. System matches against database
4. If match: Alert sent to security office
5. Security notifies toll booth operator
6. Vehicle directed to holding area for verification

**Performance Requirements**:
- Very fast detection (< 1 second) - vehicle is moving
- Must handle varying lighting conditions (day/night)
- Must handle faces at angles (driver looking forward)

### Scenario 3: Public Area Surveillance

**Setup**:
- Multiple cameras covering high-traffic areas
- Security control room with multiple monitors
- Database of 2000-10000 wanted persons

**Workflow**:
1. Continuous monitoring of public areas
2. System processes all detected faces
3. Alerts only for wanted persons
4. Security dispatches officers to location
5. Officers verify and apprehend if confirmed

**Performance Requirements**:
- Handle multiple faces per frame
- Scalable to large databases (10,000+ persons)
- Low false positive rate (to avoid alert fatigue)

---

## 📝 Terminology Guide (LEA Context)

| Technical Term | LEA Equivalent |
|----------------|----------------|
| Face Recognition | Wanted Person Identification |
| Person Enrollment | Adding to Watch List |
| Known Person | Wanted Person ⚠️ |
| Unknown Person | Regular Citizen / Not in Database |
| Recognition Log | Detection Record |
| Confidence Score | Match Certainty |
| Alert | Wanted Person Detection Alert |
| Snapshot | Evidence Photo |
| Embedding | Facial Signature (AI representation) |

---

## 🔐 Security & Privacy Considerations

### Data Protection:
- Wanted persons database must be encrypted
- Access control - only authorized LEA officers
- Audit log - track who adds/removes wanted persons
- Snapshot retention policy - comply with data laws

### False Positives:
- System should clearly show confidence score
- Officer must visually verify before taking action
- Keep logs of false positives for system improvement

### Legal Compliance:
- Follow national data protection laws
- Obtain proper warrants/legal authority for wanted persons
- Regular audit of database (remove cleared cases)
- Transparent system - notify authorities of automated surveillance

---

## 📅 Recommended Next Steps

### Immediate (This Session):
1. ✅ Fix WebSocket real-time alerts (DONE)
2. ✅ Update documentation with LEA use case (DONE)
3. Test alert system with Mujeeb/Safyan (wanted person simulation)
4. Commit all fixes

### Short-term (Next 1-2 Sessions):
1. Build admin interface for wanted persons management
2. Improve dashboard labels (Known Person → Wanted Person)
3. Add search by CNIC functionality
4. Add person update API endpoint

### Medium-term (Next Phase):
1. Multi-camera support
2. Alert priority levels
3. Audio alerts for high-priority detections
4. Basic reporting (daily detection summary)

### Long-term (Production):
1. PostgreSQL migration (handle 10,000+ wanted persons)
2. GPU acceleration (faster detection)
3. Cloud deployment (AWS/Azure)
4. Integration with existing LEA systems
5. Mobile app for officers

---

**This document should be read at the start of every session to understand the LEA context!**
