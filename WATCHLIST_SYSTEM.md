# Watchlist System - Criminal Detection

## ✅ What's Been Implemented (Phase 1)

### 1. Database Schema ✅
- Added watchlist fields to `Person` table:
  - `watchlist_status`: most_wanted, suspect, person_of_interest, banned, none
  - `threat_level`: critical, high, medium, low, none
  - `criminal_notes`: Details about the crime
  - `added_to_watchlist_at`: When added to watchlist
  - `watchlist_expires_at`: Optional expiry

- Added guard verification fields to `Alert` table:
  - `guard_verified`: Did guard verify this match?
  - `guard_action`: confirmed, false_alarm, investigating, apprehended, escalated
  - `guard_verified_by`: Guard username
  - `guard_verified_at`: Timestamp
  - `action_notes`: Guard's action notes
  - `threat_level`: Cached from person
  - `watchlist_status`: Cached from person

### 2. Admin Panel Enrollment ✅
- Added visual watchlist section with orange/amber theme
- Simple dropdowns for guards to select:
  - **Watchlist Status**: Most Wanted, Suspect, Person of Interest, Banned, None
  - **Threat Level**: Critical, High, Medium, Low, None
  - **Criminal Details**: Required text field for crime details

- Color-coded options:
  - 🔴 CRITICAL/MOST WANTED (Red)
  - 🟠 HIGH/SUSPECT (Orange)
  - 🟡 MEDIUM/POI (Amber)
  - 🔵 LOW (Blue)
  - ⚪ NONE (White)

### 3. Backend API ✅
- Both `/api/enroll` and `/api/enroll/multiple` now accept watchlist fields
- Automatically sets `added_to_watchlist_at` timestamp
- Alert system caches threat level for quick filtering

---

## 🎯 How Guards Will Use This

### For Admin (Adding Wanted Persons):

1. Open Admin Panel: http://192.168.0.117:8000/admin
2. Upload criminal's photo
3. Enter CNIC and Name
4. **Select Watchlist Status**: e.g., "MOST WANTED"
5. **Select Threat Level**: e.g., "CRITICAL (Armed/Dangerous)"
6. **Enter Criminal Details**: "Wanted for armed robbery. Case #ABC123. Do NOT approach alone."
7. Click "Add Wanted Person"

### For Guards (Monitoring):

**Current Workflow** (Phase 1 Complete):
1. Dashboard shows live stream
2. When criminal is detected → Alert appears
3. Alert includes threat level and watchlist status
4. Guard can see the person's details

**Next Phase** (To Be Implemented):
1. Alert pops up with **BIG RED WARNING** for high-risk persons
2. Side-by-side photo comparison (mugshot vs captured)
3. Guard clicks **"Verify Match"** button
4. If match correct → Guard clicks **"CONFIRM - Deploy Security"**
5. If false alarm → Guard clicks **"False Alarm - Ignore"**
6. System logs the guard's action with timestamp

---

## 🚧 What's Next (Remaining Work)

### Priority 1: Guard Action UI (Critical)
**File**: `app/static/dashboard.html` or create `alert_response.html`

Need to add:
```javascript
// When alert comes via WebSocket
if (alert.threat_level === 'critical' || alert.threat_level === 'high') {
    // Show BIG MODAL with:
    // - Red flashing border
    // - Side-by-side photos
    // - Threat level badge
    // - Criminal notes
    // - Action buttons:
    //   [✅ CONFIRM] [❌ FALSE ALARM] [🔍 INVESTIGATING]

    // Play alert sound
    playAlertSound(alert.threat_level);
}
```

### Priority 2: Alert Color Coding
Update `alerts.html` and `dashboard.html`:
- CRITICAL alerts → Bright red (#dc2626)
- HIGH alerts → Orange (#ea580c)
- MEDIUM alerts → Amber (#f59e0b)
- LOW alerts → Blue (#3b82f6)

### Priority 3: Guard Verification API
**File**: `app/api/routes/alerts.py`

Add endpoint:
```python
@router.post("/alerts/{alert_id}/verify")
async def verify_alert(
    alert_id: int,
    guard_action: str,  # 'confirmed', 'false_alarm', 'investigating'
    action_notes: str,
    current_user: User
):
    # Update alert with guard's decision
    # Log action timestamp
    # Optionally escalate if confirmed critical
```

### Priority 4: Audio/Visual Alerts
- Different sounds for different threat levels
- Browser notifications
- Flashing indicators

---

## 📊 Database Backup

Migration created automatic backup:
- **Backup file**: `face_recognition_backup_20251107_111856.db`
- All existing data preserved
- New columns added with safe defaults

---

## 🔧 Testing Checklist

### Test Enrollment:
1. ✅ Open admin panel: http://192.168.0.117:8000/admin
2. ✅ Check if watchlist fields appear (orange box)
3. ✅ Try enrolling a test "Most Wanted" person
4. ✅ Verify database has correct threat_level
5. ✅ Check alert system includes threat level

### Test Alert System:
1. Let system detect the enrolled person
2. Check if alert includes `threat_level` field
3. Verify WebSocket broadcast includes watchlist data
4. Dashboard should receive alert with threat info

---

## 🎨 UI Design (NRTC Theme)

Colors Being Used:
- **Primary Blues**: #124877, #0F518B
- **Gold Accent**: #cc9933
- **Alert Colors**:
  - Critical: #dc2626 (bright red)
  - High: #ea580c (orange)
  - Medium: #f59e0b (amber)
  - Low: #3b82f6 (blue)
- **Dark Background**: #0f172a, #1e293b

---

## 📝 Next Session Plan

1. **Test the current enrollment** (5 mins)
   - Add a test wanted person
   - Verify fields are saved

2. **Implement Guard Action Modal** (30-45 mins)
   - Create pop-up for high-risk alerts
   - Add action buttons
   - API endpoint for verification

3. **Color-code alerts** (15 mins)
   - Update CSS based on threat level
   - Test with different threat levels

4. **Audio alerts** (15 mins)
   - Add sound files
   - Different beeps for different levels

---

**Status**: Phase 1 Complete - Database & Enrollment Ready ✅
**Next**: Guard Verification UI (The most critical part for guards!)

---

*Last Updated: 2025-11-07*
*Migration: Successful*
*Server: Running on port 8000*
