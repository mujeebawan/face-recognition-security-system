# Documentation Maintenance Guide
**For Claude Code - Auto-Follow Every Session**

---

## 📂 Documentation Structure (FINAL)

### **Core Documentation (6 files - ALWAYS MAINTAIN)**

```
face_recognition_system/
├── README.md                           # Project overview (for GitHub)
├── PROJECT_PLAN.md                     # Master plan with all phases
├── DEVELOPMENT_LOG.md                  # Session-by-session detailed log
├── CURRENT_STATUS.md                   # Single source of truth (where we are NOW)
├── LEA_USE_CASE.md                     # Real-world deployment scenarios
├── TECHNOLOGY_STACK.md                 # Tech choices + justifications
│
├── PROJECT_PRESENTATION_SUMMARY.md     # For professors (full presentation)
├── QUICK_SUMMARY.md                    # For professors (one-page)
└── presentation.html                   # GitHub-styled HTML version
```

### **Archive (Old Files - Don't Touch)**
```
archive_old_docs/
├── SESSION_6_SUMMARY.md               # Merged into DEVELOPMENT_LOG.md
├── UPDATE_SUMMARY.md                  # Merged into DEVELOPMENT_LOG.md
├── PROJECT_STATUS.md                  # Replaced by CURRENT_STATUS.md
├── NEXT_PHASE_PLAN.md                 # Merged into PROJECT_PLAN.md
└── ... (other old files)
```

---

## 🤖 Claude's Responsibilities (Auto-Execute Each Session)

### **1. SESSION START (First thing every new session)**

**Read these 3 files in order:**
```python
1. Read: CURRENT_STATUS.md          # Know where we are
2. Read: PROJECT_PLAN.md            # Understand current phase
3. Read: DEVELOPMENT_LOG.md         # Last session context (read last 100 lines)
```

**Then tell user:**
```
"I've read the project status. We're at [PHASE], last session we [SUMMARY].
Ready to continue with [NEXT_TASK]."
```

---

### **2. DURING WORK (When things change)**

#### **When Phase Status Changes:**
```
IF phase starts/completes:
  → Update PROJECT_PLAN.md
  → Change status: ⏳ PENDING → 🚧 IN PROGRESS → ✅ COMPLETE
  → Add date completed
```

#### **When Performance Metrics Change:**
```
IF new benchmark/test results:
  → Update CURRENT_STATUS.md (Performance section)
  → Update numbers (latency, accuracy, GPU usage)
```

#### **When Architecture Changes:**
```
IF new models added / architecture modified:
  → Update CURRENT_STATUS.md (Architecture diagram)
  → Update PROJECT_PLAN.md (phase deliverables)
```

#### **When Tech Stack Changes:**
```
IF new library installed / version updated:
  → Update TECHNOLOGY_STACK.md
  → Document why (justification)
```

---

### **3. SESSION END (Before user says goodbye)**

#### **Add Entry to DEVELOPMENT_LOG.md:**
```markdown
## Session X: [Title]
**Date**: [Date]
**Duration**: ~X hours
**Status**: [Phase Status]

### What We Did:
1. [Achievement 1]
2. [Achievement 2]

### Files Created/Modified:
- [file1]
- [file2]

### Performance:
- [Metric]: [Value]

### Next Session:
- [Task 1]
- [Task 2]
```

#### **Update CURRENT_STATUS.md:**
```
- Update "Last Updated" date
- Update performance metrics
- Update "What's Working" section
- Update "What's Next" section
```

#### **Update PROJECT_PLAN.md:**
```
- Update phase status (if changed)
- Mark deliverables complete (✅)
```

---

## ❌ What NOT to Do

### **NEVER Create New Doc Files Unless:**
1. User explicitly requests it
2. It's a completely new major feature (rare)
3. It's for presentation/external use

### **NEVER Duplicate Information:**
- ❌ Don't copy-paste same content across files
- ✅ One piece of info = one location
- ✅ Use references: "See PROJECT_PLAN.md Phase 7"

### **NEVER Leave Outdated Info:**
- ❌ Don't keep old performance numbers
- ❌ Don't keep old architecture diagrams
- ✅ Update existing content
- ✅ Archive old files to `archive_old_docs/`

---

## ✅ What TO Do

### **Keep It Clean:**
```
IF user says "document this":
  → Add to existing doc (CURRENT_STATUS.md or PROJECT_PLAN.md)
  → Don't create new file

IF user says "we changed X":
  → Update relevant section in existing doc
  → Mark old content as [DEPRECATED] if needed

IF user says "where are we?":
  → Read CURRENT_STATUS.md
  → Give clear summary
```

### **Keep It Accurate:**
```
EVERY session end:
  → Update metrics (accuracy, latency, GPU usage)
  → Update phase status
  → Update "last updated" dates
  → Commit changes to git
```

### **Keep It Useful:**
```
Write docs as if you're talking to:
  - Future Claude (next session - needs context)
  - The user (needs to know progress)
  - Professors (needs to understand choices)
```

---

## 📋 Session Checklist (Claude Auto-Follow)

### **START OF SESSION:**
- [ ] Read CURRENT_STATUS.md
- [ ] Read PROJECT_PLAN.md (current phase section)
- [ ] Read DEVELOPMENT_LOG.md (last entry)
- [ ] Tell user: "Ready, we're at [PHASE]"

### **DURING SESSION:**
- [ ] Update docs when things change (not at end!)
- [ ] Use existing files (PROJECT_PLAN.md, CURRENT_STATUS.md)
- [ ] NO new MD files unless necessary

### **END OF SESSION:**
- [ ] Add session entry to DEVELOPMENT_LOG.md
- [ ] Update CURRENT_STATUS.md (metrics, status, next steps)
- [ ] Update PROJECT_PLAN.md (phase status if changed)
- [ ] Git commit all docs

---

## 🎯 Single Source of Truth Rules

### **Where Information Lives:**

| Information | Primary Location | Also Mentioned In |
|-------------|------------------|-------------------|
| **Where we are now** | CURRENT_STATUS.md | - |
| **What we're building** | PROJECT_PLAN.md | README.md (brief) |
| **What we did** | DEVELOPMENT_LOG.md | - |
| **Why this tech** | TECHNOLOGY_STACK.md | - |
| **Real-world use** | LEA_USE_CASE.md | - |
| **Performance metrics** | CURRENT_STATUS.md | - |
| **Architecture diagrams** | CURRENT_STATUS.md | PROJECT_PLAN.md |
| **Next steps** | CURRENT_STATUS.md | PROJECT_PLAN.md |

### **Update Frequency:**

| File | Update Frequency |
|------|------------------|
| **CURRENT_STATUS.md** | Every session (start + end) |
| **PROJECT_PLAN.md** | When phase changes |
| **DEVELOPMENT_LOG.md** | Every session (end) |
| **README.md** | Rarely (major milestones) |
| **TECHNOLOGY_STACK.md** | When stack changes |
| **LEA_USE_CASE.md** | Rarely (requirements change) |

---

## 💡 Examples

### **Good Documentation Update:**
```
User: "We added FaceNet model, it runs in 25ms"

Claude:
1. Updates CURRENT_STATUS.md:
   - Add FaceNet to "What's Working" section
   - Update performance: "4 models, 60ms total"
   - Update GPU usage: "40%"

2. Updates PROJECT_PLAN.md:
   - Mark "Add FaceNet" as ✅ COMPLETE

3. NO new file created ✅
```

### **Bad Documentation Update:**
```
User: "We added FaceNet model, it runs in 25ms"

Claude:
1. Creates: FACENET_IMPLEMENTATION.md ❌
2. Creates: SESSION_9_FACENET_NOTES.md ❌
3. Doesn't update CURRENT_STATUS.md ❌
4. Leaves old metrics in docs ❌

Result: Documentation chaos! ❌
```

---

## 🚀 Multi-Agent Cascade System - Documentation Notes

### **Current Status (Session 8):**
- **Phase 1**: ✅ COMPLETE (Infrastructure + 3 models)
- **Phase 2**: ⏳ NEXT (Cascade logic + 3-5 more models)

### **Where It's Documented:**
1. **PROJECT_PLAN.md** → Phase 7 section (complete plan)
2. **CURRENT_STATUS.md** → Current performance + next steps
3. **DEVELOPMENT_LOG.md** → Session 8 entry (what we built)

### **What to Update When Phase 2 Starts:**
- [ ] CURRENT_STATUS.md: "Current Phase" → "Multi-Agent Phase 2"
- [ ] PROJECT_PLAN.md: Phase 7 status → "🚧 IN PROGRESS (Phase 2)"
- [ ] Start new DEVELOPMENT_LOG.md entry for Session 9

---

## 📞 Quick Reference

**User asks: "Where are we?"**
→ Read: CURRENT_STATUS.md

**User asks: "What's the plan?"**
→ Read: PROJECT_PLAN.md (Phase 7)

**User asks: "What did we do last time?"**
→ Read: DEVELOPMENT_LOG.md (last entry)

**User asks: "Why this tech choice?"**
→ Read: TECHNOLOGY_STACK.md

**User says: "We're done for today"**
→ Update: DEVELOPMENT_LOG.md, CURRENT_STATUS.md, PROJECT_PLAN.md
→ Git commit

---

## 🎓 Philosophy

### **Documentation is for:**
1. **Future Claude** (next session context)
2. **The User** (track progress, understand status)
3. **Professors** (explain choices, show rigor)

### **Documentation is NOT for:**
1. ❌ Creating files to feel productive
2. ❌ Duplicating information
3. ❌ Outdated metrics/plans

### **Golden Rule:**
**"Update existing docs, don't create new ones"**

---

**This guide is for Claude Code to follow automatically every session.**
**User doesn't need to remind Claude - Claude reads this file at session start.**

**Last Updated**: October 7, 2025
**Status**: Final documentation structure established
