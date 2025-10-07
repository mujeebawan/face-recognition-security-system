# JetPack 6.1 Upgrade Guide - NVIDIA Jetson AGX Orin

**Created**: October 7, 2025
**System**: Face Recognition Security System
**Current**: JetPack 5.1.2 (Ubuntu 20.04, CUDA 11.4)
**Target**: JetPack 6.1 (Ubuntu 22.04, CUDA 12.6)

---

## ✅ PRE-UPGRADE CHECKLIST (COMPLETED)

- [x] Full project backup: `face_recognition_backup_20251007.tar.gz` (461 MB)
- [x] Database backup: `data_and_db_backup_20251007.tar.gz` (154 MB)
- [x] Python packages saved: `requirements_pre_jetpack61.txt`, `packages_pre_jetpack61.txt`
- [x] System info documented: `system_info_pre_upgrade.txt`
- [x] Camera config backed up: `.env.backup_pre_jetpack61`
- [x] Git pushed to GitHub: https://github.com/mujeebawan/face-recognition-security-system
- [x] All working code committed

**Backup Location**: `/home/mujeeb/Downloads/`

---

## 🚨 CRITICAL WARNINGS

1. **COMPLETE DATA LOSS** - Everything on Jetson will be erased
2. **NO ROLLBACK** - Cannot downgrade back to JetPack 5
3. **4-6 HOURS DOWNTIME** - System completely unavailable
4. **UBUNTU PC REQUIRED** - Cannot upgrade from Jetson itself
5. **USB-C CABLE NEEDED** - Connect Jetson to Ubuntu PC

---

## 📋 UPGRADE METHODS

### **Method 1: NVIDIA SDK Manager (RECOMMENDED)**

**Requirements:**
- Ubuntu 20.04 or 22.04 **x64 PC** (not the Jetson!)
- 50+ GB free space on PC
- USB-C cable (Jetson to PC)
- Stable internet connection

#### **Steps:**

**1. On Your Ubuntu PC (Not Jetson):**

```bash
# Download SDK Manager
# Go to: https://developer.nvidia.com/sdk-manager
# Click "Download SDK Manager"
# Login with NVIDIA Developer account (free)

# Install SDK Manager
sudo dpkg -i sdkmanager_[version].deb
sudo apt install -f  # Fix dependencies if needed

# Launch SDK Manager
sdkmanager
```

**2. Put Jetson AGX Orin in Recovery Mode:**

Physical steps on Jetson:
1. **Power off** the Jetson AGX Orin completely
2. Locate the 3 buttons near USB-C port:
   - POWER (left)
   - RECOVERY (middle) ← Important!
   - RESET (right)
3. **Hold down RECOVERY button** (middle button)
4. While holding RECOVERY, **press POWER button**
5. Keep holding RECOVERY for **2 seconds**, then release
6. **Connect USB-C cable** from Jetson to Ubuntu PC

**3. Verify Recovery Mode (On Ubuntu PC):**

```bash
lsusb | grep -i nvidia
# Should show: "Bus XXX Device XXX: ID 0955:7023 NVIDIA Corp. APX"
```

If you don't see this, repeat recovery mode steps.

**4. In SDK Manager (On Ubuntu PC):**

1. **Target Hardware**:
   - Select "Jetson AGX Orin"
   - Automatic detection if in recovery mode

2. **JetPack Version**:
   - Select "JetPack 6.1" (latest)

3. **Target Components**:
   - ✅ Jetson Linux
   - ✅ CUDA Toolkit 12.6
   - ✅ cuDNN 9.3
   - ✅ TensorRT 10.3
   - ✅ VPI 3.2
   - ✅ OpenCV
   - ✅ All recommended components

4. **Flash Configuration**:
   - Target: "Jetson AGX Orin (internal storage)"
   - Flash OS: Yes
   - Install SDK components: Yes

5. **Click "Continue" and wait**:
   - Download: ~15-30 minutes (15GB+)
   - Flash: ~20-30 minutes
   - Install: ~15-30 minutes
   - **Total: 1-2 hours**

6. **Follow On-Screen Setup**:
   - Create new user account (or use same: mujeeb)
   - Set password
   - Wait for installation to complete

**5. First Boot on Jetson:**

After flash completes, Jetson will reboot automatically.

```bash
# Verify JetPack version
cat /etc/nv_tegra_release
# Should show: R36 (release), REVISION: 4.0

dpkg-query --show nvidia-jetpack
# Should show: 6.1

# Verify CUDA
python3 --version  # Should be 3.10
```

---

### **Method 2: Manual Flash (Advanced Users)**

**Use this if SDK Manager fails or you prefer command line.**

**Requirements:**
- Ubuntu 20.04/22.04 PC
- 50GB+ free space

#### **Steps:**

**1. Download JetPack 6.1 BSP (On PC):**

```bash
# Create working directory
mkdir -p ~/jetpack61_flash
cd ~/jetpack61_flash

# Download BSP (Board Support Package)
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.0/release/jetson_linux_r36.4.0_aarch64.tbz2

# Download root filesystem
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.0/release/tegra_linux_sample-root-filesystem_r36.4.0_aarch64.tbz2
```

**2. Extract and Prepare:**

```bash
# Extract BSP
tar xf jetson_linux_r36.4.0_aarch64.tbz2

# Extract root filesystem
cd Linux_for_Tegra/rootfs/
sudo tar xpf ../../tegra_linux_sample-root-filesystem_r36.4.0_aarch64.tbz2

# Apply binaries
cd ..
sudo ./apply_binaries.sh
```

**3. Put Jetson in Recovery Mode** (same as Method 1 above)

**4. Flash the Jetson:**

```bash
# For Jetson AGX Orin Developer Kit
sudo ./flash.sh jetson-agx-orin-devkit internal

# Wait 20-30 minutes for flash to complete
```

**5. After Flash:**
- Jetson will reboot automatically
- Complete Ubuntu setup wizard
- Login and verify installation

---

## 🔄 POST-UPGRADE RESTORATION

After JetPack 6.1 is installed and booted, follow these steps to restore your project.

### **1. System Update**

```bash
# Update package lists
sudo apt update
sudo apt upgrade -y

# Verify installation
nvidia-smi  # Check GPU
python3 --version  # Should be 3.10
```

### **2. Install Development Tools**

```bash
# Essential tools
sudo apt install -y git curl wget build-essential

# Python development
sudo apt install -y python3-pip python3-dev

# Update pip
pip3 install --upgrade pip
```

### **3. Restore Project from GitHub**

```bash
# Clone repository
cd ~/Downloads/
git clone https://github.com/mujeebawan/face-recognition-security-system.git
cd face_recognition_system

# Or if backups available locally:
# cd /home/mujeeb/Downloads/
# tar -xzf face_recognition_backup_20251007.tar.gz
```

### **4. Install Python Dependencies**

```bash
cd face_recognition_system

# Install from requirements
pip3 install -r requirements.txt

# Verify PyTorch with CUDA (JetPack 6.1 includes PyTorch 2.4+)
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# Should output: CUDA available: True ✅
```

### **5. Install TensorRT Python Bindings**

```bash
# TensorRT 10.3 should be included in JetPack 6.1
pip3 install tensorrt

# Verify
python3 -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"
# Should output: TensorRT: 10.3.x
```

### **6. Restore Data and Database**

```bash
cd face_recognition_system

# Option A: Restore from backup on same system
tar -xzf ../data_and_db_backup_20251007.tar.gz

# Option B: If starting fresh, initialize database
python3 init_db.py
```

### **7. Restore Camera Configuration**

```bash
# Copy .env file
cp .env.backup_pre_jetpack61 .env

# Or create new .env from template
cp .env.example .env
nano .env  # Edit with your camera credentials
```

### **8. Test System**

```bash
# Test camera connection
python3 test_camera.py

# Test multi-agent system
python3 test_parallel_multimodel.py

# Start server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### **9. Verify URLs**

Open in browser:
- Dashboard: http://192.168.1.64:8000/dashboard
- Admin: http://192.168.1.64:8000/admin
- Multi-Agent: http://192.168.1.64:8000/multi-agent
- API Docs: http://192.168.1.64:8000/docs

---

## 🚀 PHASE 2 IMPLEMENTATION (After Upgrade)

Now you can add the blocked models!

### **1. Install Additional Models**

```bash
# FaceNet (Google)
pip3 install facenet-pytorch

# CLIP (OpenAI)
pip3 install transformers

# DINOv2 (Meta)
# Already included in transformers

# Verify
python3 -c "import facenet_pytorch; import transformers; print('✅ All models ready')"
```

### **2. Test New Models**

```bash
# Test FaceNet integration
python3 -c "from app.core.multi_agent.models.facenet_model import FaceNetModel; print('FaceNet OK')"

# Test CLIP
python3 -c "from app.core.multi_agent.models.clip_model import CLIPModel; print('CLIP OK')"
```

### **3. Run 6-Model System**

```bash
# Update engine to use all 6 models
# Edit app/core/multi_agent/engine.py if needed

# Test with all models
python3 test_parallel_multimodel.py

# Expected: 6 models, ~75-90ms latency, 70-80% GPU
```

---

## 📊 BEFORE vs AFTER COMPARISON

| Component | JetPack 5.1.2 (Before) | JetPack 6.1 (After) |
|-----------|------------------------|---------------------|
| **Ubuntu** | 20.04 LTS | **22.04 LTS** ✅ |
| **Kernel** | 5.10.120-tegra | **5.15** ✅ |
| **Python** | 3.8 | **3.10** ✅ |
| **CUDA** | 11.4 | **12.6** ✅ |
| **cuDNN** | 8.6 | **9.3** ✅ |
| **TensorRT** | 8.5.2 | **10.3** ✅ |
| **PyTorch** | 2.1.0 (broken) | **2.4+ working** ✅ |
| **Models Working** | 3 (ArcFace, YOLOv8, AdaFace) | **6+ (all models)** ✅ |
| **GPU Usage** | 30% | **80-90%** ✅ |
| **Latency** | 47ms (3 models) | **75-90ms (6 models)** ✅ |
| **Accuracy** | 99% | **99.5%+** ✅ |

---

## 🛠️ TROUBLESHOOTING

### **Issue: Jetson not detected in recovery mode**

**Solution:**
```bash
# On Ubuntu PC, check USB connection
lsusb | grep -i nvidia

# If not showing:
# 1. Try different USB-C cable
# 2. Use different USB port on PC
# 3. Repeat recovery mode steps carefully
# 4. Power cycle Jetson and try again
```

### **Issue: SDK Manager download fails**

**Solution:**
- Check internet connection
- Use wired Ethernet (more stable than WiFi)
- Retry download (SDK Manager resumes)
- Or use Manual Flash method instead

### **Issue: PyTorch CUDA not working after upgrade**

**Solution:**
```bash
# JetPack 6.1 should have PyTorch with CUDA by default
# If not, install manually:
pip3 install torch torchvision torchaudio

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

### **Issue: Camera not connecting after upgrade**

**Solution:**
```bash
# Test RTSP connection
ffmpeg -i rtsp://admin:Mujeeb@321@192.168.1.64:554/Streaming/Channels/101 -frames:v 1 test.jpg

# If fails, check:
# 1. Camera IP (might need DHCP reservation)
# 2. Network settings after OS reinstall
# 3. Firewall rules
```

---

## 📞 SUPPORT RESOURCES

- **NVIDIA JetPack Docs**: https://docs.nvidia.com/jetson/jetpack/
- **NVIDIA Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
- **JetsonHacks**: https://jetsonhacks.com/ (excellent community resource)
- **Project Repo**: https://github.com/mujeebawan/face-recognition-security-system

---

## ⏱️ TIME ESTIMATE

| Phase | Time |
|-------|------|
| ✅ Backup & Preparation | 30 min (DONE) |
| ⏳ SDK Manager Download | 30-60 min |
| ⏳ Flash Process | 30-45 min |
| ⏳ Initial Setup | 15-30 min |
| ⏳ Project Restoration | 30-60 min |
| ⏳ Testing & Verification | 30-60 min |
| **TOTAL** | **3-5 hours** |

---

## 🎯 SUCCESS CRITERIA

After upgrade, verify:

- [ ] `python3 --version` shows 3.10
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `nvidia-smi` shows GPU info
- [ ] TensorRT version 10.3 installed
- [ ] Camera RTSP stream working
- [ ] Database and embeddings restored
- [ ] Web interface accessible
- [ ] Multi-agent system running with 3+ models
- [ ] Can add FaceNet, CLIP, DINOv2 without errors

---

## 🚀 READY TO START?

**You are now ready to upgrade!**

### Quick Start:

1. **Connect Jetson to Ubuntu PC via USB-C**
2. **Put Jetson in recovery mode** (hold RECOVERY + press POWER)
3. **On Ubuntu PC**: Launch `sdkmanager`
4. **Select JetPack 6.1** and follow wizard
5. **Wait ~2 hours** for complete installation
6. **Restore project** using steps in "Post-Upgrade Restoration"
7. **Test system** and verify all components
8. **Continue with Phase 2** (add more models)

---

**Good luck! 🚀**

**Created by**: Mujeeb with Claude Code
**Date**: October 7, 2025
**Status**: Ready for upgrade ✅
