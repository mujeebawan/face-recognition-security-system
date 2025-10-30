# Phase 5 Issues - Ready to Create

Copy-paste these directly into GitHub issues. Go to:
https://github.com/mujeebawan/face-recognition-security-system/issues/new

---

## Issue #1: Dependencies Setup

**Title:**
```
Phase 5.1: Create requirements-genai.txt for Stable Diffusion dependencies
```

**Labels:** `phase-5` `feature` `dependencies`

**Assignee:** @mujeebawan

**Milestone:** Phase 5 - AI Augmentation

**Description:**
```markdown
## 🎯 Objective
Create a separate requirements file containing all dependencies needed for Stable Diffusion 1.5 and ControlNet data augmentation on Jetson AGX Orin.

## 📋 Tasks
- [ ] Research Stable Diffusion 1.5 dependencies compatible with Jetson ARM64
- [ ] Identify ControlNet dependencies and versions
- [ ] Check PyTorch version compatibility (current: from JetPack 6.1)
- [ ] Test installation on Jetson AGX Orin hardware
- [ ] Document platform-specific build requirements (if any)
- [ ] Decide: separate file vs add to main requirements.txt

## 📦 Dependencies to Include

### Core SD Dependencies
```txt
diffusers>=0.21.0           # Hugging Face Diffusers library
transformers>=4.30.0        # Transformer models (CLIP, etc.)
accelerate>=0.20.0          # Memory-efficient model loading
safetensors>=0.3.1          # Safe tensor storage format
```

### ControlNet Dependencies
```txt
controlnet-aux>=0.0.6       # ControlNet preprocessing utilities
opencv-python>=4.8.0        # Image processing (may already exist)
einops>=0.6.1               # Tensor operations
```

### Optional/Utility
```txt
omegaconf>=2.3.0           # Configuration management
xformers>=0.0.20           # Memory-efficient attention (if available for ARM64)
```

## 🔧 Jetson-Specific Considerations

**ARM64 Compatibility:**
- Check if PyPI has ARM64 wheels for each package
- If not available, may need to build from source
- Document build process in README

**CUDA Compatibility:**
- Current: CUDA 12.2 (JetPack 6.1)
- Verify all packages support CUDA 12.x

**Memory Constraints:**
- Target GPU: 8GB VRAM available
- May need to use model quantization (FP16)
- Test memory usage during installation

**Known Issues:**
- `xformers` may not have ARM64 wheels (check alternatives)
- Some packages may require compilation (can take 30+ minutes)

## ✅ Acceptance Criteria
- [ ] File `requirements-genai.txt` created in project root
- [ ] All dependencies listed with pinned versions
- [ ] Successfully installed on Jetson AGX Orin without errors
- [ ] Installation documented in README (new section: "AI Augmentation Setup")
- [ ] GPU memory usage verified (<8GB)
- [ ] Import test successful: `python3 -c "from diffusers import StableDiffusionPipeline"`

## 📝 Documentation Updates Needed
Update README.md with new section:
```markdown
### AI Augmentation Setup (Optional)

For Stable Diffusion-based data augmentation:

\`\`\`bash
pip install -r requirements-genai.txt
\`\`\`

Note: This installation may take 30-60 minutes on Jetson due to source compilation.
```

## ⏱️ Estimated Time
**3-4 hours** (including testing and documentation)

## 🔗 Related
- Part of **Phase 5: AI-Powered Data Augmentation**
- Blocks: Issues #2, #3, #4, #5
- See: PROJECT_PLAN.md for full roadmap

## 👥 Assignment
**Primary**: @mujeebawan
**Reviewer**: @AsjalAlvi1 or @muhammadmahadazher

## 📌 Priority
**High** - Blocks all other Phase 5 work
```

---

## Issue #2: Core Module Implementation

**Title:**
```
Phase 5.2: Implement generative_augmentation.py module
```

**Labels:** `phase-5` `feature` `backend` `ai-models`

**Assignee:** @mujeebawan

**Milestone:** Phase 5 - AI Augmentation

**Description:**
```markdown
## 🎯 Objective
Create the core generative augmentation module using Stable Diffusion 1.5 and ControlNet for generating multiple face angles from a single input image.

## 📋 Tasks
- [ ] Implement `GenerativeAugmentor` class in `app/core/generative_augmentation.py`
- [ ] Initialize Stable Diffusion 1.5 pipeline with FP16 optimization
- [ ] Integrate ControlNet for pose-guided generation
- [ ] Implement face angle generation (5 poses: frontal, left, right, up, down)
- [ ] Add quality assessment for generated images (blur detection, face confidence)
- [ ] Optimize memory usage for Jetson (<8GB GPU RAM)
- [ ] Add comprehensive error handling and logging
- [ ] Write unit tests for the module

## 🏗️ Module Structure

```python
# app/core/generative_augmentation.py

class GenerativeAugmentor:
    def __init__(self,
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 use_fp16: bool = True,
                 device: str = "cuda"):
        \"\"\"Initialize SD pipeline with Jetson optimizations\"\"\"

    def generate_angles(self,
                       image: np.ndarray,
                       num_angles: int = 5) -> List[np.ndarray]:
        \"\"\"Generate multiple face angles from input image\"\"\"

    def assess_quality(self, image: np.ndarray) -> float:
        \"\"\"Assess generated image quality (0-1 score)\"\"\"

    def cleanup(self):
        \"\"\"Release GPU memory\"\"\"
```

## 🎨 Generation Strategy

**5 Angle Poses**:
1. **Frontal**: Face looking straight (baseline)
2. **Left Profile**: Face turned 45° left
3. **Right Profile**: Face turned 45° right
4. **Looking Up**: Face tilted upward 30°
5. **Looking Down**: Face tilted downward 30°

**Quality Thresholds**:
- Face confidence: >0.7 (using SCRFD detector)
- Blur score: <0.3 (using Laplacian variance)
- Brightness: 50-200 (average pixel value)
- Resolution: Maintain 512x512 minimum

## ⚡ Jetson Optimizations

**Memory Management**:
```python
# FP16 precision
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# Attention slicing (reduces VRAM)
pipe.enable_attention_slicing()

# Sequential CPU offload (if needed)
pipe.enable_sequential_cpu_offload()
```

**Performance Targets**:
- Model loading: <15s
- Generation per image: <2s
- Total for 5 angles: <10s
- GPU memory usage: <8GB

## 🧪 Testing Requirements

Create `tests/unit/test_generative_augmentation.py`:

```python
def test_pipeline_initialization():
    \"\"\"Test SD pipeline loads successfully\"\"\"

def test_generate_single_angle():
    \"\"\"Test generating one face angle\"\"\"

def test_generate_all_angles():
    \"\"\"Test generating all 5 angles\"\"\"

def test_quality_assessment():
    \"\"\"Test quality scoring works\"\"\"

def test_memory_cleanup():
    \"\"\"Test GPU memory is released\"\"\"
```

## 📊 Success Metrics

**Functional**:
- [ ] Generates 5 distinct face angles
- [ ] Output images have consistent face identity
- [ ] Quality filtering removes poor generations

**Performance**:
- [ ] Generation time: <2s per image
- [ ] GPU memory: <8GB during generation
- [ ] CPU memory: <4GB

**Quality**:
- [ ] Generated faces recognized by ArcFace (>0.8 similarity to input)
- [ ] No artifacts or distortions
- [ ] Natural pose transitions

## 📝 Documentation

Add docstrings following this format:
```python
def generate_angles(self, image: np.ndarray, num_angles: int = 5) -> List[np.ndarray]:
    \"\"\"
    Generate multiple face angles using Stable Diffusion + ControlNet.

    Args:
        image: Input face image (BGR format, any size)
        num_angles: Number of angles to generate (1-10, default 5)

    Returns:
        List of generated images (BGR format, 512x512)

    Raises:
        ValueError: If no face detected in input image
        RuntimeError: If SD pipeline fails

    Example:
        >>> augmentor = GenerativeAugmentor()
        >>> angles = augmentor.generate_angles(face_image, num_angles=5)
        >>> print(f"Generated {len(angles)} face angles")
    \"\"\"
```

## ⏱️ Estimated Time
**8-10 hours** (2-3 days of development)

## 🔗 Dependencies
- **Blocks**: Issues #3, #4
- **Depends on**: Issue #1 (requirements-genai.txt must be completed first)

## 👥 Assignment
**Primary**: @mujeebawan
**Code Review**: @muhammadmahadazher
**Testing**: @AsjalAlvi1

## 📌 Priority
**High** - Core functionality for Phase 5
```

---

## Issue #3: API Integration

**Title:**
```
Phase 5.3: Integrate SD augmentation with enrollment API endpoints
```

**Labels:** `phase-5` `feature` `backend` `api`

**Assignee:** @muhammadmahadazher

**Milestone:** Phase 5 - AI Augmentation

**Description:**
```markdown
## 🎯 Objective
Integrate the generative augmentation module with existing enrollment API endpoints to enable SD-based face angle generation during person enrollment.

## 📋 Tasks
- [ ] Add SD augmentation to `/api/enroll` endpoint (single image)
- [ ] Add SD augmentation to `/api/enroll/multiple` endpoint (multiple images)
- [ ] Add new parameter: `use_sd_augmentation` (boolean, default: false)
- [ ] Update response schema to include generated image count
- [ ] Add error handling for SD pipeline failures (graceful degradation)
- [ ] Update API documentation (Swagger/OpenAPI)
- [ ] Write integration tests
- [ ] Performance testing and benchmarking

## 🔌 API Endpoint Changes

### `/api/enroll` (Single Image Enrollment)

**Request (Updated)**:
```python
POST /api/enroll
Content-Type: multipart/form-data

name: string (required)
cnic: string (required)
file: File (required)
use_augmentation: boolean (default: true)  # Existing
use_sd_augmentation: boolean (default: false)  # NEW
sd_num_angles: int (default: 5, range: 1-10)  # NEW
```

**Response (Updated)**:
```json
{
  "success": true,
  "message": "Successfully enrolled John Doe",
  "person_id": 42,
  "embeddings_created": 15,
  "augmentation_breakdown": {
    "original": 1,
    "traditional": 9,
    "stable_diffusion": 5
  },
  "sd_generation_time": 8.5,
  "quality_scores": {
    "average": 0.92,
    "min": 0.85,
    "max": 0.97
  }
}
```

### `/api/enroll/multiple` (Multi-Image Enrollment)

**Request (Updated)**:
```python
POST /api/enroll/multiple
Content-Type: multipart/form-data

name: string (required)
cnic: string (required)
files: File[] (required, 1-10 images)
use_augmentation: boolean (default: true)  # Existing
use_sd_augmentation: boolean (default: false)  # NEW
sd_num_angles: int (default: 3, range: 1-5)  # NEW per image
```

## 🏗️ Implementation Plan

### 1. Update Enrollment Logic

```python
# app/api/routes/recognition.py

from app.core.generative_augmentation import GenerativeAugmentor

# Initialize augmentor (singleton)
sd_augmentor: Optional[GenerativeAugmentor] = None

def get_sd_augmentor() -> GenerativeAugmentor:
    global sd_augmentor
    if sd_augmentor is None:
        logger.info("Initializing Stable Diffusion augmentor...")
        sd_augmentor = GenerativeAugmentor()
    return sd_augmentor

@router.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    cnic: str = Form(...),
    file: UploadFile = File(...),
    use_augmentation: bool = Form(True),
    use_sd_augmentation: bool = Form(False),  # NEW
    sd_num_angles: int = Form(5),  # NEW
    db: Session = Depends(get_db)
):
    # ... existing validation ...

    all_embeddings = []

    # 1. Extract embedding from original
    original_embedding = recognizer.extract_embedding(image)
    all_embeddings.append(original_embedding)

    # 2. Traditional augmentation
    if use_augmentation:
        variations = augmentor.generate_variations(image, num_variations=5)
        for var in variations:
            emb = recognizer.extract_embedding(var)
            if emb:
                all_embeddings.append(emb)

    # 3. Stable Diffusion augmentation (NEW)
    sd_images = []
    sd_generation_time = 0
    if use_sd_augmentation:
        try:
            start_time = time.time()
            sd_augmentor = get_sd_augmentor()
            sd_images = sd_augmentor.generate_angles(image, num_angles=sd_num_angles)
            sd_generation_time = time.time() - start_time

            for sd_img in sd_images:
                # Quality check
                quality = sd_augmentor.assess_quality(sd_img)
                if quality > 0.7:  # Threshold
                    emb = recognizer.extract_embedding(sd_img)
                    if emb:
                        all_embeddings.append(emb)
        except Exception as e:
            logger.error(f"SD augmentation failed: {e}")
            # Continue without SD (graceful degradation)

    # ... rest of enrollment logic ...
```

### 2. Error Handling Strategy

```python
try:
    sd_images = sd_augmentor.generate_angles(image)
except torch.cuda.OutOfMemoryError:
    logger.error("GPU OOM during SD generation - falling back to traditional augmentation")
    # Continue with traditional augmentation only
except Exception as e:
    logger.error(f"SD generation failed: {e}")
    # Continue without SD augmentation
```

### 3. Response Builder

```python
response = {
    "success": True,
    "message": f"Successfully enrolled {name}",
    "person_id": person.id,
    "embeddings_created": len(all_embeddings),
    "augmentation_breakdown": {
        "original": 1,
        "traditional": len(variations) if use_augmentation else 0,
        "stable_diffusion": len(sd_images) if use_sd_augmentation else 0
    }
}

if use_sd_augmentation:
    response["sd_generation_time"] = sd_generation_time
    response["quality_scores"] = {
        "average": np.mean(quality_scores),
        "min": np.min(quality_scores),
        "max": np.max(quality_scores)
    }
```

## 🧪 Testing Requirements

Create `tests/integration/test_sd_enrollment.py`:

```python
def test_enroll_with_sd_augmentation():
    \"\"\"Test enrollment with SD augmentation enabled\"\"\"

def test_enroll_sd_graceful_degradation():
    \"\"\"Test fallback when SD fails\"\"\"

def test_enroll_multiple_with_sd():
    \"\"\"Test multi-image enrollment with SD\"\"\"

def test_sd_quality_filtering():
    \"\"\"Test low-quality SD images are filtered\"\"\"
```

## 📊 Performance Benchmarks

Run and document:
```bash
# Enrollment without SD
time curl -X POST /api/enroll -F "file=@test.jpg" -F "name=Test" -F "cnic=12345"

# Enrollment with SD (5 angles)
time curl -X POST /api/enroll -F "file=@test.jpg" -F "name=Test" -F "cnic=12345" \
  -F "use_sd_augmentation=true" -F "sd_num_angles=5"
```

**Expected Times**:
- Without SD: ~1-2s
- With SD (5 angles): ~10-12s

## ✅ Acceptance Criteria
- [ ] API accepts new SD parameters
- [ ] SD augmentation generates specified number of angles
- [ ] Quality filtering works (rejects poor generations)
- [ ] Graceful degradation on SD failure
- [ ] Response includes SD metrics
- [ ] API documentation updated
- [ ] All tests pass
- [ ] Performance acceptable (<15s total for SD enrollment)

## ⏱️ Estimated Time
**6-8 hours** (1-2 days)

## 🔗 Dependencies
- **Depends on**: Issue #2 (generative_augmentation.py must be implemented)
- **Blocks**: Issue #4 (frontend needs working API)

## 👥 Assignment
**Primary**: @muhammadmahadazher
**Code Review**: @mujeebawan
**Testing**: @AsjalAlvi1

## 📌 Priority
**High** - Required for frontend integration
```

---

## Issue #4: Frontend UI

**Title:**
```
Phase 5.4: Add SD augmentation UI controls to admin panel
```

**Labels:** `phase-5` `feature` `frontend` `ui`

**Assignee:** @AsjalAlvi1

**Milestone:** Phase 5 - AI Augmentation

**Description:**
```markdown
## 🎯 Objective
Update the admin panel enrollment interface to provide user controls for Stable Diffusion augmentation, including progress indicators and preview of generated images.

## 📋 Tasks
- [ ] Add SD augmentation toggle checkbox
- [ ] Add number of angles selector (1-10)
- [ ] Show loading animation during SD generation
- [ ] Display progress messages ("Generating angle 3/5...")
- [ ] Show preview grid of generated images before enrollment
- [ ] Add status indicator showing SD availability
- [ ] Update help text explaining SD augmentation
- [ ] Add toggle to compare "with SD" vs "without SD" results
- [ ] Update success message with augmentation details
- [ ] Write frontend tests

## 🎨 UI Design

### Admin Panel Updates (`app/static/admin.html`)

#### 1. Augmentation Options Section

```html
<div class="augmentation-options" style="margin-top: 20px; padding: 15px; background: #1e293b; border-radius: 8px;">
    <h3 style="color: #f59e0b; margin-bottom: 10px;">📐 Data Augmentation Options</h3>

    <!-- Traditional Augmentation (Existing) -->
    <div class="form-check">
        <input type="checkbox" id="useAugmentation" checked>
        <label for="useAugmentation">
            ✨ Use Traditional Augmentation (brightness, rotation, etc.)
        </label>
    </div>

    <!-- SD Augmentation (NEW) -->
    <div class="form-check" style="margin-top: 10px;">
        <input type="checkbox" id="useSdAugmentation">
        <label for="useSdAugmentation">
            🤖 Use AI Augmentation (Stable Diffusion - Generate Multiple Angles)
            <span class="badge badge-new">NEW</span>
        </label>
        <p class="help-text">
            Generate realistic face angles (frontal, left, right, up, down) using AI.
            Significantly improves recognition accuracy. Takes ~10-15 seconds.
        </p>
    </div>

    <!-- Number of Angles (shown only when SD is enabled) -->
    <div id="sdAngleSelector" style="display: none; margin-top: 10px; margin-left: 30px;">
        <label for="sdNumAngles">Number of angles to generate:</label>
        <select id="sdNumAngles" class="form-input" style="width: 200px;">
            <option value="3">3 angles (Fast)</option>
            <option value="5" selected>5 angles (Recommended)</option>
            <option value="7">7 angles (Thorough)</option>
            <option value="10">10 angles (Maximum)</option>
        </select>
        <p class="help-text">
            More angles = better accuracy but slower enrollment
        </p>
    </div>
</div>
```

#### 2. SD Status Indicator

```html
<div id="sdStatus" class="status-indicator" style="margin-top: 10px;">
    <span class="status-icon">🔄</span>
    <span class="status-text">Checking Stable Diffusion availability...</span>
</div>
```

```javascript
// Check if SD is available
async function checkSdAvailability() {
    try {
        const response = await fetch('/api/sd/status');
        const data = await response.json();

        const statusDiv = document.getElementById('sdStatus');
        if (data.available) {
            statusDiv.innerHTML = `
                <span class="status-icon">✅</span>
                <span class="status-text">Stable Diffusion: Ready (${data.model_name})</span>
            `;
            document.getElementById('useSdAugmentation').disabled = false;
        } else {
            statusDiv.innerHTML = `
                <span class="status-icon">⚠️</span>
                <span class="status-text">Stable Diffusion: Not Available</span>
            `;
            document.getElementById('useSdAugmentation').disabled = true;
        }
    } catch (error) {
        console.error('Failed to check SD status:', error);
    }
}

// Call on page load
checkSdAvailability();
```

#### 3. Loading Animation During Generation

```html
<div id="sdLoadingOverlay" class="modal" style="display: none;">
    <div class="modal-content" style="max-width: 500px; text-align: center;">
        <div class="modal-header">🤖 Generating Face Angles with AI</div>
        <div class="modal-body">
            <div class="spinner-large"></div>
            <p id="sdProgressText" style="margin-top: 20px; font-size: 1.1em; color: #60a5fa;">
                Initializing Stable Diffusion...
            </p>
            <div class="progress-bar" style="margin-top: 20px;">
                <div id="sdProgressBar" style="width: 0%; height: 30px; background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 5px; transition: width 0.3s;"></div>
            </div>
            <p style="margin-top: 10px; color: #94a3b8; font-size: 0.9em;">
                This may take 10-15 seconds. AI is generating realistic face angles...
            </p>
        </div>
    </div>
</div>
```

```javascript
function showSdLoading() {
    document.getElementById('sdLoadingOverlay').style.display = 'flex';

    // Simulate progress updates
    let progress = 0;
    const messages = [
        "Initializing Stable Diffusion...",
        "Generating frontal angle...",
        "Generating left profile...",
        "Generating right profile...",
        "Generating upward tilt...",
        "Generating downward tilt...",
        "Assessing quality...",
        "Finalizing..."
    ];

    const interval = setInterval(() => {
        progress += 12.5;
        document.getElementById('sdProgressBar').style.width = progress + '%';

        const msgIndex = Math.floor(progress / 12.5);
        if (msgIndex < messages.length) {
            document.getElementById('sdProgressText').textContent = messages[msgIndex];
        }

        if (progress >= 100) {
            clearInterval(interval);
        }
    }, 1250);
}

function hideSdLoading() {
    document.getElementById('sdLoadingOverlay').style.display = 'none';
}
```

#### 4. Generated Images Preview

```html
<div id="sdPreviewModal" class="modal" style="display: none;">
    <div class="modal-content" style="max-width: 800px;">
        <div class="modal-header">🎨 AI-Generated Face Angles Preview</div>
        <div class="modal-body">
            <p style="color: #94a3b8; margin-bottom: 15px;">
                Review the generated face angles. These will be added to improve recognition accuracy.
            </p>
            <div id="sdPreviewGrid" class="image-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                <!-- Generated images will be populated here -->
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary" onclick="rejectSdImages()">Reject & Use Original Only</button>
            <button class="btn btn-primary" onclick="acceptSdImages()">✓ Accept & Enroll with These Angles</button>
        </div>
    </div>
</div>
```

#### 5. Updated Form Submission

```javascript
form.addEventListener('submit', async function(e) {
    e.preventDefault();

    const useSdAugmentation = document.getElementById('useSdAugmentation').checked;
    const sdNumAngles = document.getElementById('sdNumAngles').value;

    const formData = new FormData();
    formData.append('name', name);
    formData.append('cnic', cnic);
    formData.append('file', photo);
    formData.append('use_augmentation', useAugmentation);
    formData.append('use_sd_augmentation', useSdAugmentation);
    formData.append('sd_num_angles', sdNumAngles);

    // Show loading if SD is enabled
    if (useSdAugmentation) {
        showSdLoading();
    } else {
        loading.style.display = 'block';
    }

    try {
        const response = await fetch('/api/enroll', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (useSdAugmentation) {
            hideSdLoading();
        }

        if (response.ok && data.success) {
            let message = `✅ Successfully enrolled "${name}"!`;

            if (data.augmentation_breakdown) {
                message += `\n\n📊 Augmentation Details:
• Original: ${data.augmentation_breakdown.original} image
• Traditional: ${data.augmentation_breakdown.traditional} variations
• AI Generated: ${data.augmentation_breakdown.stable_diffusion} angles
• Total Embeddings: ${data.embeddings_created}`;

                if (data.sd_generation_time) {
                    message += `\n• Generation Time: ${data.sd_generation_time.toFixed(1)}s`;
                }
            }

            showSuccess(message);
        } else {
            showError(data.detail || 'Failed to enroll person');
        }
    } catch (error) {
        if (useSdAugmentation) {
            hideSdLoading();
        }
        showError('Network error: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        loading.style.display = 'none';
    }
});
```

## 🎨 Styling Updates

Add CSS for new elements:

```css
.augmentation-options {
    background: #1e293b;
    border-radius: 8px;
    padding: 15px;
    margin-top: 20px;
}

.form-check {
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
}

.form-check input[type="checkbox"] {
    margin-right: 10px;
    margin-top: 3px;
}

.badge-new {
    background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7em;
    margin-left: 5px;
    font-weight: bold;
}

.spinner-large {
    width: 80px;
    height: 80px;
    border: 8px solid #1e293b;
    border-top: 8px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.progress-bar {
    width: 100%;
    background: #1e293b;
    border-radius: 5px;
    overflow: hidden;
}

.status-indicator {
    display: flex;
    align-items: center;
    padding: 10px;
    background: #0f172a;
    border-radius: 6px;
}

.status-icon {
    font-size: 1.2em;
    margin-right: 10px;
}

.status-text {
    color: #94a3b8;
    font-size: 0.9em;
}

.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
}

.image-grid img {
    width: 100%;
    border-radius: 8px;
    border: 2px solid #334155;
}
```

## ✅ Acceptance Criteria
- [ ] SD augmentation checkbox appears and functions
- [ ] Number of angles selector works
- [ ] SD status indicator shows availability
- [ ] Loading animation appears during SD generation
- [ ] Progress messages update during generation
- [ ] Success message shows augmentation breakdown
- [ ] UI is responsive and user-friendly
- [ ] Help text is clear and informative
- [ ] All controls are accessible (keyboard navigation)

## 🧪 Testing
- [ ] Manual testing: Enable/disable SD toggle
- [ ] Test with SD available and unavailable
- [ ] Test different angle counts (3, 5, 7, 10)
- [ ] Test error handling (SD generation fails)
- [ ] Test on different screen sizes (mobile, tablet, desktop)

## ⏱️ Estimated Time
**8-10 hours** (2-3 days)

## 🔗 Dependencies
- **Depends on**: Issue #3 (API must support SD parameters)

## 👥 Assignment
**Primary**: @AsjalAlvi1
**Code Review**: @mujeebawan
**UX Review**: @muhammadmahadazher

## 📌 Priority
**Medium-High** - User-facing feature
```

---

## Issue #5: Optimization

**Title:**
```
Phase 5.5: Optimize SD pipeline for Jetson AGX Orin
```

**Labels:** `phase-5` `performance` `optimization` `gpu`

**Assignee:** @mujeebawan

**Milestone:** Phase 5 - AI Augmentation

**Description:**
```markdown
## 🎯 Objective
Optimize Stable Diffusion inference pipeline to run efficiently on Jetson AGX Orin hardware with limited GPU memory (8GB), targeting <2s per image generation while maintaining quality.

## 📋 Tasks
- [ ] Profile current SD pipeline memory usage and performance
- [ ] Implement FP16 precision throughout pipeline
- [ ] Enable attention slicing to reduce VRAM usage
- [ ] Explore model quantization (INT8 if possible)
- [ ] Implement intelligent caching (embeddings, VAE)
- [ ] Add VRAM monitoring and auto-scaling
- [ ] Implement graceful degradation on OOM errors
- [ ] Benchmark before/after performance
- [ ] Test recognition accuracy with optimized pipeline
- [ ] Document optimization techniques used

## 🔧 Optimization Techniques

### 1. FP16 Precision (Priority: High)

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # Use FP16
    safety_checker=None  # Disable for speed
)
pipe = pipe.to("cuda")
```

**Expected Impact**:
- Memory: ~6GB → ~3GB (50% reduction)
- Speed: Marginal improvement

### 2. Attention Slicing (Priority: High)

```python
# Reduces memory at slight speed cost
pipe.enable_attention_slicing(slice_size="auto")

# Or manual control
pipe.enable_attention_slicing(slice_size=1)  # Most memory efficient
```

**Expected Impact**:
- Memory: ~3GB → ~2GB (33% reduction)
- Speed: +10-20% slower (acceptable trade-off)

### 3. Model Offloading (Priority: Medium)

```python
# Sequential CPU offload (only if needed)
pipe.enable_sequential_cpu_offload()

# Or manual module offloading
pipe.enable_model_cpu_offload()
```

**Expected Impact**:
- Memory: Significant reduction
- Speed: 30-50% slower (last resort)

### 4. VAE Slicing (Priority: Medium)

```python
# Decode large images in slices
pipe.enable_vae_slicing()
```

**Expected Impact**:
- Memory: ~500MB reduction for decoding
- Speed: Minimal impact

### 5. Embedding Caching (Priority: High)

```python
class OptimizedGenerativeAugmentor:
    def __init__(self):
        self.pipe = ...
        self.text_embeddings_cache = {}

    def _get_text_embeddings(self, prompt: str):
        if prompt not in self.text_embeddings_cache:
            self.text_embeddings_cache[prompt] = self.pipe.encode_prompt(prompt)
        return self.text_embeddings_cache[prompt]
```

**Expected Impact**:
- Speed: 15-20% faster for repeated prompts
- Memory: +200MB (acceptable)

### 6. Batch Processing (Priority: Low)

```python
# Generate multiple angles in one batch
images = pipe(
    prompt=[prompt] * num_angles,
    num_inference_steps=20,  # Reduce from 50
    guidance_scale=7.5
).images
```

**Expected Impact**:
- Speed: 30-40% faster overall
- Memory: Higher peak usage (risky on Jetson)

### 7. Reduced Inference Steps (Priority: Medium)

```python
# Reduce from 50 to 20-25 steps
images = pipe(
    prompt=prompt,
    num_inference_steps=20,  # vs default 50
    guidance_scale=7.5
).images
```

**Expected Impact**:
- Speed: 60% faster (50 steps → 20 steps)
- Quality: Slight reduction (acceptable)

### 8. TensorRT Optimization (Priority: Future)

```python
# Compile model with TensorRT (advanced)
from torch2trt import torch2trt

# Convert UNet to TensorRT
pipe.unet = torch2trt(pipe.unet, ...)
```

**Expected Impact**:
- Speed: 2-3x faster
- Complexity: High (Phase 6 task)

## 📊 Performance Benchmarking

Create `scripts/utilities/benchmark_sd.py`:

```python
import time
import torch
from app.core.generative_augmentation import GenerativeAugmentor

def benchmark_sd_performance():
    print("=" * 60)
    print("Stable Diffusion Performance Benchmark")
    print("=" * 60)

    # Load test image
    test_image = cv2.imread("tests/fixtures/test_face.jpg")

    # Initialize augmentor
    augmentor = GenerativeAugmentor()

    # Benchmark different configurations
    configs = [
        {"name": "Baseline (FP32, no optimizations)", "fp16": False, "slice": False},
        {"name": "FP16 only", "fp16": True, "slice": False},
        {"name": "FP16 + Attention Slicing", "fp16": True, "slice": True},
        {"name": "FP16 + All Optimizations", "fp16": True, "slice": True, "cache": True}
    ]

    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 60)

        # Apply configuration
        augmentor.configure(config)

        # Warm-up run
        _ = augmentor.generate_angles(test_image, num_angles=1)

        # Benchmark run
        start_memory = torch.cuda.memory_allocated() / 1024**2
        start_time = time.time()

        images = augmentor.generate_angles(test_image, num_angles=5)

        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2

        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Avg per image: {(end_time - start_time) / 5:.2f}s")
        print(f"Peak GPU memory: {peak_memory:.1f} MB")
        print(f"Generated images: {len(images)}")

        torch.cuda.empty_cache()
```

**Run Benchmark**:
```bash
python3 scripts/utilities/benchmark_sd.py
```

**Expected Results**:

| Configuration | Time (5 images) | Per Image | Peak Memory | Quality |
|---------------|----------------|-----------|-------------|---------|
| Baseline (FP32) | ~30s | ~6s | 8.5GB | 1.0x |
| FP16 only | ~28s | ~5.6s | 4.2GB | 0.99x |
| FP16 + Slicing | ~12s | ~2.4s | 3.1GB | 0.98x |
| **Optimized (Target)** | **~10s** | **<2s** | **<3GB** | **>0.95x** |

## 🧪 Quality Assessment

Test that optimization doesn't degrade recognition:

```python
def test_optimized_quality():
    \"\"\"Test optimized images still recognized by ArcFace\"\"\"

    # Generate with baseline
    baseline_images = baseline_augmentor.generate_angles(input)

    # Generate with optimized
    optimized_images = optimized_augmentor.generate_angles(input)

    # Compare recognition similarity
    for baseline_img, optimized_img in zip(baseline_images, optimized_images):
        baseline_emb = recognizer.extract_embedding(baseline_img)
        optimized_emb = recognizer.extract_embedding(optimized_img)

        similarity = cosine_similarity(baseline_emb, optimized_emb)

        # Should be >0.9 similar
        assert similarity > 0.9, f"Quality degradation: {similarity:.3f}"
```

## 📝 Documentation Updates

Update `docs/development/OPTIMIZATION.md`:

```markdown
## Stable Diffusion Optimization for Jetson

### Hardware Constraints
- GPU Memory: 8GB shared with system
- Available for SD: ~6-7GB typically
- Target usage: <3GB for stable operation

### Applied Optimizations
1. FP16 precision: 50% memory reduction
2. Attention slicing: 33% additional reduction
3. VAE slicing: 500MB saved on decoding
4. Embedding caching: 20% speed improvement
5. Reduced inference steps: 20 steps (60% faster)

### Performance Metrics
- Generation time: 1.8s per image (target: <2s) ✅
- GPU memory: 2.8GB (target: <3GB) ✅
- Quality: 97% of baseline (target: >95%) ✅

### Troubleshooting OOM Errors
If you encounter Out-of-Memory errors:
1. Enable model CPU offloading
2. Reduce batch size to 1
3. Lower image resolution (512 → 384)
4. Clear cache between generations
```

## ✅ Acceptance Criteria
- [ ] Generation time: <2s per image (average)
- [ ] Peak GPU memory: <3GB during generation
- [ ] Recognition similarity: >0.9 vs baseline
- [ ] No OOM errors during normal operation
- [ ] Graceful degradation if OOM occurs
- [ ] Benchmark results documented
- [ ] Optimization techniques documented

## ⏱️ Estimated Time
**10-12 hours** (2-3 days)

## 🔗 Dependencies
- **Depends on**: Issue #2 (needs base implementation to optimize)

## 👥 Assignment
**Primary**: @mujeebawan
**Testing**: @AsjalAlvi1
**Review**: @muhammadmahadazher

## 📌 Priority
**Medium** - Important for user experience but not blocking
```

---

**END OF PHASE 5 ISSUES**

Save this file and use these templates to create issues on GitHub!
