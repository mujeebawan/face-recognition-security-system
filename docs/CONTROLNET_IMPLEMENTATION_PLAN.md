# ControlNet Implementation Plan for Precise Face Pose Control

**Created:** 2025-11-03
**Status:** 📋 Planning Phase
**Target:** Phase 6.0 - ControlNet Integration

---

## 🎯 Objective

Replace current Stable Diffusion img2img approach with **ControlNet + IP-Adapter** to generate **precise, identity-preserving face angles** from a single enrollment photo.

**Goal:** Generate 5 face angles (left, right, up, down, frontal) that:
1. **Preserve exact identity** - Same person, same features
2. **Precise pose control** - Specific yaw/pitch/roll angles
3. **High quality** - Photorealistic, suitable for face recognition training
4. **Consistent results** - Same input = same output

---

## 🔧 Current System (Phase 5.5)

### What We Have
- **Stable Diffusion 1.5** with img2img pipeline
- FP16 precision for Jetson optimization
- Attention slicing for memory efficiency
- ~1.5-3s per 512x512 image generation

### Current Problems
1. **Identity drift** - Generated faces look "similar" but not exactly the same person
2. **Limited pose control** - Prompts like "looking left" don't guarantee specific angles
3. **Inconsistent results** - Same prompt can produce different poses
4. **No geometric accuracy** - Can't specify exact rotation degrees

---

## 🚀 Proposed Solution: ControlNet + IP-Adapter

### Architecture
```
Input Image (frontal face)
     ↓
[Extract Face ID Embedding] ← IP-Adapter-FaceID
     ↓
[Generate Depth/Normal Map] ← ControlNet Depth
     ↓
[Rotate Depth Map] ← Manual transformation for target angle
     ↓
[Generate Image with SD 1.5] ← Guided by rotated depth + face ID
     ↓
Output: Same person, different angle
```

---

## 📦 Required Components

### 1. ControlNet Models
**Primary Choice: ControlNet Depth**
```
Model: lllyasviel/control_v11f1p_sd15_depth
Size: ~1.5GB
Input: Depth map (grayscale image)
Purpose: Preserves 3D structure while allowing pose changes
```

**Alternative: ControlNet Normal**
```
Model: lllyasviel/control_v11p_sd15_normalbae
Size: ~1.5GB
Input: Surface normal map
Purpose: Better for subtle angle changes
```

### 2. IP-Adapter for Identity Preservation
```
Model: h94/IP-Adapter-FaceID
Size: ~600MB
Input: Face embedding (512-D or face image)
Purpose: Strong identity preservation during generation
```

### 3. Depth Estimation Model (for preprocessing)
```
Model: Intel/dpt-hybrid-midas (MiDaS)
Size: ~400MB
Purpose: Extract depth map from input face image
```

---

## 🔄 Implementation Workflow

### Step 1: Extract Face ID Embedding
```python
from insightface.app import FaceAnalysis

# Extract face embedding for identity
face_analyzer = FaceAnalysis()
face_analyzer.prepare(ctx_id=0)  # GPU

faces = face_analyzer.get(input_image)
face_embedding = faces[0].embedding  # 512-D vector
```

### Step 2: Generate Depth Map
```python
from transformers import DPTImageProcessor, DPTForDepthEstimation

processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

inputs = processor(images=input_image, return_tensors="pt")
depth_map = model(**inputs).predicted_depth
```

### Step 3: Transform Depth Map for Target Angle
```python
import cv2
import numpy as np

def rotate_depth_map(depth_map, yaw, pitch, roll):
    """
    Rotate depth map to simulate head rotation.

    Args:
        yaw: Left-right rotation (-90 to +90 degrees)
        pitch: Up-down rotation (-45 to +45 degrees)
        roll: Tilt rotation (-30 to +30 degrees)
    """
    # Apply 3D rotation matrix
    # Project back to 2D depth map
    # ... (detailed implementation in code)
    return rotated_depth_map

# Generate target angles
angles = {
    'left': {'yaw': -30, 'pitch': 0, 'roll': 0},
    'right': {'yaw': 30, 'pitch': 0, 'roll': 0},
    'up': {'yaw': 0, 'pitch': -20, 'roll': 0},
    'down': {'yaw': 0, 'pitch': 20, 'roll': 0},
}
```

### Step 4: Generate with ControlNet + IP-Adapter
```python
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler
)
from ip_adapter import IPAdapterFaceID

# Load models
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Load IP-Adapter for face identity
ip_adapter = IPAdapterFaceID(pipe, "h94/IP-Adapter-FaceID")

# Generate
for angle_name, angle_params in angles.items():
    # Rotate depth map
    rotated_depth = rotate_depth_map(
        depth_map,
        **angle_params
    )

    # Generate with identity + pose control
    image = pipe(
        prompt="high quality face portrait, professional photo",
        negative_prompt="blurry, distorted, different person",
        image=rotated_depth,  # ControlNet input
        ip_adapter_image=input_image,  # Identity reference
        ip_adapter_scale=0.8,  # Strong identity preservation
        controlnet_conditioning_scale=0.9,  # Strong pose control
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
```

---

## 📊 Expected Performance (Jetson AGX Orin)

### Inference Time
- Depth extraction: ~200-300ms
- Depth transformation: ~50ms
- ControlNet + SD generation: ~3-5s per image
- **Total per variation: ~3.5-5.5s**
- **5 variations: ~20-30 seconds total**

### Memory Usage
- ControlNet model: ~2GB
- IP-Adapter: ~800MB
- SD 1.5: ~4GB
- Depth model: ~600MB
- **Peak VRAM: ~10-12GB** (within Jetson limits)

### Quality Expectations
- Identity preservation: 95%+ (IP-Adapter)
- Pose accuracy: ±5 degrees (ControlNet depth)
- Photorealism: High (SD 1.5)
- Consistency: Deterministic with fixed seed

---

## 🛠️ Implementation Steps

### Phase 1: Setup & Testing (Days 1-2)
1. Install dependencies
   ```bash
   pip3 install insightface ip-adapter-faceid diffusers transformers
   ```

2. Download models (cache to ~/.cache/huggingface)
   - ControlNet Depth (~1.5GB)
   - IP-Adapter-FaceID (~600MB)
   - MiDaS depth (~400MB)

3. Create test script
   - Load models
   - Test depth extraction
   - Test ControlNet generation
   - Verify GPU usage

### Phase 2: Depth Transformation (Days 3-4)
1. Implement `rotate_depth_map()` function
   - 3D rotation matrices
   - Perspective projection
   - Handle edge cases

2. Test with predefined angles
   - Yaw: -30, -15, 0, +15, +30
   - Pitch: -20, -10, 0, +10, +20

3. Validate output quality

### Phase 3: Integration (Days 5-6)
1. Update `generative_augmentation.py`
   - Replace img2img with ControlNet pipeline
   - Add IP-Adapter integration
   - Implement depth rotation

2. Add configuration options
   - Enable/disable ControlNet vs img2img
   - Adjustable angle values
   - Quality vs speed trade-offs

3. Memory optimization
   - Model offloading
   - VRAM management
   - Batch processing

### Phase 4: Testing & Tuning (Days 7-8)
1. Test with real enrollment photos
2. Measure face recognition accuracy
3. Compare with img2img results
4. Tune hyperparameters:
   - IP-Adapter scale
   - ControlNet conditioning scale
   - Guidance scale
   - Inference steps

5. Performance profiling
   - Generation speed
   - Memory usage
   - GPU utilization

### Phase 5: Production Integration (Days 9-10)
1. Update enrollment API
2. Add UI controls for ControlNet
3. Documentation updates
4. Create rollback plan (keep img2img as fallback)

---

## 🎛️ Configuration Options

### Angle Presets
```python
ANGLE_PRESETS = {
    'subtle': {  # Minimal rotation for slight variations
        'left': {'yaw': -15, 'pitch': 0},
        'right': {'yaw': 15, 'pitch': 0},
        'up': {'yaw': 0, 'pitch': -10},
        'down': {'yaw': 0, 'pitch': 10},
    },
    'moderate': {  # Balanced (default)
        'left': {'yaw': -30, 'pitch': 0},
        'right': {'yaw': 30, 'pitch': 0},
        'up': {'yaw': 0, 'pitch': -20},
        'down': {'yaw': 0, 'pitch': 20},
    },
    'extreme': {  # Maximum rotation for edge cases
        'left': {'yaw': -45, 'pitch': 0},
        'right': {'yaw': 45, 'pitch': 0},
        'up': {'yaw': 0, 'pitch': -30},
        'down': {'yaw': 0, 'pitch': 30},
    }
}
```

### Quality/Speed Modes
```python
GENERATION_MODES = {
    'fast': {  # Quick generation, lower quality
        'num_inference_steps': 20,
        'controlnet_scale': 0.8,
        'ip_adapter_scale': 0.7
    },
    'balanced': {  # Recommended
        'num_inference_steps': 30,
        'controlnet_scale': 0.9,
        'ip_adapter_scale': 0.8
    },
    'quality': {  # Best quality, slower
        'num_inference_steps': 50,
        'controlnet_scale': 1.0,
        'ip_adapter_scale': 0.9
    }
}
```

---

## 🚨 Potential Challenges & Solutions

### Challenge 1: GPU Memory (10-12GB needed)
**Solutions:**
- Model offloading to CPU when not in use
- Sequential loading (load depth → extract → unload → load SD)
- Reduce SD to 1.4 (smaller than 1.5)
- Use attention slicing (already implemented)

### Challenge 2: Slow Generation (~5s per image)
**Solutions:**
- Accept slower speed for better quality
- Option to skip ControlNet for fast enrollment
- Batch processing (precompute depth maps)
- Consider LCM-LoRA for 4x speedup (experimental)

### Challenge 3: Depth Map Quality on Profile Views
**Solutions:**
- Use ControlNet Normal instead of Depth
- Combine multiple ControlNets (depth + normal)
- Fine-tune depth transformation algorithm
- Add face landmark guidance

### Challenge 4: Identity Drift Despite IP-Adapter
**Solutions:**
- Increase IP-Adapter scale (0.8 → 0.9)
- Add face embedding loss during generation
- Use InstantID instead of IP-Adapter (newer)
- Post-process: verify identity with ArcFace before saving

---

## ✅ Success Criteria

1. **Identity Match:** ArcFace similarity > 0.7 between original and generated
2. **Pose Accuracy:** Generated angles within ±10° of target
3. **Recognition Performance:** Generated images improve multi-angle recognition accuracy by >10%
4. **Speed:** Total generation time < 30s for 5 variations
5. **Reliability:** Success rate > 95% (valid face detected in output)

---

## 📝 Code Structure

### New Files
```
app/core/controlnet_augmentation.py  - ControlNet implementation
app/core/depth_utils.py              - Depth extraction & transformation
app/utils/face_rotation.py           - 3D rotation utilities
```

### Modified Files
```
app/core/generative_augmentation.py  - Add ControlNet option
app/api/routes/recognition.py        - Update enrollment to use ControlNet
app/static/admin.html                - Add ControlNet UI controls
app/config.py                        - Add ControlNet settings
```

---

## 🔄 Rollback Plan

If ControlNet doesn't work as expected:

1. Keep img2img as fallback mode (default=img2img)
2. Add `use_controlnet=False` parameter to enrollment
3. Document issues encountered
4. Consider 3D face reconstruction as alternative

---

## 📚 References

- ControlNet Paper: https://arxiv.org/abs/2302.05543
- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter
- InstantID: https://github.com/InstantID/InstantID
- MiDaS Depth: https://github.com/isl-org/MiDaS
- Diffusers ControlNet: https://huggingface.co/docs/diffusers/using-diffusers/controlnet

---

## 🎯 Next Actions

1. ✅ Documentation complete
2. ⏭️ Install dependencies (insightface, ip-adapter, etc.)
3. ⏭️ Download ControlNet and IP-Adapter models
4. ⏭️ Implement depth extraction pipeline
5. ⏭️ Create depth rotation algorithm
6. ⏭️ Integrate with existing enrollment flow
7. ⏭️ Test and validate results

---

**Estimated Timeline:** 10 days
**Difficulty:** High
**Impact:** Very High (significant improvement in face recognition accuracy)
**Priority:** High
