# Training Workflow Guide — Step-by-Step

**Complete guide for training all 4 synthesis models from scratch**

---

## 📋 Pre-Training Checklist

- [ ] Python 3.9+ installed
- [ ] PyTorch installed (GPU recommended)
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r synthesis-module/requirements.txt`
- [ ] BraTS2021 dataset available
- [ ] Adequate disk space (~50GB for dataset + models + logs)
- [ ] GPU with 12GB+ VRAM (or reduce batchsize for smaller GPUs)

---

## 🔧 Phase 0: Environment Setup

### 1. Create Virtual Environment
```bash
cd c:\Source\Thesis\thesis-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Core Dependencies
```bash
# PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Medical imaging
pip install nibabel torchio tensorboard

# Training utilities
pip install tqdm numpy scipy scikit-image
```

### 3. Install Synthesis Module Requirements
```bash
pip install -r models/diffusion-for-mri-tumor-brain-creation/requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
python -c "import nibabel; print('OK')"
python -c "import torchio; print('OK')"
```

---

## 📊 Phase 1: Data Preparation

### Option A: Using Existing BraTS Dataset

```bash
cd models/diffusion-for-mri-tumor-brain-creation

# If data is already in 'dataset/' folder, skip to Phase 2

# Otherwise, prepare dataset
python prepare_brats_dataset.py \
  --source /path/to/BRATS2021/source \
  --output dataset \
  --modalities t1,t1ce,t2,flair
```

### Option B: Create Split

```bash
# Create train/val/test split
python -c "
from dataset_brats import make_split_case_lists
make_split_case_lists(
    data_root='dataset',
    output_dir='splits',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)
"
```

**Expected Output**:
```
dataset/
  ├── BraTS2021_00000/
  │   ├── BraTS2021_00000_flair.nii.gz
  │   ├── BraTS2021_00000_t1.nii.gz
  │   ├── BraTS2021_00000_t1ce.nii.gz
  │   └── BraTS2021_00000_t2.nii.gz
  ├── BraTS2021_00001/
  └── ...

splits/
  ├── brats_split_8_1_1.json    ← Training/val/test split
  └── case_id.txt               ← All case IDs
```

---

## 🚂 Phase 2: Configure Training

### Edit Configurations (Optional)

If you want custom hyperparameters, edit the config files:

```bash
# Custom T1 model training
cat > configs/synthesis-models/t1_synthesis_config.yaml << 'EOF'
synthesis:
  model:
    name: "t1_synthesis_diffusion"
    latent_dim: 128
    num_channels: 64
    
  training:
    batchsize: 2
    num_epochs: 5000
    learning_rate: 1.0e-4
    
  data:
    target_modality: "t1"
    condition_modalities: ["t1ce", "t2", "flair"]
    input_size: 64
    depth_size: 144
EOF
```

### Check Config File
```bash
# View current settings
cat configs/synthesis-models/synthesis_base_config.yaml | grep -A 10 "training:"
```

---

## 🚀 Phase 3: Train All 4 Models

### Option A: Automatic Training (Recommended)

```bash
cd models/diffusion-for-mri-tumor-brain-creation

# Uses default settings (5000 epochs, batch=2)
bash scripts/train_all_modalities.sh
```

**Timeline**: ~6-8 hours on RTX 3090, ~12-16 hours on RTX 2080Ti

### Option B: Custom Parameters

```bash
bash scripts/train_all_modalities.sh \
  5000 \                    # EPOCHS (default: 5000)
  2 \                       # BATCH_SIZE (default: 2)
  64 \                      # INPUT_SIZE (default: 64)
  144 \                     # DEPTH_SIZE (default: 144)
  250 \                     # TIMESTEPS (default: 250)
  200 \                     # SAVE_EVERY (default: 200)
  0 \                       # GPU_ID (default: 0)
  dataset \                 # DATA_ROOT (default: dataset)
  splits/brats_split_8_1_1.json  # SPLIT_JSON
  1e-4                      # TRAIN_LR (default: 1e-4)
```

### Option C: Train Models Sequentially with Monitoring

```bash
# T1 training
python train_brats.py \
  --task 3to1 \
  --data_root dataset \
  --cond_modalities "t1ce,t2,flair" \
  --target_modality "t1" \
  --epochs 5000 \
  --batchsize 2 \
  --save_and_sample_every 200 \
  --train_lr 1e-4 \
  --with_condition

# T1ce training
python train_brats.py \
  --task 3to1 \
  --data_root dataset \
  --cond_modalities "t1,t2,flair" \
  --target_modality "t1ce" \
  --epochs 5000 \
  --batchsize 2 \
  --save_and_sample_every 200 \
  --train_lr 1e-4 \
  --with_condition

# ... (repeat for T2 and FLAIR)
```

### Option D: Resume Interrupted Training

```bash
# If training was interrupted, resume from latest checkpoint
python train_brats.py \
  --task 3to1 \
  --cond_modalities "t1ce,t2,flair" \
  --target_modality "t1" \
  --epochs 5000 \
  --resume_weight "results_brats/model/model_brats.pt"
```

---

## 📈 Phase 4: Monitor Training

### Option A: TensorBoard (Real-time Monitoring)

```bash
# In new terminal
tensorboard --logdir=results_brats/logs --port=6006
```

Navigate to `http://localhost:6006` in browser.

**What to monitor**:
- Training loss (should decrease)
- Validation loss (should stabilize)
- Learning rate schedule
- Sample visualizations

### Option B: Console Output

Training script prints:
```
[400/5000] loss: 0.0234 | val_loss: 0.0245 | lr: 1.00e-04
[500/5000] loss: 0.0201 | val_loss: 0.0219 | lr: 1.00e-04
...
```

### Option C: Check Results Directory

```bash
# After each save_and_sample_every (default 200 steps)
ls -lh results_brats/model/
ls -lh results_brats/sample/

# View generated samples
# results_brats/sample/epoch_XXX/ contains synthesized modalities
```

---

## 💾 Phase 5: Save Models

After training completes, models are saved to `models/`:

```bash
# List trained models
ls -lh models/model_*.pt

# Output should show:
# -rw-r--r--  model_t1_from_t1ce_t2_flair.pt       (~500MB)
# -rw-r--r--  model_t1ce_from_t1_t2_flair.pt       (~500MB)
# -rw-r--r--  model_t2_from_t1_t1ce_flair.pt       (~500MB)
# -rw-r--r--  model_flair_from_t1_t1ce_t2.pt       (~500MB)
```

### Verify Model Integrity

```python
import torch

# Check T1 model
ckpt = torch.load('models/model_t1_from_t1ce_t2_flair.pt', map_location='cpu')
print(f"Keys: {ckpt.keys()}")
print(f"Size: {sum(p.numel() for p in ckpt.values()) / 1e6:.1f}M parameters")
```

---

## 🧪 Phase 6: Test Inference

Before full pipeline, test a single model:

```python
import torch
import nibabel as nib
import numpy as np
from diffusion_model.unet_brats import create_model
from fast_sampling.inference_ddpm import DDPM_Sampler

# Load model
model = create_model()
model.load_state_dict(torch.load('models/model_t1_from_t1ce_t2_flair.pt'))
model.to('cuda')
model.eval()

# Load test case (T1ce, T2, FLAIR)
t1ce = torch.from_numpy(
    nib.load('dataset/BraTS2021_00100/BraTS2021_00100_t1ce.nii.gz').get_fdata()
).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

t2 = torch.from_numpy(...).unsqueeze(0).unsqueeze(0)
flair = torch.from_numpy(...).unsqueeze(0).unsqueeze(0)

# Test inference
conditioning = torch.cat([t1ce, t2, flair], dim=1).to('cuda')  # (1, 3, D, H, W)

sampler = DDPM_Sampler(model=model, device='cuda', num_steps=50)
with torch.no_grad():
    t1_synthesized = sampler.sample(conditioning=conditioning)

print(f"Input shape: {conditioning.shape}")
print(f"Output shape: {t1_synthesized.shape}")
print(f"Value range: [{t1_synthesized.min():.3f}, {t1_synthesized.max():.3f}]")

# Sanity check: output should have similar range to inputs
print(f"Conditioning range: [{conditioning.min():.3f}, {conditioning.max():.3f}]")
```

---

## 🔄 Phase 7: Integrate with Pipeline

Update main pipeline config to use trained models:

```bash
# Edit pipeline_config.yaml
# Or place in configs/synthesis-models/ and reference

cat >> configs/pipeline_config.yaml << 'EOF'

synthesis:
  weights:
    t1: "models/diffusion-for-mri-tumor-brain-creation/models/model_t1_from_t1ce_t2_flair.pt"
    t1ce: "models/diffusion-for-mri-tumor-brain-creation/models/model_t1ce_from_t1_t2_flair.pt"
    t2: "models/diffusion-for-mri-tumor-brain-creation/models/model_t2_from_t1_t1ce_flair.pt"
    flair: "models/diffusion-for-mri-tumor-brain-creation/models/model_flair_from_t1_t1ce_t2.pt"
EOF
```

Or let the pipeline auto-detect:

```python
# run_pipeline.py will automatically look for models in:
# models/diffusion-for-mri-tumor-brain-creation/models/
```

---

## 🧹 Phase 8: Cleanup & Optimization

### Optional: Compress Models

```bash
# Models are already compressed, but you can:
# Store old checkpoints separately
mkdir -p models/old_checkpoints
mv results_brats/checkpoints/* models/old_checkpoints/ 2>/dev/null

# Remove intermediate results
rm -rf results_brats/

# Save only final models
du -sh models/model_*.pt  # Should be ~500MB × 4 = ~2GB
```

### Optional: Convert to Half Precision (for inference efficiency)

```python
import torch

for modality in ['t1', 't1ce', 't2', 'flair']:
    # Load full precision
    ckpt = torch.load(f'models/model_{modality}*.pt')
    
    # Convert to FP16
    for key in ckpt:
        if isinstance(ckpt[key], torch.Tensor):
            ckpt[key] = ckpt[key].half()
    
    # Save
    torch.save(ckpt, f'models/model_{modality}*_fp16.pt')
```

---

## 📊 Training Statistics

### Expected Results (5000 epochs, batch=2, RTX 3090)

| Metric | Value |
|--------|-------|
| Training time per modality | 1.5-2 hours |
| Total for 4 modalities | 6-8 hours |
| Final model size | ~500 MB each |
| Final loss | ~0.01-0.03 |
| SSIM validation | ~0.85-0.90 |

### Quality Indicators

✅ Good training if:
- Loss smoothly decreases
- No spikes or divergence
- Validation loss follows training loss
- Synthesized samples look realistic

⚠️ Warning signs:
- Training loss plateaus early
- Validation loss increases (overfitting)
- NaN values appear
- Synthesized samples are blurry/noisy

---

## 🆘 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Method 1: Reduce batch size
bash scripts/train_all_modalities.sh 5000 1  # batchsize=1

# Method 2: Reduce input size
python train_brats.py \
  --input_size 48 \
  --depth_size 96 \
  --batchsize 2

# Method 3: Use gradient accumulation (in config)
```

### Issue: Training Very Slow

**Solution**:
```bash
# Check GPU usage
nvidia-smi

# Enable cuDNN auto-tuner
export CUDA_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Use PyTorch lightning or DDP for multi-GPU
# Or reduce data loading workers
python train_brats.py --num_workers 0
```

### Issue: Loss Not Decreasing

**Solution**:
```bash
# Check learning rate
# Try warmer LR schedule
python train_brats.py \
  --train_lr 5e-4 \  # Start higher
  --epochs 10000     # Train longer

# Or reduce LR
python train_brats.py --train_lr 5e-5
```

### Issue: Models Not Saving

**Solution**:
```bash
# Check disk space
df -h

# Verify write permissions
ls -ld models/

# Check checkpoint directory
mkdir -p results_brats/model
```

---

## ✅ Success Criteria

Training is successful when:
- [ ] All 4 models trained without errors
- [ ] Models save to `models/model_*.pt`
- [ ] Each model is ~500MB
- [ ] Loss curves show convergence
- [ ] Validation metrics improve
- [ ] Inference test passes (Phase 6)

---

## 📝 Next Steps After Training

1. **Test in pipeline**: Run `python src/run_pipeline.py` with missing modality
2. **Evaluate quality**: Compare synthesis to ground truth
3. **Fine-tune**: Adjust hyperparameters if needed
4. **Deploy**: Use in production pipeline
5. **Archive**: Save training logs and checkpoints

---

**Estimated Total Time**: 8-12 hours on GPU (depending on hardware & parameters)
