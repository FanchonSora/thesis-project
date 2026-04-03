# Brain Tumor Analysis Pipeline
**End-to-end MRI analysis framework**: **preprocess → synthesise missing modalities → 3-D segment → visualise**

## Quick Start

```bash
# 1. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nibabel scikit-image scipy matplotlib pyyaml tqdm
pip install -r synthesis-module/requirements.txt

# 3. Run inference pipeline (with all 4 modalities)
python src/run_pipeline.py \
  -i results/input/BraTS-GLI-00000-000 \
  -o results/output/BraTS2021_00000 \
  -c configs/pipeline_config.yaml

# 4. Visualize results
python src/visualize/visualize_results.py \
  -p results/output/BraTS2021_00000
```

---

## Data Flow

```
Input MRI (any 3/4 modalities)
  ↓
[PREPROCESSING]
  • Intensity clip (0.5 / 99.5 percentile)
  • Z-score normalization (brain region)
  • Brain mask extraction
  ↓
[SYNTHESIS] - Fill missing modalities
  • Conditional diffusion model (50 DDIM steps)
  • Modality: T1, T1ce, T2, FLAIR
  ↓
[SEGMENTATION] - UNet3D (4 classes)
  • 0: Background
  • 1: NCR (Necrotic Core)
  • 2: ED (Edema)
  • 3: ET (Enhancing Tumor)
  ↓
[VISUALIZATION & REPORT]
  • Segmentation masks + slices
  • Region volumes & metrics
  • 3-D mesh (Pred vs GT)
  ↓
Output: {segmentation, synthesis, report, 3d_mesh}
```

---

## Project Structure

```
thesis-project/
├── configs/
│   ├── pipeline_config.yaml           ← Main pipeline config
│   └── synthesis-models/
│       ├── t1_synthesis_config.yaml
│       ├── t1ce_synthesis_config.yaml
│       ├── t2_synthesis_config.yaml
│       └── flair_synthesis_config.yaml
├── models/
│   ├── diffusion-for-mri-tumor-brain-creation/
│   │   ├── train_brats.py             ← Training entry point
│   │   ├── inference_all_modalities.py
│   │   ├── dataset_brats.py
│   │   ├── scripts/
│   │   │   └── train_all_modalities.sh ← Train 4 models
│   │   ├── diffusion_model/
│   │   │   ├── unet_brats.py
│   │   │   └── trainer_brats.py
│   │   ├── fast_sampling/
│   │   │   ├── inference_ddpm.py
│   │   │   └── inference_deis.py
│   │   └── utils/
│   ├── segmentation-module/
│   │   └── model-weight/
│   │       └── final_model_unet.pth
│   └── synthesis-module/
│       ├── main.py
│       ├── train.py
│       ├── model/
│       │   └── architecture.py
│       ├── model-weight/
│       │   └── epoch_118.pth
│       └── requirements.txt
├── src/
│   ├── run_pipeline.py                ← Main inference CLI
│   ├── preprocessing.py               ← Preprocessing utilities
│   ├── web_api.py
│   ├── models/
│   │   └── unet3d.py                  ← UNet3D + curriculum learning
│   └── visualize/
│       └── visualize_results.py
├── notebooks/
│   └── integrated_pipeline.ipynb
├── results/
│   ├── input/                         ← Input NIFTI files
│   └── output/                        ← Predictions & outputs
└── README.md
```

---

## Installation

**Requirements**: Python 3.9–3.11, GPU (CUDA 11.8+) recommended

### Setup Steps

#### 1. Create Virtual Environment
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

#### 2. Install PyTorch (choose one)
```bash
# GPU (CUDA 11.8) — recommended
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

#### 3. Install Core Dependencies
```bash
pip install -q \
  nibabel scikit-image scipy matplotlib pyyaml \
  tqdm numpy pandas einops tensorboard
```

#### 4. Install Module Dependencies
```bash
pip install -r synthesis-module/requirements.txt
```

#### 5. Optional Packages (Recommended)
| Package | Purpose | Install |
|---------|---------|---------|
| MONAI | Sliding-window inference (handles large volumes) | `pip install monai` |
| Plotly | Interactive 3-D mesh visualization | `pip install plotly` |
| Trimesh | Mesh smoothing & processing | `pip install trimesh` |
| TorchIO | Medical image augmentation | `pip install torchio` |

> ⚠️ **Note**: Without MONAI, inference runs single forward pass. For volumes >128³, install MONAI to avoid OOM.

---

## Training Diffusion Synthesis Models

The `diffusion-for-mri-tumor-brain-creation` module contains training scripts for generating missing MRI modalities.

### Training Setup

```bash
cd models/diffusion-for-mri-tumor-brain-creation

# Install training dependencies
pip install -r requirements.txt

# (Optional) Prepare BraTS dataset
python prepare_brats_dataset.py --source /path/to/BRATS2021
python preprocess_brats_data.py --data_root dataset
```

### Train All 4 Modality Synthesis Models

The script trains 4 independent models, each learning to generate one missing modality from 3 available:

```bash
bash scripts/train_all_modalities.sh \
  [EPOCHS=5000] \
  [BATCH_SIZE=2] \
  [INPUT_SIZE=64] \
  [DEPTH_SIZE=144] \
  [TIMESTEPS=250] \
  [SAVE_EVERY=200] \
  [GPU_ID=0] \
  [DATA_ROOT=dataset] \
  [SPLIT_JSON=splits/brats_split_8_1_1.json] \
  [TRAIN_LR=1e-4]
```

**What it trains**:
1. `T1ce, T2, FLAIR → T1`
2. `T1, T2, FLAIR → T1ce`
3. `T1, T1ce, FLAIR → T2`
4. `T1, T1ce, T2 → FLAIR`

**Output**: 4 model files saved in `models/`

### Train Single Modality Model

```bash
python train_brats.py \
  --task 3to1 \
  --data_root dataset \
  --cond_modalities "t1ce,t2,flair" \
  --target_modality "t1" \
  --input_size 64 \
  --depth_size 144 \
  --batchsize 2 \
  --epochs 5000 \
  --timesteps 250 \
  --save_and_sample_every 200 \
  --train_lr 1e-4 \
  --split_json splits/brats_split_8_1_1.json \
  --with_condition
```

### Training Configuration

All model hyperparameters are defined in YAML config files under `configs/synthesis-models/`:

```yaml
model:
  name: "diffusion_synthesis"
  latent_dim: 128
  num_timesteps: 1000
  num_modalities: 4
  num_domains: 3
  num_channels: 64
  num_res_blocks: 2

training:
  optimizer: "adamw"
  learning_rate: 1e-4
  batchsize: 2
  epochs: 5000
  timesteps: 250
  save_every: 200
  
data:
  input_size: 64
  depth_size: 144
  modalities: ["t1", "t1ce", "t2", "flair"]
  split_seed: 42
```

### Resuming Training

```bash
python train_brats.py \
  --task 3to1 \
  --data_root dataset \
  --cond_modalities "t1ce,t2,flair" \
  --target_modality "t1" \
  --epochs 5000 \
  --resume_weight "models/model_t1_from_t1ce_t2_flair.pt"
```

### Inference with Trained Models

After training, use the generated models in the main pipeline:

```bash
python src/run_pipeline.py \
  -i results/input/BraTS-GLI-00000-000 \
  -o results/output/BraTS2021_00000 \
  --syn-w models/diffusion-for-mri-tumor-brain-creation/models/model_t1_from_t1ce_t2_flair.pt
```

Or update the config:

```yaml
synthesis:
  weights:
    t1: "models/diffusion-for-mri-tumor-brain-creation/models/model_t1_from_t1ce_t2_flair.pt"
    t1ce: "models/diffusion-for-mri-tumor-brain-creation/models/model_t1ce_from_t1_t2_flair.pt"
    t2: "models/diffusion-for-mri-tumor-brain-creation/models/model_t2_from_t1_t1ce_flair.pt"
    flair: "models/diffusion-for-mri-tumor-brain-creation/models/model_flair_from_t1_t1ce_t2.pt"
```

---

## Synthesis Pipeline Overview

The synthesis module generates missing MRI modalities using **conditional diffusion models**:

```
Missing Modality + 3 Available Modalities
  ↓
Diffusion Model (Reverse Diffusion Process)
  ↓
Synthesized 4-Modality Volume
```

### Key Components

- **Forward Diffusion**: Gradually add noise to target modality (training)
- **Reverse Diffusion**: Denoise with conditioning on available modalities (inference)
- **Conditioning**: Concatenate 3 available modalities as network input
- **Sampling**: DDIM (50 steps) or DDPM (250 steps) at inference

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `sampling_steps` | 50 | DDIM steps (lower = faster, less quality) |
| `eta` | 0.0 | Determinism: 0=deterministic, 1=stochastic |
| `temperature` | 1.0 | Sampling temperature |
| `unconditional_prob` | 0.0 | Dropout probability for conditioning |

Adjust in `configs/pipeline_config.yaml`:

```yaml
synthesis:
  inference:
    sampling_steps: 50      # 20-250, trade speed vs quality
    eta: 0.0                # 0, 0.5, 1.0
    temperature: 1.0
    unconditional_prob: 0.1 # Enable dropout for robustness
```

---

## Input format

Place your NIfTI files inside one folder per case. The pipeline automatically
detects filenames using both separators (`_` / `-`) and both naming conventions
(BraTS 2021 and BraTS 2023):

```
<input_dir>/
  <case_id>_flair.nii.gz   or   <case_id>-t2f.nii.gz
  <case_id>_t1.nii.gz      or   <case_id>-t1n.nii.gz
  <case_id>_t1ce.nii.gz    or   <case_id>-t1c.nii.gz
  <case_id>_t2.nii.gz      or   <case_id>-t2w.nii.gz
```

If `<case_id>` does not match any filename prefix, the folder is scanned
automatically and the most common prefix is used instead.

Missing modalities are handled transparently:
- Synthesis model loaded → diffusion model generates the missing volume.
- Otherwise → mean of available modalities (zero if all missing).

---

## Step 1 — Run inference

### Minimal (CPU, all modalities present)

```bash
python run_pipeline.py \
  --case-id   BraTS2021_00000 \
  --input-dir ./data/BraTS2021_00000
```

### GPU

```bash
python run_pipeline.py \
  --case-id   BraTS2021_00000 \
  --input-dir ./data/BraTS2021_00000 \
  --device    cuda
```

### Custom weights

```bash
python run_pipeline.py \
  --case-id   BraTS2021_00000 \
  --input-dir ./data/BraTS2021_00000 \
  --seg-w     segmentation-module/model-weight/final_model_unet.pth \
  --syn-w     synthesis-module/model-weight/epoch_118.pth \
  --device    cuda
```

### Large volumes — sliding-window inference (requires MONAI)

```bash
python run_pipeline.py \
  --case-id   BraTS2021_00000 \
  --input-dir ./data/BraTS2021_00000 \
  --device    cuda \
  --roi       128 128 64
```

> Rule of thumb for `--roi`: start with `128 128 64`. Reduce to `96 96 48` or `64 64 32` if you still get OOM.

### All `run_pipeline.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--case-id` | *(required)* | Case / patient identifier |
| `--input-dir` | *(required)* | Folder containing the NIfTI files |
| `--out-dir` | `./output` | Root directory for all outputs |
| `--seg-w` | `segmentation-module/model-weight/final_model_unet.pth` | Segmentation model weights |
| `--syn-w` | `synthesis-module/model-weight/epoch_118.pth` | Synthesis model weights (optional) |
| `--device` | auto (`cuda` if available, else `cpu`) | `cuda` / `cpu` / `mps` |
| `--roi X Y Z` | *(none — auto)* | Sliding-window patch size (MONAI required) |
| `--syn-steps` | `50` | DDIM sampling steps for synthesis |
| `--max-size` | `240` | Downsample when any spatial dim exceeds this value |

---

## Step 2 — Visualise results

```bash
python visualize.py \
  --result-dir output/BraTS2021_00000 \
  --input-dir  ./data/BraTS2021_00000
```

`--input-dir` is used to locate the ground-truth segmentation (`*_seg.nii.gz` or `*-seg.nii.gz`).
If no GT is found, metrics and the GT mesh are skipped gracefully.

### All `visualize.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--result-dir` | *(required)* | Case output folder from Step 1 |
| `--input-dir` | *(required)* | Original NIfTI input folder (for GT segmentation) |

---

## Outputs

### Inference — `output/<case_id>/`

```
BraTS2021_00000_pred.nii.gz    ← segmentation labels (uint8)
BraTS2021_00000_wt.obj         ← whole-tumour surface mesh (Wavefront OBJ)
BraTS2021_00000_report.json    ← region volumes + pipeline metadata
```

### Visualisation — appended to `output/<case_id>/`

```
BraTS2021_00000_slices.png     ← axial / coronal / sagittal: GT vs Prediction
BraTS2021_00000_metrics.png    ← Dice / Precision / Sensitivity / Specificity per region
BraTS2021_00000_volumes.png    ← region volume bar chart + modality availability
BraTS2021_00000_mesh3d.html    ← interactive 3-D: Prediction mesh vs GT mesh side-by-side
BraTS2021_00000_meshes/        ← per-region OBJ files for pred and gt
dashboard.html                 ← summary page linking all outputs
```

### Segmentation labels

| Value | Region | Description |
|---|---|---|
| 0 | BG | Background |
| 1 | NCR | Necrotic / non-enhancing tumour core |
| 2 | ED | Peritumoral oedema |
| 3 | ET | GD-enhancing tumour |

Composite regions (derived from labels above):

| Region | Definition |
|---|---|
| WT — Whole Tumour | NCR ∪ ED ∪ ET  (labels 1 + 2 + 3) |
| TC — Tumour Core | NCR ∪ ET  (labels 1 + 3) |

### Report JSON

```json
{
  "case_id": "BraTS2021_00000",
  "status": "completed",
  "pred_path": "output/BraTS2021_00000/BraTS2021_00000_pred.nii.gz",
  "mesh_path": "output/BraTS2021_00000/BraTS2021_00000_wt.obj",
  "missing_flags": {"flair": 0, "t1": 0, "t1ce": 0, "t2": 0},
  "downsample_factor": 1.0,
  "region_volumes": {
    "WT": 12450, "TC": 6200, "ET": 3100, "NCR": 3100, "ED": 6250
  }
}
```

---

## Programmatic usage (Python)

```python
from run_pipeline import process_case
from preprocessing import build_modality_paths

paths  = build_modality_paths("BraTS2021_00000", "./data/BraTS2021_00000")
report = process_case(
    case_id   = "BraTS2021_00000",
    paths     = paths,
    out_dir   = "./output",
    seg_w     = "segmentation-module/model-weight/final_model_unet.pth",
    syn_w     = "synthesis-module/model-weight/epoch_118.pth",
    device    = "cuda",
    roi       = (128, 128, 64),   # None to disable sliding window
    syn_steps = 50,
    max_size  = 240,
)
print(report["region_volumes"])
```

### Quick test with synthetic volumes

```python
import numpy as np, nibabel as nib, tempfile, os
from run_pipeline import process_case

case = "demo"
tmp  = tempfile.mkdtemp()
paths = {}
for mod in ("flair", "t1", "t1ce", "t2"):
    arr = np.random.rand(64, 64, 64).astype(np.float32)
    p   = os.path.join(tmp, f"{case}_{mod}.nii.gz")
    nib.save(nib.Nifti1Image(arr, np.eye(4)), p)
    paths[mod] = p

report = process_case(
    case_id="demo", paths=paths, out_dir="./output",
    seg_w="segmentation-module/model-weight/final_model_unet.pth",
    device="cpu",
)
print(report)
```

---

## Preprocessing conventions

The inference pipeline uses **exactly the same** preprocessing as the training notebook to avoid train/inference mismatch:

| Step | Detail |
|---|---|
| Modality order | `flair → t1 → t1ce → t2`  (indices 0–3) |
| Thresholding | Adaptive percentile clip — computed on **non-zero voxels only** (lower=0.5, upper=99.9) |
| Normalisation | Z-score per channel — mean and std computed on **non-zero voxels only** |
| Label remap | BraTS label `4 → 3`  (ET) |
| Inference | `sliding_window_inference` · roi=(64,64,64) · overlap=0.5 · mode=gaussian |
| Deep supervision | `extract_logits()` unwraps `(main_out, ds_list)` tuple before argmax |

---

## Module overview

| File | Role |
|---|---|
| `run_pipeline.py` | Inference CLI — `process_case()`, model loading, OBJ mesh export |
| `visualize.py` | Visualisation CLI — slice plots, metrics, volume charts, 3-D mesh HTML, dashboard |
| `src/preprocessing.py` | `preprocess_multimodal()`, `adaptive_threshold_per_modality()`, `normalize_per_modality()`, brain mask, NIfTI I/O |
| `src/models/unet3d.py` | `UNet3D` architecture, `UNET_Curriculum` wrapper, `create_unet_curriculum()` |
| `synthesis-module/model/architecture.py` | Diffusion synthesis model |
| `configs/pipeline_config.yaml` | All configuration parameters |

---

## Configuration

Key sections of `configs/pipeline_config.yaml`:

```yaml
preprocessing:
  max_size: 240               # isotropic downsample threshold (voxels)
  clip:
    lower_percentile: 0.5
    upper_percentile: 99.9

synthesis:
  enabled: true
  inference:
    sampling_steps: 50        # fewer = faster, lower quality

segmentation:
  post_processing:
    min_component_size: 100   # remove isolated islands < this size (voxels)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'models'`**
Run from the project root and ensure `src/` is on the path:
```bash
cd /path/to/project
python run_pipeline.py --case-id ...
```

**Out of memory during inference**
Install MONAI and use sliding-window inference:
```bash
pip install monai
python run_pipeline.py --case-id ... --device cuda --roi 96 96 48
```
Or reduce `--max-size 128` to force downsampling before inference.

**Prediction shape mismatch with GT during visualisation**
`visualize.py` automatically resamples the prediction to match the GT shape before computing metrics. No manual intervention needed.

**Synthesis model not loading**
The pipeline warns and falls back to mean-fill automatically. Verify that `synthesis-module/model/architecture.py` exists and the checkpoint path is correct.

**`plotly` not installed — 3-D HTML skipped**
```bash
pip install plotly
```

**`trimesh` not installed — mesh smoothing skipped**
```bash
pip install trimesh
```
Meshes are still exported; Laplacian smoothing is simply skipped.

**GT segmentation not found in `--input-dir`**
`visualize.py` looks for `*_seg.nii.gz` or `*-seg.nii.gz`. Metrics and the GT mesh panel are skipped if no file is found; all other plots are generated normally.