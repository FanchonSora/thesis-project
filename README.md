# Brain Tumor Analysis Pipeline

End-to-end MRI pipeline: **preprocess → synthesise missing modalities → 3-D segment → visualise**.

```
data/CASE123/
  CASE123_flair.nii.gz
  CASE123_t1.nii.gz
  CASE123_t1ce.nii.gz      ← any of these can be missing
  CASE123_t2.nii.gz
        │
        ▼
  [Preprocessing]   intensity clip (adaptive, 0.5/99.9 pct) + z-score on non-zero voxels + brain mask
        │
        ▼
  [Synthesis]       diffusion model fills missing modalities  (mean-fill fallback)
        │
        ▼
  [Segmentation]    UNet3D  →  0=BG  1=NCR  2=ED  3=ET
        │
        ▼
  [Visualisation]   slice comparison · metrics · region volumes · 3-D mesh (Pred vs GT)
```

---

## Project structure

```
├── configs/
│   └── pipeline_config.yaml          ← all tunable parameters
├── segmentation-module/
│   └── model-weight/
│       └── final_model_unet.pth
├── synthesis-module/
│   ├── model/
│   │   └── architecture.py
│   ├── model-weight/
│   │   └── epoch_118.pth
│   └── requirements.txt
├── src/
│   ├── models/
│   │   └── unet3d.py                 ← UNet3D + UNET_Curriculum
│   └── preprocessing.py              ← intensity normalisation, brain mask, stacking
├── output/                           ← auto-created at runtime
├── run_pipeline.py                   ← inference CLI entry-point
├── visualize.py                      ← visualisation CLI entry-point
└── README.md
```

---

## Installation

> Tested on Python 3.9 – 3.11.  GPU recommended but not required.

### 1. Create and activate a virtual environment

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch

Choose the command for your hardware from https://pytorch.org/get-started/locally/

```bash
# CUDA 11.8  (recommended for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

### 3. Install remaining dependencies

```bash
pip install nibabel scikit-image scipy matplotlib pyyaml
pip install -r synthesis-module/requirements.txt
```

### 4. Optional — recommended for best results

| Package | Purpose | Install |
|---|---|---|
| `monai` | Sliding-window inference on large volumes | `pip install monai` |
| `plotly` | Interactive 3-D HTML mesh viewer | `pip install plotly` |
| `trimesh` | Laplacian mesh smoothing | `pip install trimesh` |

> Without MONAI the model runs a single forward pass on the whole volume.
> For volumes larger than ~128³ this may cause OOM — install MONAI and use `--roi`.

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