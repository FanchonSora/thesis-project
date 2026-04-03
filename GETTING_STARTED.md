# 🧠 Brain Tumor Analysis Pipeline - Getting Started

Welcome! This guide will help you get the entire brain tumor analysis pipeline up and running in just a few minutes.

---

## ⚡ Quick Start (5 minutes)

### Windows
```powershell
cd c:\Source\Thesis\thesis-project
.\start_server.ps1
```

### Linux/Mac
```bash
cd ~/thesis-project
chmod +x start_server.sh
./start_server.sh
```

Then open your browser and go to: **http://localhost:8000**

---

## 📋 Prerequisites

### 1. Python & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install all dependencies
pip install fastapi uvicorn torch torchvision torchaudio
pip install nibabel numpy scipy scikit-image pillow
pip install tqdm pyyaml tensorboard
```

### 2. Model Weights
The pipeline requires two pre-trained models:

| Model | Location | Size | Purpose |
|-------|----------|------|---------|
| **Segmentation** | `models/segmentaion-module/model-weight/final_model_unet.pth` | ~100MB | Tumor segmentation (UNet3D) |
| **Synthesis** | `models/synthesis-module/model-weight/epoch_118.pth` | ~500MB | Missing modality synthesis (Diffusion) |

**Status**: ✅ Both models are already in the repository

### 3. GPU (Recommended but Optional)
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If True: GPU acceleration enabled ✅
# If False: Will run on CPU (slower, ~2-5x slower)
```

---

## 🚀 Starting the Server

### Option 1: Quick Start Scripts (Recommended)

**Windows:**
```powershell
.\start_server.ps1
```

**Linux/Mac:**
```bash
./start_server.sh
```

**With custom settings:**
```powershell
# Windows - different host/port/workers
.\start_server.ps1 -Host "localhost" -Port 8080 -Workers 4
```

### Option 2: Manual Start

```bash
# Ensure venv is activated
python -m uvicorn src.web_api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Production Mode (Multiple Workers)

```bash
python -m uvicorn src.web_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level warning
```

---

## 🌐 Access Points

Once the server is running:

| Link | Purpose |
|------|---------|
| **http://localhost:8000** | 🎨 Main Web UI (upload & analyze) |
| **http://localhost:8000/docs** | 📚 Interactive API Documentation |
| **http://localhost:8000/api/health** | ❤️ Health Check Endpoint |

---

## 📊 Using the Web Interface

### Step 1: Enter Case ID
Type a case identifier (e.g., `BraTS2021_00000`)

### Step 2: Upload MRI Modalities
Upload at least **3 out of 4** modalities:
- ✅ **T1** - Structural image
- ✅ **T1ce** - T1 with contrast enhancement
- ✅ **T2** - Fast spin echo
- ✅ **FLAIR** - Fluid-suppressed image

**File format**: `.nii` or `.nii.gz` (NIfTI format)
**Max size**: 500 MB per file
**Drag & drop**: Supported! Drag files directly onto upload cards

### Step 3: Submit for Analysis
Click **"Analyze Brain"** button
- The upload will be validated
- A job ID will be generated
- Progress will update in real-time

### Step 4: View Results
Once processing completes, view:
- **Metrics Tab** - Tumor statistics (volumes, percentages)
- **Details Tab** - Segmentation data and modality info
- **Volumes Tab** - 3D mesh visualization

### Step 5: Download Results
- 📄 **Report JSON** - Complete analysis results
- 🧬 **Prediction** - Segmentation masks as NIfTI file
- 🎨 **3D Mesh** - OBJ file for 3D visualization

---

## 🔧 API Reference (For Developers)

### Create Analysis Job
```bash
curl -X POST http://localhost:8000/jobs \
  -F "case_id=BraTS2021_00000" \
  -F "t1=@t1.nii.gz" \
  -F "t1ce=@t1ce.nii.gz" \
  -F "t2=@t2.nii.gz" \
  -F "flair=@flair.nii.gz"
```

**Response:**
```json
{
  "job_id": "abc123-def456-...",
  "case_id": "BraTS2021_00000",
  "status": "QUEUED"
}
```

### Check Job Status
```bash
curl http://localhost:8000/jobs/abc123-def456-...
```

**Response:**
```json
{
  "job_id": "abc123-def456-...",
  "status": "PROCESSING",
  "progress": 45,
  "started_at": "2024-01-15T10:30:00",
  "results": null
}
```

### Download Results
```bash
# Get prediction file
curl -O http://localhost:8000/jobs/abc123-def456-.../file/prediction

# Get report JSON
curl -O http://localhost:8000/jobs/abc123-def456-.../file/report

# Get 3D mesh
curl -O http://localhost:8000/jobs/abc123-def456-.../file/mesh
```

---

## 📂 Project Structure

```
thesis-project/
├── src/
│   ├── web_api.py              # FastAPI backend
│   ├── web_data/               # Frontend files
│   │   ├── index.html          # Web UI
│   │   ├── style.css           # Styling
│   │   └── app.js              # Frontend logic
│   ├── preprocessing.py        # Data preparation
│   └── visualize/
│       └── visualize_results.py  # Result visualization
│
├── models/
│   ├── segmentaion-module/     # UNet3D segmentation
│   └── synthesis-module/       # Diffusion synthesis
│
├── configs/
│   └── pipeline_config.yaml    # Pipeline configuration
│
├── start_server.ps1            # Windows quick start
├── start_server.sh             # Linux/Mac quick start
└── SERVER_GUIDE.md             # Detailed server guide
```

---

## 🐛 Troubleshooting

### "Module not found" Error
```bash
# Reactivate virtual environment
# Windows:
.\.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate

# Reinstall dependencies
pip install -r models/synthesis-module/requirements.txt
```

### "Address already in use" Error
```bash
# Port 8000 is in use. Try different port:
python -m uvicorn src.web_api:app --port 8080
```

### Model Weights Not Found
Check these paths:
- `models/segmentaion-module/model-weight/final_model_unet.pth` (must exist)
- `models/synthesis-module/model-weight/epoch_118.pth` (must exist)

### Slow Processing (CPU)
```bash
# Check if GPU is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# If False, install GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"
Reduce batch size or process smaller images:
```python
# In web_api.py, reduce chunk size:
CHUNK_SIZE = 32  # Default: 64
```

---

## 📚 Detailed Guides

For in-depth information, see:
- **[SERVER_GUIDE.md](SERVER_GUIDE.md)** - Complete server deployment & configuration
- **[README.md](README.md)** - Project overview & setup instructions
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - API documentation & integration

---

## 🎯 Common Workflows

### Workflow 1: Analyze Single Patient
1. Collect 4 MRI modalities in NIfTI format
2. Upload via web interface
3. Wait for processing (2-10 minutes depending on resolution)
4. Download segmentation report and 3D mesh

### Workflow 2: Batch Processing (CLI)
```python
from src.web_api import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Submit job
response = client.post("/jobs", 
    data={"case_id": "patient_001"},
    files={
        "t1": ("t1.nii.gz", open("t1.nii.gz", "rb")),
        "t1ce": ("t1ce.nii.gz", open("t1ce.nii.gz", "rb")),
        "t2": ("t2.nii.gz", open("t2.nii.gz", "rb")),
        "flair": ("flair.nii.gz", open("flair.nii.gz", "rb")),
    }
)

job_id = response.json()["job_id"]
# ... poll for completion
```

### Workflow 3: Docker Deployment
```bash
# Build container
docker build -t brain-tumor-analysis .

# Run container
docker run -p 8000:8000 brain-tumor-analysis

# Access at http://localhost:8000
```

---

## 📊 Performance Tips

| Setting | Impact | How |
|---------|--------|-----|
| **GPU** | 5-10x faster | Ensure CUDA installed correctly |
| **Workers** | Better concurrency | Use `--workers 4` for production |
| **Reload** | Dev convenience | Use only in development (`--reload`) |
| **Resolution** | Memory usage | Smaller MRI crops process faster |

---

## ✅ Verification Checklist

Before analyzing your first case, verify:

- [ ] Python 3.9+ installed: `python --version`
- [ ] Virtual environment activated: `pip list` shows `fastapi`
- [ ] Both model files exist (check paths above)
- [ ] Server starts without errors: `python -m uvicorn src.web_api:app --reload`
- [ ] Web UI loads: `http://localhost:8000`
- [ ] Can upload sample NIfTI file
- [ ] Job processing starts (check console for messages)

---

## 🆘 Getting Help

1. **Check logs**: Server prints detailed messages to console
2. **Review API docs**: Visit http://localhost:8000/docs while server runs
3. **See detailed guide**: Read [SERVER_GUIDE.md](SERVER_GUIDE.md)
4. **Check GitHub issues**: Search for similar problems

---

## 🎉 Next Steps

1. ✅ **Start server** using `start_server.ps1` or `start_server.sh`
2. ✅ **Open browser** to http://localhost:8000
3. ✅ **Prepare test data** - Get sample NIfTI files
4. ✅ **Submit analysis** - Follow web interface prompts
5. ✅ **Review results** - Check metrics, visualization, and download outputs

---

**Happy analyzing!** 🧠💙

For production deployment, see [SERVER_GUIDE.md](SERVER_GUIDE.md) sections on Docker and security hardening.
