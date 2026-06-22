# Brain Tumor Segmentation & Synthesis Platform

Welcome to the **Brain Tumor Analysis Platform**, an advanced, full-stack pipeline designed to automate the segmentation and synthesis of MRI modalities for brain tumor analysis.

This project uses Deep Learning (U-Net for segmentation, DDPM for synthesis) and a modern web interface to provide comprehensive 3D visual context for medical professionals and researchers.

---

## 🌟 Key Features

1. **Automated 3D Tumor Segmentation**
   - Segments MRI into three sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
   - Generates high-quality 3D meshes for interactive visualization.
   
2. **Missing Modality Synthesis (Diffusion Models)**
   - Automatically detects missing MRI modalities (among T1, T1ce, T2, FLAIR).
   - Uses conditioned Denoising Diffusion Probabilistic Models (DDPM) to synthesize the missing scans before segmentation.

3. **Advanced 3D Web Viewer**
   - View 2D anatomical planes (axial, coronal, sagittal).
   - Interactive 3D visualization using WebGL, showcasing a semi-transparent brain outer shell for exact anatomical context alongside the tumor sub-regions.
   - Ground Truth (GT) upload and side-by-side comparison for evaluating predictions.

4. **SOLID API Architecture**
   - Built with **FastAPI** for high performance.
   - Clean, modular backend separating routing, job management, file handling, and ML pipelines.

---

## 🏗️ Project Architecture

```text
thesis-project/
├── Dockerfile                  # Container definition for the web API
├── docker-compose.yml          # Docker Compose for easy deployment
├── start_web.sh                # Shell script to start the service natively
├── TRAINING_WORKFLOW.md        # Detailed guide on training the synthesis DDPM models
├── configs/                    # YAML Configurations for models and training
├── models/                     # Deep Learning Model weights
│   ├── segmentation_module/    # U-Net weights for Tumor Segmentation
│   └── synthesis_module/       # DDPM weights for MRI Modality Synthesis
└── src/                        # Main Application Code
    ├── web_api.py              # Main FastAPI application entry point
    ├── run_pipeline.py         # Entry point for the core ML segmentation pipeline
    ├── preprocessing.py        # Z-score normalization for segmentation
    ├── synthesis.py            # Synthesis diffusion model inference
    ├── synthesis_preprocess.py # Percentile normalization for synthesis
    ├── segmentation.py         # U-Net inference wrapper
    ├── mesh_export.py          # Marching cubes and decimation for 3D meshes
    ├── core/                   # Refactored SOLID API modules
    │   ├── config.py           # Path configurations and constants
    │   ├── job_manager.py      # Thread-safe async job state management
    │   ├── pipeline_runner.py  # Ties the API to run_pipeline.py
    │   ├── report_builder.py   # Formats pipeline outputs to JSON payloads
    │   ├── file_handler.py     # Safely saves uploads and resolves paths
    │   └── utils.py            # Generic helpers (e.g., JSON encoders)
    └── web_data/               # Frontend Assets (HTML, CSS, JS)
```

---

## 🚀 Getting Started

### Prerequisites

- **OS:** Linux / Windows (WSL2 recommended)
- **GPU:** NVIDIA GPU with at least 8GB VRAM (CUDA 11.8+ supported)
- **Docker:** If running via containers (requires `nvidia-container-toolkit`)

### Option 1: Quick Start via Docker (Recommended)

1. Ensure you have Docker and Docker Compose installed.
2. Build and spin up the container:
   ```bash
   docker-compose up -d --build
   ```
3. The platform will be running at [http://localhost:8001](http://localhost:8001).

### Option 2: Native Setup

1. Create a Python Virtual Environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install PyTorch (ensure your CUDA version matches):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   ./start_web.sh
   # Or manually: python src/web_api.py
   ```
5. Open your browser to [http://localhost:8001](http://localhost:8001).

---

## 🖥️ Using the Platform

1. **Upload MRI Scans**: On the homepage, enter a unique Case ID. Upload any combination of `FLAIR`, `T1`, `T1ce`, and `T2` NIfTI (`.nii.gz`) files.
2. **Synthesis Toggle**: If any modalities are missing, ensure "Enable Modality Synthesis" is turned on.
3. **Submit**: Click **Run Analysis**. The server will process the files in the background.
4. **View Results**: Once completed, the UI will update to show 2D slices. Click "View 3D Model" to enter the 3D Viewer.
5. **Ground Truth Comparison**: Inside the 3D viewer, you can upload a Ground Truth segmentation `.nii.gz` to automatically generate side-by-side 3D comparisons.

---

## 🧠 Training & Model Modification

The project consists of two independent model modules. For a complete Vietnamese installation and training guide, refer to [`Huong_dan_cai_dat.txt`](Huong_dan_cai_dat.txt) or English installation guide [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

### 1. Segmentation Module (Enhanced UNet)
The segmentation model is built using an Enhanced U-Net architecture. All training and evaluation logic is contained within Jupyter Notebooks located in `models/segmentation_module/`.
- **Training**: Open and run the `unet-novel.ipynb` notebook. Ensure the `TRAIN_DATASET_PATH` is correctly configured to point to your BraTS2021 dataset. You can find dataset BraTS 2021 on www.synapse.org - The RSNA-ASNR-MICCAI BraTS 2021 Challenge makes publicly available the largest and most diverse retrospective cohort of glioma patients. Ample manually-annotated multi-institutional routine clinically-acquired mpMRI scans of glioma are used as the training, validation, and testing data for this year’s BraTS challenge. *Key parameters: 100 Epochs, Batch size 6, Optimizer AdamW, Learning Rate 1e-4, Mixed Precision FP16*.
- **Evaluation**: Use `unet-novel-eval.ipynb` to evaluate the trained model.
- **Weights**: The trained model weights will be saved into `models/segmentation_module/model-weight/`.

### 2. Synthesis Module (Diffusion Models)
The synthesis module uses conditioned Denoising Diffusion Probabilistic Models (DDPM) to synthesize missing MRI modalities.
- Deep learning scripts, architectures, and data preparation tools for synthesis are stored in `models/synthesis_module/`. *Key parameters: 50,000 Epochs, Batch size 2, Optimizer Adam, Learning Rate 2e-5, Save & Sample every 1000 steps*.
- Refer to the [**TRAINING_WORKFLOW.md**](TRAINING_WORKFLOW.md) for a comprehensive A-Z guide on setting up the BraTS datasets, configuring hyperparameters, and executing the `train_brats.py` or `train_all_modalities.sh` script for all diffusion models. You can find both datasets BraTS 2021 on www.synapse.org and BraTS 2023 on www.synapse.org - BraTS Challenge Series.

---

## 🛠️ Codebase Design Notes (SOLID)

The backend (`src/`) has been heavily refactored to conform to **SOLID principles**, specifically the **Single Responsibility Principle**. 

- The main API endpoint (`src/web_api.py`) is clean and delegative.
- Business logic (executing ML models) is separated from state management (`JobManager`) and data serialization (`ReportBuilder`).
- Preprocessing steps for Synthesis (Percentile) and Segmentation (Z-Score) are explicitly isolated in different files to prevent data-leakage or configuration mismatch. 

Enjoy exploring and extending the Brain Tumor Segmentation Platform!
