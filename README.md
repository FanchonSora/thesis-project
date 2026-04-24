# Brain Tumor MRI Analysis Platform

A comprehensive framework for multimodal brain MRI analysis, featuring automatic tumor segmentation, missing modality synthesis using conditional diffusion models, and interactive 3D visualization.

## Overview

This project implements an end-to-end pipeline for brain tumor analysis from multimodal MRI scans (T1, T1CE, T2, FLAIR). The system can handle incomplete data by synthesizing missing modalities and provides both command-line and web-based interfaces for analysis and visualization.

### Key Features

- **Multimodal Segmentation**: 3D UNet-based segmentation of brain tumors into clinically relevant regions (ET, TC, WT)
- **Modality Synthesis**: Conditional diffusion models for reconstructing missing MRI sequences
- **Preprocessing Pipeline**: Adaptive intensity normalization and patch-based training
- **Web Interface**: Interactive upload, analysis, and 3D visualization
- **REST API**: Programmatic access for integration
- **Cross-Dataset Evaluation**: Trained on BraTS 2021, evaluated on BraTS 2023

## System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8-3.11
- **GPU**: NVIDIA GPU with CUDA 11.0+ (recommended for training/synthesis)
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ for datasets and models

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/brain-tumor-analysis.git
cd brain-tumor-analysis
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download Models

Download pre-trained models and place them in the appropriate directories:

- Segmentation model: `models/segmentation_module/model-weight/final_model_unet.pth`
- Synthesis models: `models/synthesis_module/models/` (4 model files)

## Project Structure

```
├── configs/                 # Configuration files
│   ├── pipeline_config.yaml
│   └── synthesis-models/    # Synthesis model configs
├── models/                  # Pre-trained models
│   ├── segmentation_module/
│   └── synthesis_module/
├── src/                     # Source code
│   ├── run_pipeline.py      # Main pipeline script
│   ├── web_api.py          # Web API server
│   ├── preprocessing.py    # Data preprocessing
│   ├── models/
│   │   └── unet3d.py       # Segmentation model
│   ├── visualize/
│   └── web_data/           # Web interface files
├── results/                 # Output directory
└── README.md               # This file
```

## Usage

### Command Line Interface

#### Basic Segmentation

```bash
python src/run_pipeline.py \
  --case-id BraTS-GLI-00001-000 \
  --input-dir /path/to/brats/data \
  --out-dir ./results
```

#### With Synthesis (Missing Modalities)

```bash
python src/run_pipeline.py \
  --case-id BraTS-GLI-00001-000 \
  --input-dir /path/to/brats/data \
  --out-dir ./results \
  --syn-w models/synthesis_module/models
```

#### Full Options

```bash
python src/run_pipeline.py --help
```

### Web Interface

#### Start Server

```bash
python src/web_api.py
```

Server will start at `http://localhost:8000`

#### Web Usage

1. Open browser to `http://localhost:8000`
2. Upload MRI files (T1, T1CE, T2, FLAIR in .nii or .nii.gz format)
3. Enter case ID
4. Click "Analyze" to start processing
5. View results:
   - 2D slices with segmentation overlay
   - 3D brain visualization
   - Volume measurements
   - Download reports and meshes

### REST API

#### Start API Server

```bash
uvicorn src.web_api:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

- `POST /analyze`: Submit analysis job
- `GET /jobs/{job_id}/status`: Check job status
- `GET /jobs/{job_id}/results`: Get results summary
- `GET /jobs/{job_id}/file/{type}`: Download files

## Data Format

### Input Data

- **Format**: NIfTI (.nii or .nii.gz)
- **Modalities**: T1, T1CE, T2, FLAIR
- **Naming**: Standard BraTS convention (e.g., `BraTS-GLI-00001-000-t1.nii.gz`)
- **Resolution**: 1mm³ isotropic (automatically resampled if needed)

### Directory Structure

```
input_directory/
├── BraTS-GLI-00001-000/
│   ├── BraTS-GLI-00001-000-t1.nii.gz
│   ├── BraTS-GLI-00001-000-t1ce.nii.gz
│   ├── BraTS-GLI-00001-000-t2.nii.gz
│   └── BraTS-GLI-00001-000-flair.nii.gz
└── ...
```

## Modules

### Segmentation Module

- **Architecture**: 3D UNet with residual SE blocks and attention gates
- **Input**: 4-channel multimodal MRI (64×64×64 patches)
- **Output**: Voxel-wise tumor segmentation (4 classes)
- **Training**: Patch-based with tumor-focused sampling

### Synthesis Module

- **Architecture**: Conditional diffusion models (DDPM)
- **Purpose**: Synthesize missing modalities from available ones
- **Models**: 4 separate models (one for each target modality)
- **Input**: 3 modalities → Output: 1 synthesized modality

### Visualization Module

- **2D Views**: Axial, coronal, sagittal slices with overlays
- **3D Views**: Interactive brain mesh with tumor regions
- **Formats**: PNG images, OBJ meshes, JSON reports

## Training

### Segmentation Model

```bash
# Requires BraTS 2021 training data
python train_segmentation.py --config configs/segmentation_config.yaml
```

### Synthesis Models

```bash
cd models/synthesis_module
bash scripts/train_all_modalities.sh 5000 2
```

## Evaluation

### Metrics

- **Dice Score**: ET, TC, WT regions
- **Hausdorff Distance**: Boundary accuracy
- **Volume Correlation**: Size estimation

### Cross-Dataset Validation

- Train: BraTS 2021 (1251 cases)
- Test: BraTS 2021 held-out (20%) + BraTS 2023 (219 cases)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configs
   - Use CPU mode: `--device cpu`
   - Enable gradient checkpointing

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Model Loading Errors**
   - Verify model file paths
   - Check PyTorch version compatibility

4. **Web Interface Issues**
   - Clear browser cache
   - Check console for JavaScript errors
   - Ensure port 8000 is available

### Performance Tips

- Use GPU for faster inference
- Process cases individually for memory efficiency
- Use lower LOD for 3D visualization on slower machines

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```
@thesis{your_thesis,
  title={Brain Tumor Analysis from Multimodal MRI using Deep Learning},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

## Contact

For questions or issues:
- Open GitHub issue
- Email: your.email@example.com