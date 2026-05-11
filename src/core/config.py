import torch
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent
FRONTEND_ROOT = APP_ROOT / "web_data"
DATA_ROOT = APP_ROOT / "web_data" / "uploads"
OUTPUT_ROOT = APP_ROOT / "web_output"

# Ensure directories exist
DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

PREVIEW_PLANES = ("axial", "coronal", "sagittal")
DEFAULT_MESH_LOD = "low"
