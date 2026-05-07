"""Shared dataclasses, constants and utilities for the brain tumor pipeline."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Constants
PREVIEW_PLANES = ("axial", "coronal", "sagittal")
PLANE_TO_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
REGION_COLORS = {
    1: np.array([168, 85, 247], dtype=np.uint8),
    2: np.array([16, 224, 160], dtype=np.uint8),
    3: np.array([255, 71, 87], dtype=np.uint8),
}

# Dataclasses
@dataclass
class PipelinePaths:
    pred_raw_path: str
    pred_post_path: str
    mesh_path: Optional[str]
    mesh_paths: Dict[str, Optional[str]]
    brain_mesh_path: Optional[str]
    brain_mesh_paths: Dict[str, Optional[str]]
    wt_mesh_path: Optional[str]
    wt_mesh_paths: Dict[str, Optional[str]]
    tc_mesh_path: Optional[str]
    tc_mesh_paths: Dict[str, Optional[str]]
    et_mesh_path: Optional[str]
    et_mesh_paths: Dict[str, Optional[str]]
    preview_paths: Dict[str, Optional[str]]
    synthesis_preview_paths: Dict[str, Optional[str]]
    report_path: str

@dataclass
class PipelineResult:
    case_id: str
    status: str
    paths: PipelinePaths
    missing_flags: Dict[str, int]
    synthesis_status: str
    downsample_factor: float
    region_volumes_voxels: Dict[str, int]
    region_volumes_mm3: Dict[str, float]
    affine: Optional[list]
    errors: list
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PipelineConfig:
    seg_w: str
    syn_w: str = ""
    device: str = "cpu"
    roi: Optional[Tuple[int, int, int]] = None
    syn_steps: int = 50
    max_size: int = 240
    post_min_size: int = 100
    save_raw_prediction: bool = True
    save_post_prediction: bool = True
    generate_mesh: bool = True

# Utilities
def to_jsonable(obj: Any):
    """Recursively convert numpy/Path types to JSON-serializable equivalents."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj
