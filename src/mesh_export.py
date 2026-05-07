from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import scipy.ndimage
LOGGER = logging.getLogger("brain_tumor_pipeline")

def write_obj(mask: np.ndarray, obj_path: Path) -> Optional[str]:
    try:
        from skimage import measure
    except Exception:
        return None
    wt = (mask > 0).astype(np.float32)
    if wt.sum() < 100:
        LOGGER.warning("Whole-tumor mask too small; mesh skipped")
        return None
    verts, faces, _, _ = measure.marching_cubes(wt, level=0.5, spacing=(1.0, 1.0, 1.0))
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with obj_path.open("w", encoding="utf-8") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    return str(obj_path)

def export_region_mesh_lods(mask: np.ndarray, out_prefix: str, min_voxels: int = 100) -> Dict[str, Optional[str]]:
    mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) < min_voxels:
        return {"low": None, "medium": None, "high": None}
    lod_to_scale = {"low": 0.45, "medium": 0.7, "high": 1.0}
    results: Dict[str, Optional[str]] = {}
    for lod, scale in lod_to_scale.items():
        if scale == 1.0:
            region_mask = mask
        else:
            reduced = scipy.ndimage.zoom(mask.astype(np.float32), (scale, scale, scale), order=0)
            region_mask = (reduced > 0).astype(np.uint8)
        results[lod] = write_obj(region_mask, Path(f"{out_prefix}_{lod}.obj"))
    return results

def export_wt_mesh_lods(pred: np.ndarray, out_prefix: str) -> Dict[str, Optional[str]]:
    return export_region_mesh_lods((pred > 0).astype(np.uint8), out_prefix, min_voxels=100)

def export_tc_mesh_lods(pred: np.ndarray, out_prefix: str) -> Dict[str, Optional[str]]:
    return export_region_mesh_lods(np.isin(pred, [1, 3]).astype(np.uint8), out_prefix, min_voxels=60)


def export_et_mesh_lods(pred: np.ndarray, out_prefix: str) -> Dict[str, Optional[str]]:
    return export_region_mesh_lods((pred == 3).astype(np.uint8), out_prefix, min_voxels=30)


def export_brain_mesh_lods(brain_mask: np.ndarray, out_prefix: str) -> Dict[str, Optional[str]]:
    if brain_mask is None or int(brain_mask.sum()) < 1000:
        return {"low": None, "medium": None, "high": None}
    lod_to_scale = {"low": 0.45, "medium": 0.7, "high": 1.0}
    results: Dict[str, Optional[str]] = {}
    for lod, scale in lod_to_scale.items():
        if scale == 1.0:
            mask = brain_mask
        else:
            reduced = scipy.ndimage.zoom(brain_mask.astype(np.float32), (scale, scale, scale), order=0)
            mask = reduced.astype(np.uint8)
        results[lod] = write_obj(mask, Path(f"{out_prefix}_{lod}.obj"))
    return results
