"""
Preview image rendering and volume metrics for the pipeline UI.

Generates 2D preview images (axial/coronal/sagittal) overlaying
segmentation predictions on MRI slices, and synthesis preview images.
Also computes region volume statistics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline_types import PLANE_TO_AXIS, PREVIEW_PLANES, REGION_COLORS
from preprocessing import MODALITY_ORDER, voxel_spacing_from_affine

LOGGER = logging.getLogger("brain_tumor_pipeline")


# ---------------------------------------------------------------------------
# Volume metrics
# ---------------------------------------------------------------------------

def compute_region_volumes(pred: np.ndarray, affine: Optional[np.ndarray]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute voxel counts and physical volumes (mm³) for each tumor region."""
    voxel_counts = {
        "WT": int((pred > 0).sum()),
        "TC": int(np.isin(pred, [1, 3]).sum()),
        "ET": int((pred == 3).sum()),
        "NCR": int((pred == 1).sum()),
        "ED": int((pred == 2).sum()),
    }
    sx, sy, sz = voxel_spacing_from_affine(affine)
    voxel_mm3 = sx * sy * sz
    physical = {key: float(value * voxel_mm3) for key, value in voxel_counts.items()}
    return voxel_counts, physical


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _pick_slice(seg: np.ndarray, axis: int) -> int:
    """Pick the slice with the most tumor voxels along the given axis."""
    collapsed = np.sum(seg > 0, axis=tuple(i for i in range(3) if i != axis))
    idx = int(np.argmax(collapsed))
    return idx if int(collapsed[idx]) > 0 else seg.shape[axis] // 2


def _extract_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along the given axis."""
    if axis == 0:
        return volume[idx, :, :]
    if axis == 1:
        return volume[:, idx, :]
    return volume[:, :, idx]


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D float slice to uint8 using percentile windowing."""
    arr = np.asarray(image, dtype=np.float32)
    if not np.any(np.isfinite(arr)):
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.nan_to_num(arr)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _render_preview(base_slice: np.ndarray, seg_slice: np.ndarray) -> np.ndarray:
    """Render a preview image with segmentation overlay."""
    base = np.rot90(_normalize_to_uint8(base_slice))
    seg = np.rot90(seg_slice.astype(np.uint8))
    rgb = np.stack([base, base, base], axis=-1)
    for label, color in REGION_COLORS.items():
        mask = seg == label
        if not np.any(mask):
            continue
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Public preview functions
# ---------------------------------------------------------------------------

def save_preview_images(case_id: str, out_dir: Path, pred: np.ndarray, base_volume: np.ndarray) -> Dict[str, Optional[str]]:
    """Save segmentation preview images for axial, coronal, sagittal views."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_paths: Dict[str, Optional[str]] = {}
    for plane in PREVIEW_PLANES:
        axis = PLANE_TO_AXIS[plane]
        idx = _pick_slice(pred, axis)
        image = _render_preview(_extract_slice(base_volume, axis, idx), _extract_slice(pred, axis, idx))
        path = out_dir / f"{case_id}_{plane}.png"
        Image.fromarray(image).save(path)
        preview_paths[plane] = str(path)
    return preview_paths


def save_synthesis_previews(
    case_id: str,
    out_dir: Path,
    stacked: np.ndarray,
    missing_flags: Dict[str, int],
    synthesis_status: str,
) -> Dict[str, Optional[str]]:
    """
    Save axial preview images for each synthesized modality.
    Returns dict mapping modality name -> file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Optional[str]] = {}

    if synthesis_status not in ("success", "fallback_mean"):
        return result

    for i, mod in enumerate(MODALITY_ORDER):
        if not missing_flags.get(mod, 0):
            continue
        try:
            vol = stacked[i]  # (H, W, D)
            # Pick the middle axial slice
            mid_idx = vol.shape[2] // 2
            slc = vol[:, :, mid_idx]
            img = np.rot90(_normalize_to_uint8(slc))
            rgb = np.stack([img, img, img], axis=-1)
            path = out_dir / f"{case_id}_syn_{mod}.png"
            Image.fromarray(rgb).save(path)
            result[mod] = str(path)
            LOGGER.info("[syn] Saved synthesis preview for %s -> %s", mod, path)
        except Exception as exc:
            LOGGER.warning("[syn] Failed to save synthesis preview for %s: %s", mod, exc)
            result[mod] = None

    return result
