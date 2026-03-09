import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.filters import threshold_otsu
from scipy.ndimage import zoom as nd_zoom

_MODALITY_CANDIDATES: Dict[str, list] = {
    "flair": [("_", "flair"), ("-", "flair"), ("_", "t2f"), ("-", "t2f")],
    "t1":    [("_", "t1"),    ("-", "t1"),    ("_", "t1n"), ("-", "t1n")],
    "t1ce":  [("_", "t1ce"),  ("-", "t1ce"),  ("_", "t1c"), ("-", "t1c")],
    "t2":    [("_", "t2"),    ("-", "t2"),    ("_", "t2w"), ("-", "t2w")],
}
MODALITY_ORDER = ("flair", "t1", "t1ce", "t2")

def _scan_folder_for_prefix(folder: str) -> Optional[str]:
    all_suffixes = {f"{sep}{suf}"
                    for cands in _MODALITY_CANDIDATES.values()
                    for sep, suf in cands}
    try:
        files = [f for f in os.listdir(folder)
                 if f.endswith(".nii.gz") or f.endswith(".nii")]
    except OSError:
        return None
    votes: Dict[str, int] = {}
    for fname in files:
        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        for suf in all_suffixes:
            if stem.endswith(suf):
                prefix = stem[: -len(suf)]
                votes[prefix] = votes.get(prefix, 0) + 1
                break
    return max(votes, key=lambda k: votes[k]) if votes else None

def detect_modality_paths(case_id: str, folder: str) -> Dict[str, str]:
    folder = str(folder)
    def _try_prefix(prefix: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for mod_key, candidates in _MODALITY_CANDIDATES.items():
            found = ""
            for sep, suffix in candidates:
                for ext in (".nii.gz", ".nii"):
                    p = os.path.join(folder, f"{prefix}{sep}{suffix}{ext}")
                    if os.path.exists(p):
                        found = p
                        break
                if found:
                    break
            out[mod_key] = found
        return out
    found   = _try_prefix(case_id)
    n_found = sum(bool(v) for v in found.values())
    if n_found == 0:
        actual = _scan_folder_for_prefix(folder)
        if actual and actual != case_id:
            print(f"  [detect] case-id '{case_id}' not matched; "
                  f"using prefix '{actual}' found in folder")
            found   = _try_prefix(actual)
            n_found = sum(bool(v) for v in found.values())
    detected   = [k for k, v in found.items() if v]
    undetected = [k for k, v in found.items() if not v]
    print(f"  [detect] found   : {detected}")
    if undetected:
        print(f"  [detect] missing : {undetected}  (zero-filled)")
    return found

def build_modality_paths(case_id: str, input_dir: str) -> Dict[str, str]:
    subfolder = os.path.join(input_dir, case_id)
    folder    = subfolder if os.path.isdir(subfolder) else input_dir
    return detect_modality_paths(case_id, folder)

def load_nifti(
    path: str,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img  = nib.load(path)
    data = img.get_fdata(dtype=dtype)
    return data, img.affine, img.header

def save_nifti(
    path: str,
    data: np.ndarray,
    affine: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), path)

def adaptive_threshold_per_modality(
    vol: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.9,
) -> np.ndarray:
    for c in range(vol.shape[-1]):
        img = vol[..., c]
        if img.max() > 0:
            nonzero = img[img > 0]
            if len(nonzero) > 10:
                lower = np.percentile(nonzero, lower_percentile)
                upper = np.percentile(nonzero, upper_percentile)
                vol[..., c] = np.clip(img, lower, upper)
    return vol

def fixed_threshold_per_modality(
    vol: np.ndarray,
    threshold_dict: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    if threshold_dict is None:
        threshold_dict = {0: 0.5, 1: 0.2, 2: 0.8, 3: 0.5}
    for c in range(vol.shape[-1]):
        thresh = threshold_dict.get(c, 0)
        vol[..., c][vol[..., c] < thresh] = 0
    return vol

def normalize_per_modality(vol: np.ndarray) -> np.ndarray:
    for c in range(vol.shape[-1]):
        img  = vol[..., c]
        mask = img > 0
        if mask.any():
            m, s = img[mask].mean(), img[mask].std()
            if s > 1e-7:
                vol[..., c][mask] = (img[mask] - m) / s
            else:
                vol[..., c][mask] = img[mask] - m
    return vol

def generate_brain_mask(
    volume:  np.ndarray,   # (H, W, D)  single modality
    closing: int = 5,
    opening: int = 3,
) -> np.ndarray:
    pos = volume[volume > 0]
    if pos.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    try:
        thr = threshold_otsu(volume)
    except Exception:
        thr = np.median(pos)
    mask = (volume > thr).astype(bool)
    mask = ndimage.binary_closing(mask, structure=np.ones((closing, closing, closing)))
    mask = ndimage.binary_opening(mask, structure=np.ones((opening, opening, opening)))
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    sizes    = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)

def preprocess_multimodal(
    paths:    Dict[str, str],
    lower:    float = 0.5,
    upper:    float = 99.9,   
    max_size: int   = 240,    # set < 240 to downsample (e.g. 128 for low-VRAM)
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float]:
    vols:    list = []
    affines: list = []
    HWD:     Optional[Tuple] = None
    ds_factor = 1.0
    # ── 1. Load ──────────────────────────────────────────────────────
    for key in MODALITY_ORDER:
        p = paths.get(key, "")
        if not p or not os.path.exists(p):
            vols.append(None)
            affines.append(None)
            continue
        v, aff, _ = load_nifti(p)
        v = v.astype(np.float32)
        if HWD is None:
            max_dim = max(v.shape)
            if max_dim > max_size:
                ds_factor = max_size / max_dim
                print(f"  ⚠️  Volume {v.shape} > max_size={max_size}; "
                      f"down-sampling ×{ds_factor:.3f}")
        if ds_factor < 1.0:
            v = nd_zoom(v, ds_factor, order=1)
        if HWD is None:
            HWD = v.shape
        vols.append(v)
        affines.append(aff)
    if HWD is None:
        warnings.warn("No valid modality files found — returning zero array (64³).")
        return np.zeros((4, 64, 64, 64), dtype=np.float32), None, None, 1.0
    # ── 2. Assemble (H, W, D, 4) — zero-fill missing channels ────────
    # Layout matches training: np.stack(vols, axis=-1) → (H, W, D, C)
    vol_hwdc = np.zeros((*HWD, 4), dtype=np.float32)
    for i, v in enumerate(vols):
        if v is not None:
            vol_hwdc[..., i] = v
    # ── 3. Brain mask (from first available modality) ─────────────────
    available = [v for v in vols if v is not None]
    brain_mask = generate_brain_mask(available[0]) if available else None
    # ── 4 & 5. Threshold then normalise — same calls as training ──────
    vol_hwdc = adaptive_threshold_per_modality(
        vol_hwdc, lower_percentile=lower, upper_percentile=upper
    )
    vol_hwdc = normalize_per_modality(vol_hwdc)
    # ── 6. Transpose to (C, H, W, D) ─────────────────────────────────
    # Matches training: X = torch.from_numpy(vol_patch).permute(3,0,1,2)
    stacked = vol_hwdc.transpose(3, 0, 1, 2).copy()   # (4, H, W, D)
    first_affine = next((a for a in affines if a is not None), None)
    return stacked, brain_mask, first_affine, ds_factor

class IntensityNormalizer:
    def __init__(self, lower: float = 0.5, upper: float = 99.9):
        self.lower = lower
        self.upper = upper

    def normalize(self, vol: np.ndarray) -> np.ndarray:
        vol = adaptive_threshold_per_modality(vol, self.lower, self.upper)
        vol = normalize_per_modality(vol)
        return vol