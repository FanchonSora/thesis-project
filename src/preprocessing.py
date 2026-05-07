from __future__ import annotations
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom as nd_zoom
from skimage.filters import threshold_otsu

_MODALITY_CANDIDATES: Dict[str, list] = {
    "flair": [("_", "flair"), ("-", "flair"), ("_", "t2f"), ("-", "t2f")],
    "t1": [("_", "t1"), ("-", "t1"), ("_", "t1n"), ("-", "t1n")],
    "t1ce": [("_", "t1ce"), ("-", "t1ce"), ("_", "t1c"), ("-", "t1c")],
    "t2": [("_", "t2"), ("-", "t2"), ("_", "t2w"), ("-", "t2w")],
}
MODALITY_ORDER = ("flair", "t1", "t1ce", "t2")

@dataclass
class VolumeInfo:
    path: str
    original_shape: Tuple[int, int, int]
    processed_shape: Tuple[int, int, int]
    voxel_spacing_mm: Tuple[float, float, float]
    affine: Optional[np.ndarray]

@dataclass
class PreprocessResult:
    stacked: np.ndarray  # (C, H, W, D)
    brain_mask: Optional[np.ndarray]
    affine: Optional[np.ndarray]
    ds_factor: float
    case_shape: Tuple[int, int, int]
    available_modalities: List[str]
    missing_modalities: List[str]
    per_modality_info: Dict[str, VolumeInfo]
    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        if self.affine is not None:
            out["affine"] = self.affine.tolist()
        for key, value in out["per_modality_info"].items():
            if value.get("affine") is not None:
                value["affine"] = np.asarray(value["affine"]).tolist()
        return out

@dataclass
class RawLoadResult:
    """Result of loading raw volumes without any normalization.

    Used when synthesis is needed: raw volumes are kept separate from
    normalization so that synthesis preprocessing and segmentation
    preprocessing can be applied independently.
    """
    raw_vols: Dict[str, Optional[np.ndarray]]  # modality -> raw volume (H, W, D) or None
    affine: Optional[np.ndarray]
    ds_factor: float
    case_shape: Tuple[int, int, int]
    available_modalities: List[str]
    missing_modalities: List[str]
    per_modality_info: Dict[str, VolumeInfo]
    brain_mask: Optional[np.ndarray]

def _scan_folder_for_prefix(folder: str) -> Optional[str]:
    all_suffixes = {
        f"{sep}{suf}"
        for candidates in _MODALITY_CANDIDATES.values()
        for sep, suf in candidates
    }
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".nii") or f.endswith(".nii.gz")]
    except OSError:
        return None
    votes: Dict[str, int] = {}
    for fname in files:
        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        for suffix in all_suffixes:
            if stem.endswith(suffix):
                prefix = stem[: -len(suffix)]
                votes[prefix] = votes.get(prefix, 0) + 1
                break
    return max(votes, key=votes.get) if votes else None

def detect_modality_paths(case_id: str, folder: str) -> Dict[str, str]:
    folder = str(folder)
    def _try_prefix(prefix: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for mod_key, candidates in _MODALITY_CANDIDATES.items():
            found = ""
            for sep, suffix in candidates:
                for ext in (".nii.gz", ".nii"):
                    candidate = os.path.join(folder, f"{prefix}{sep}{suffix}{ext}")
                    if os.path.exists(candidate):
                        found = candidate
                        break
                if found:
                    break
            out[mod_key] = found
        return out
    found = _try_prefix(case_id)
    if not any(found.values()):
        actual_prefix = _scan_folder_for_prefix(folder)
        if actual_prefix and actual_prefix != case_id:
            print(f"  [detect] case-id '{case_id}' not matched; using prefix '{actual_prefix}'")
            found = _try_prefix(actual_prefix)
    detected = [k for k, v in found.items() if v]
    missing = [k for k, v in found.items() if not v]
    print(f"  [detect] found   : {detected}")
    if missing:
        print(f"  [detect] missing : {missing}")
    return found

def build_modality_paths(case_id: str, input_dir: str) -> Dict[str, str]:
    subfolder = os.path.join(input_dir, case_id)
    folder = subfolder if os.path.isdir(subfolder) else input_dir
    return detect_modality_paths(case_id, folder)

def load_nifti(path: str, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(path)
    data = img.get_fdata(dtype=dtype)
    return data, img.affine, img.header

def save_nifti(path: str, data: np.ndarray, affine: Optional[np.ndarray] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)

def voxel_spacing_from_affine(affine: Optional[np.ndarray]) -> Tuple[float, float, float]:
    if affine is None:
        return (1.0, 1.0, 1.0)
    aff = np.asarray(affine, dtype=np.float64)
    return tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))

def adaptive_threshold_per_modality(
    vol: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.9,
) -> np.ndarray:
    out = vol.copy()
    for c in range(out.shape[-1]):
        img = out[..., c]
        nonzero = img[img > 0]
        if nonzero.size > 10:
            lo = np.percentile(nonzero, lower_percentile)
            hi = np.percentile(nonzero, upper_percentile)
            out[..., c] = np.clip(img, lo, hi)
    return out

def normalize_per_modality(vol: np.ndarray) -> np.ndarray:
    out = vol.copy()
    for c in range(out.shape[-1]):
        img = out[..., c]
        mask = img > 0
        if not np.any(mask):
            continue
        mean = float(img[mask].mean())
        std = float(img[mask].std())
        if std > 1e-7:
            out[..., c][mask] = (img[mask] - mean) / std
        else:
            out[..., c][mask] = img[mask] - mean
    return out

def generate_brain_mask(
    volume: np.ndarray,
    strategy: str = "nonzero",
    closing: int = 5,
    opening: int = 3,
) -> np.ndarray:
    if strategy == "none":
        return np.ones_like(volume, dtype=np.uint8)
    pos = volume[volume > 0]
    if pos.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    if strategy == "nonzero":
        mask = volume > 0
    else:
        try:
            thr = threshold_otsu(pos)
        except Exception:
            thr = np.median(pos)
        mask = volume > thr
    mask = ndimage.binary_closing(mask, structure=np.ones((closing, closing, closing), dtype=bool))
    mask = ndimage.binary_opening(mask, structure=np.ones((opening, opening, opening), dtype=bool))
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)

def _resize_if_needed(volume: np.ndarray, ds_factor: float, order: int) -> np.ndarray:
    if ds_factor >= 1.0:
        return volume
    return nd_zoom(volume, ds_factor, order=order)


# ---------------------------------------------------------------------------
# Raw loading (no normalization) — used when synthesis path is needed
# ---------------------------------------------------------------------------

def load_raw_volumes(
    paths: Dict[str, str],
    max_size: int = 240,
    brain_mask_strategy: str = "nonzero",
) -> RawLoadResult:
    """Load raw NIfTI volumes WITHOUT any normalization.

    This is the first step when the synthesis path is active.
    Returns raw float32 volumes, optionally down-sampled, along with
    metadata needed by downstream stages.
    """
    raw_vols: Dict[str, Optional[np.ndarray]] = {}
    infos: Dict[str, VolumeInfo] = {}
    first_affine: Optional[np.ndarray] = None
    case_shape: Optional[Tuple[int, int, int]] = None
    ds_factor = 1.0

    for modality in MODALITY_ORDER:
        path = paths.get(modality, "")
        if not path or not os.path.exists(path):
            raw_vols[modality] = None
            continue

        vol, affine, header = load_nifti(path)
        vol = vol.astype(np.float32, copy=False)

        if first_affine is None:
            first_affine = affine

        if case_shape is None:
            max_dim = max(vol.shape)
            if max_dim > max_size:
                ds_factor = max_size / max_dim
                print(f"  [pre-raw] volume {vol.shape} > max_size={max_size}; down-sampling x{ds_factor:.3f}")
            case_shape = (
                tuple(int(round(s * ds_factor)) for s in vol.shape)
                if ds_factor < 1.0
                else tuple(vol.shape)
            )

        vol_proc = _resize_if_needed(vol, ds_factor, order=1)
        raw_vols[modality] = vol_proc

        infos[modality] = VolumeInfo(
            path=path,
            original_shape=tuple(vol.shape),
            processed_shape=tuple(vol_proc.shape),
            voxel_spacing_mm=tuple(float(v) for v in header.get_zooms()[:3]),
            affine=affine,
        )

    available = [m for m in MODALITY_ORDER if raw_vols.get(m) is not None]
    missing = [m for m in MODALITY_ORDER if raw_vols.get(m) is None]

    if case_shape is None:
        case_shape = (64, 64, 64)

    # Generate brain mask from first available raw volume
    first_available_vol = next((raw_vols[m] for m in MODALITY_ORDER if raw_vols.get(m) is not None), None)
    brain_mask = generate_brain_mask(first_available_vol, strategy=brain_mask_strategy) if first_available_vol is not None else None

    return RawLoadResult(
        raw_vols=raw_vols,
        affine=first_affine,
        ds_factor=float(ds_factor),
        case_shape=case_shape,
        available_modalities=available,
        missing_modalities=missing,
        per_modality_info=infos,
        brain_mask=brain_mask,
    )


def apply_segmentation_preprocessing(
    raw_vols: Dict[str, Optional[np.ndarray]],
    case_shape: Tuple[int, int, int],
    lower: float = 0.5,
    upper: float = 99.9,
) -> np.ndarray:
    """Apply segmentation-style preprocessing (adaptive threshold + z-score)
    to raw volumes and return a stacked (C, H, W, D) array.

    This should be called AFTER synthesis has filled missing modalities
    with synthesized raw-scale outputs.
    """
    vol_hwdc = np.zeros((*case_shape, 4), dtype=np.float32)
    for i, modality in enumerate(MODALITY_ORDER):
        vol = raw_vols.get(modality)
        if vol is not None:
            vol_hwdc[..., i] = vol

    # Segmentation preprocessing: adaptive percentile threshold + z-score
    vol_hwdc = adaptive_threshold_per_modality(vol_hwdc, lower_percentile=lower, upper_percentile=upper)
    vol_hwdc = normalize_per_modality(vol_hwdc)

    # (H, W, D, C) → (C, H, W, D)
    stacked = vol_hwdc.transpose(3, 0, 1, 2).copy()
    return stacked


# ---------------------------------------------------------------------------
# Legacy: full preprocess in one shot (used when NO synthesis is needed)
# ---------------------------------------------------------------------------

def preprocess_case(
    paths: Dict[str, str],
    lower: float = 0.5,
    upper: float = 99.9,
    max_size: int = 240,
    brain_mask_strategy: str = "nonzero",
) -> PreprocessResult:
    vols: List[Optional[np.ndarray]] = []
    affines: List[Optional[np.ndarray]] = []
    headers = []
    infos: Dict[str, VolumeInfo] = {}
    case_shape: Optional[Tuple[int, int, int]] = None
    ds_factor = 1.0
    for modality in MODALITY_ORDER:
        path = paths.get(modality, "")
        if not path or not os.path.exists(path):
            vols.append(None)
            affines.append(None)
            continue
        vol, affine, header = load_nifti(path)
        vol = vol.astype(np.float32, copy=False)
        if case_shape is None:
            max_dim = max(vol.shape)
            if max_dim > max_size:
                ds_factor = max_size / max_dim
                print(f"  [pre] volume {vol.shape} > max_size={max_size}; down-sampling x{ds_factor:.3f}")
            case_shape = tuple(int(round(s * ds_factor)) for s in vol.shape) if ds_factor < 1.0 else tuple(vol.shape)
        elif tuple(vol.shape) != tuple(infos[next(iter(infos))].original_shape):
            warnings.warn(
                f"Modality '{modality}' has shape {vol.shape}, which differs from the first modality. "
                "This pipeline assumes pre-registered BraTS-style inputs."
            )
        vol_proc = _resize_if_needed(vol, ds_factor, order=1)
        vols.append(vol_proc)
        affines.append(affine)
        headers.append(header)
        infos[modality] = VolumeInfo(
            path=path,
            original_shape=tuple(vol.shape),
            processed_shape=tuple(vol_proc.shape),
            voxel_spacing_mm=tuple(float(v) for v in header.get_zooms()[:3]),
            affine=affine,
        )
    if case_shape is None:
        warnings.warn("No valid modality files found — returning zero array (64^3).")
        return PreprocessResult(
            stacked=np.zeros((4, 64, 64, 64), dtype=np.float32),
            brain_mask=None,
            affine=None,
            ds_factor=1.0,
            case_shape=(64, 64, 64),
            available_modalities=[],
            missing_modalities=list(MODALITY_ORDER),
            per_modality_info={},
        )
    vol_hwdc = np.zeros((*case_shape, 4), dtype=np.float32)
    for i, vol in enumerate(vols):
        if vol is not None:
            vol_hwdc[..., i] = vol
    available_modalities = [m for m in MODALITY_ORDER if paths.get(m) and os.path.exists(paths[m])]
    missing_modalities = [m for m in MODALITY_ORDER if m not in available_modalities]
    first_available = next((v for v in vols if v is not None), None)
    brain_mask = generate_brain_mask(first_available, strategy=brain_mask_strategy) if first_available is not None else None
    vol_hwdc = adaptive_threshold_per_modality(vol_hwdc, lower_percentile=lower, upper_percentile=upper)
    vol_hwdc = normalize_per_modality(vol_hwdc)
    stacked = vol_hwdc.transpose(3, 0, 1, 2).copy()
    first_affine = next((a for a in affines if a is not None), None)
    return PreprocessResult(
        stacked=stacked,
        brain_mask=brain_mask,
        affine=first_affine,
        ds_factor=float(ds_factor),
        case_shape=tuple(case_shape),
        available_modalities=available_modalities,
        missing_modalities=missing_modalities,
        per_modality_info=infos,
    )

def preprocess_multimodal(
    paths: Dict[str, str],
    lower: float = 0.5,
    upper: float = 99.9,
    max_size: int = 240,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float]:
    result = preprocess_case(paths=paths, lower=lower, upper=upper, max_size=max_size)
    return result.stacked, result.brain_mask, result.affine, result.ds_factor

class IntensityNormalizer:
    def __init__(self, lower: float = 0.5, upper: float = 99.9):
        self.lower = lower
        self.upper = upper
    def normalize(self, vol: np.ndarray) -> np.ndarray:
        return normalize_per_modality(
            adaptive_threshold_per_modality(vol, self.lower, self.upper)
        )
