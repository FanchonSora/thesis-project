from __future__ import annotations
import argparse
import inspect
import json
import logging
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from models.synthesis_module.diffusion_model.unet_brats import create_model  
from models.synthesis_module.diffusion_model.trainer_brats import GaussianDiffusion
import numpy as np
import scipy.ndimage
from PIL import Image

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch not found.") from exc

try:
    from monai.inferers import sliding_window_inference

    HAS_MONAI = True
except Exception:
    HAS_MONAI = False
    sliding_window_inference = None

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from preprocessing import MODALITY_ORDER, build_modality_paths, preprocess_case, save_nifti, voxel_spacing_from_affine

try:
    from models.unet3d import create_unet_curriculum  # type: ignore
except Exception:
    from unet3d import create_unet_curriculum  # type: ignore


LOGGER = logging.getLogger("brain_tumor_pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PREVIEW_PLANES = ("axial", "coronal", "sagittal")
PLANE_TO_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
REGION_COLORS = {
    1: np.array([168, 85, 247], dtype=np.uint8),
    2: np.array([16, 224, 160], dtype=np.uint8),
    3: np.array([255, 71, 87], dtype=np.uint8),
}


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


def extract_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def _load_state_dict(path: str, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def to_jsonable(obj: Any):
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


class SynthesisWrapper:
    """
    Missing-modality synthesis wrapper.

    Important fixes in this version:
    1) `stacked` is already 4-channel in MODALITY_ORDER. We preserve that order.
    2) Fallback mean fill writes into the missing channel in-place instead of concatenating
       an extra channel (the old code could return 5 channels).
    3) `num_steps` is threaded into the sampler when the diffusion implementation supports it.
    4) Extra logging makes it obvious whether synthesis was actually executed or skipped.
    """

    def __init__(self, models_dir: Optional[str], device: str = "cpu"):
        self.device = device
        self.models: Dict[str, torch.nn.Module] = {}
        self.diffusions: Dict[str, torch.nn.Module] = {}
        self.status = "skipped"
        self.error: Optional[str] = None
        self.loaded_targets: list[str] = []

        if not models_dir:
            self.status = "skipped"
            self.error = "Synthesis disabled because syn_w is empty"
            LOGGER.info("[syn] Disabled: syn_w is empty")
            return

        if not os.path.exists(models_dir):
            self.status = "fallback_mean"
            self.error = f"Synthesis models directory not found: {models_dir}"
            LOGGER.warning("[syn] %s", self.error)
            return

        try:
            self.create_model = create_model
            self.GaussianDiffusion = GaussianDiffusion
        except ImportError as exc:
            self.status = "fallback_mean"
            self.error = f"Could not import synthesis modules: {exc}"
            LOGGER.exception("[syn] Failed importing synthesis modules")
            return

        self.model_config = {
            "flair": ("model_flair_from_t1_t1ce_t2.pt", ["t1", "t1ce", "t2"]),
            "t1": ("model_t1_from_t1ce_t2_flair.pt", ["t1ce", "t2", "flair"]),
            "t1ce": ("model_t1ce_from_t1_t2_flair.pt", ["t1", "t2", "flair"]),
            "t2": ("model_t2_from_t1_t1ce_flair.pt", ["t1", "t1ce", "flair"]),
        }

        LOGGER.info("[syn] Looking for synthesis weights in %s", models_dir)

        for target, (model_file, _) in self.model_config.items():
            model_path = os.path.join(models_dir, model_file)
            if not os.path.exists(model_path):
                LOGGER.warning("[syn] Missing weight for %s: %s", target, model_path)
                continue
            try:
                model = self.create_model(
                    image_size=128,
                    num_channels=64,
                    num_res_blocks=2,
                    channel_mult="1,2,3,4",
                    in_channels=4,   # 3 condition channels + noisy target channel
                    out_channels=1,
                ).to(device)

                diffusion = self.GaussianDiffusion(
                    model,
                    image_size=128,
                    depth_size=144,
                    timesteps=250,
                    loss_type="l2",
                    with_condition=True,
                    channels=1,
                ).to(device)

                ckpt = torch.load(model_path, map_location=device)
                if isinstance(ckpt, dict) and "ema" in ckpt:
                    diffusion.load_state_dict(ckpt["ema"], strict=False)
                elif isinstance(ckpt, dict) and "model" in ckpt:
                    diffusion.load_state_dict(ckpt["model"], strict=False)
                else:
                    diffusion.load_state_dict(ckpt, strict=False)

                diffusion.eval()
                self.models[target] = model
                self.diffusions[target] = diffusion
                self.loaded_targets.append(target)
                LOGGER.info("[syn] Loaded model for target=%s", target)
            except Exception as exc:
                LOGGER.exception("[syn] Failed to load model for %s", target)
                self.error = f"Failed to load model for {target}: {exc}"

        if self.models:
            self.status = "ready"
            LOGGER.info("[syn] Ready. Loaded targets: %s", self.loaded_targets)
        else:
            self.status = "fallback_mean"
            if self.error is None:
                self.error = "No synthesis models loaded successfully"
            LOGGER.warning("[syn] %s", self.error)

    def _fallback_fill(self, completed_full: np.ndarray, missing_flags: Sequence[int], target_index: int) -> np.ndarray:
        avail_vols = [completed_full[j] for j in range(4) if not missing_flags[j] and j != target_index]
        if avail_vols:
            fill = np.mean(avail_vols, axis=0).astype(np.float32, copy=False)
        else:
            fill = np.zeros_like(completed_full[0], dtype=np.float32)
        completed_full[target_index] = fill
        return completed_full

    def _sample_diffusion(self, diffusion, cond_tensor: torch.Tensor, num_steps: int) -> torch.Tensor:
        sample_fn = getattr(diffusion, "sample")
        kwargs_candidates = [
            {"batch_size": 1, "condition_tensors": cond_tensor, "num_infer_steps": num_steps},
            {"batch_size": 1, "condition_tensors": cond_tensor, "num_steps": num_steps},
            {"batch_size": 1, "condition_tensors": cond_tensor, "timesteps": num_steps},
            {"batch_size": 1, "condition_tensors": cond_tensor},
        ]
        last_exc = None
        for kwargs in kwargs_candidates:
            try:
                sig = inspect.signature(sample_fn)
                accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
                # keep required kwargs even if signature is permissive (*args/**kwargs)
                if "batch_size" not in accepted:
                    accepted["batch_size"] = 1
                if "condition_tensors" not in accepted:
                    accepted["condition_tensors"] = cond_tensor
                return sample_fn(**accepted)
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Could not call diffusion.sample()")

    def synthesize(self, stacked: np.ndarray, missing_flags: Sequence[int], num_steps: int = 50) -> np.ndarray:
        if stacked.ndim != 4 or stacked.shape[0] != 4:
            raise ValueError(f"Expected stacked shape (4, H, W, D), got {stacked.shape}")

        completed_full = stacked.copy().astype(np.float32, copy=False)

        missing_mods = [MODALITY_ORDER[i] for i, flag in enumerate(missing_flags) if flag]
        LOGGER.info("[syn] Requested synthesis for missing modalities: %s", missing_mods)

        if not missing_mods:
            self.status = "skipped"
            LOGGER.info("[syn] No missing modalities. Nothing to synthesize.")
            return completed_full

        if not self.models:
            LOGGER.warning("[syn] No diffusion models available. Falling back to mean fill.")
            for i, is_missing in enumerate(missing_flags):
                if is_missing:
                    completed_full = self._fallback_fill(completed_full, missing_flags, i)
            self.status = "fallback_mean"
            return completed_full

        any_success = False

        for i, is_missing in enumerate(missing_flags):
            if not is_missing:
                continue

            target_mod = MODALITY_ORDER[i]
            LOGGER.info("[syn] Synthesizing target=%s", target_mod)

            if target_mod not in self.diffusions:
                LOGGER.warning("[syn] No model for target=%s. Using fallback mean.", target_mod)
                completed_full = self._fallback_fill(completed_full, missing_flags, i)
                continue

            try:
                cond_mods = self.model_config[target_mod][1]
                cond_indices = [MODALITY_ORDER.index(m) for m in cond_mods]
                cond_volumes = completed_full[cond_indices]  # (3, H, W, D)

                target_shape = (128, 128, 144)  # H, W, D
                cond_resized = np.stack(
                    [
                        scipy.ndimage.zoom(
                            v,
                            (
                                target_shape[0] / v.shape[0],
                                target_shape[1] / v.shape[1],
                                target_shape[2] / v.shape[2],
                            ),
                            order=1,
                        )
                        for v in cond_volumes
                    ],
                    axis=0,
                )

                cond_min = float(cond_resized.min())
                cond_max = float(cond_resized.max())
                if cond_max > cond_min:
                    cond_resized = (cond_resized - cond_min) / (cond_max - cond_min)
                cond_resized = (cond_resized * 2.0) - 1.0

                # (3, H, W, D) -> (1, 3, D, H, W)
                cond_tensor = (
                    torch.from_numpy(cond_resized)
                    .float()
                    .to(self.device)
                    .permute(0, 3, 1, 2)
                    .unsqueeze(0)
                )

                with torch.no_grad():
                    gen = self._sample_diffusion(self.diffusions[target_mod], cond_tensor, num_steps=num_steps)

                synthesized = gen[0, 0].detach().cpu().numpy()  # (D, H, W) expected
                synthesized = (synthesized + 1.0) / 2.0
                synthesized = np.clip(synthesized, 0.0, 1.0)

                original_shape = completed_full.shape[1:]  # (H, W, D)
                synthesized_hwd = np.transpose(synthesized, (1, 2, 0))  # (H, W, D)
                synthesized_resized = scipy.ndimage.zoom(
                    synthesized_hwd,
                    (
                        original_shape[0] / synthesized_hwd.shape[0],
                        original_shape[1] / synthesized_hwd.shape[1],
                        original_shape[2] / synthesized_hwd.shape[2],
                    ),
                    order=1,
                ).astype(np.float32)

                completed_full[i] = synthesized_resized
                any_success = True
                LOGGER.info("[syn] Completed target=%s", target_mod)
            except Exception as exc:
                LOGGER.exception("[syn] Failed generating target=%s. Using fallback mean.", target_mod)
                self.error = f"Synthesis failed for {target_mod}: {exc}"
                completed_full = self._fallback_fill(completed_full, missing_flags, i)

        self.status = "success" if any_success else "fallback_mean"
        return completed_full


def load_trained_model(
    checkpoint_path: str,
    base_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    class_weights=None,
    use_deep_supervision: bool = True,
):
    LOGGER.info("Loading segmentation model from %s", checkpoint_path)
    model, _, _, device = create_unet_curriculum(
        base_lr=base_lr,
        weight_decay=weight_decay,
        class_weights=class_weights,
        use_deep_supervision=use_deep_supervision,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info("Loaded checkpoint epoch=%s", checkpoint.get("epoch", "?"))
    return model, device


def post_process(seg: np.ndarray, brain_mask: Optional[np.ndarray] = None, min_size: int = 100) -> np.ndarray:
    from scipy.ndimage import label as nd_label

    out = seg.copy()
    for cls in np.unique(out):
        if int(cls) == 0:
            continue
        labeled, num = nd_label(out == cls)
        for i in range(1, num + 1):
            if int((labeled == i).sum()) < min_size:
                out[labeled == i] = 0
    if brain_mask is not None and brain_mask.shape == out.shape:
        out[brain_mask == 0] = 0
    return out.astype(np.uint8)


def _write_obj(mask: np.ndarray, obj_path: Path) -> Optional[str]:
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
        results[lod] = _write_obj(region_mask, Path(f"{out_prefix}_{lod}.obj"))
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
        results[lod] = _write_obj(mask, Path(f"{out_prefix}_{lod}.obj"))
    return results


def _compute_region_volumes(pred: np.ndarray, affine: Optional[np.ndarray]) -> Tuple[Dict[str, int], Dict[str, float]]:
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


def _run_segmentation(
    model: torch.nn.Module,
    stacked: np.ndarray,
    device: str,
    roi: Optional[Tuple[int, int, int]],
) -> np.ndarray:
    device_t = torch.device(device)
    model.to(device_t).eval()
    x = torch.from_numpy(stacked[None]).float().to(device_t)
    orig_shape = stacked.shape[1:]
    roi_use = roi if roi else (64, 64, 64)
    infer_scale = 1.0
    pred: Optional[np.ndarray] = None

    for attempt in range(3):
        try:
            with torch.no_grad():
                x_in = x
                if infer_scale != 1.0:
                    x_np = scipy.ndimage.zoom(
                        x.cpu().numpy(),
                        (1, 1, infer_scale, infer_scale, infer_scale),
                        order=1,
                    )
                    x_in = torch.from_numpy(x_np).to(device_t)
                    LOGGER.warning("OOM retry; inference at shape=%s", tuple(x_in.shape[2:]))

                if HAS_MONAI:
                    logits = sliding_window_inference(
                        x_in,
                        roi_use,
                        sw_batch_size=1,
                        predictor=lambda inp: extract_logits(model, inp),
                        overlap=0.5,
                        mode="gaussian",
                    )
                else:
                    logits = extract_logits(model, x_in)

            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            break
        except RuntimeError as exc:
            msg = str(exc).lower()
            oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg) or "alloc" in msg
            if oom and attempt < 2:
                infer_scale *= 0.5
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    if pred is None:
        raise RuntimeError("Segmentation inference failed without producing output")

    if infer_scale != 1.0:
        zoom_f = tuple(orig_shape[i] / pred.shape[i] for i in range(3))
        pred = scipy.ndimage.zoom(pred, zoom_f, order=0).astype(np.uint8)

    return pred


def _pick_slice(seg: np.ndarray, axis: int) -> int:
    collapsed = np.sum(seg > 0, axis=tuple(i for i in range(3) if i != axis))
    idx = int(np.argmax(collapsed))
    return idx if int(collapsed[idx]) > 0 else seg.shape[axis] // 2


def _extract_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return volume[idx, :, :]
    if axis == 1:
        return volume[:, idx, :]
    return volume[:, :, idx]


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
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
    base = np.rot90(_normalize_to_uint8(base_slice))
    seg = np.rot90(seg_slice.astype(np.uint8))
    rgb = np.stack([base, base, base], axis=-1)
    for label, color in REGION_COLORS.items():
        mask = seg == label
        if not np.any(mask):
            continue
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
    return rgb


def save_preview_images(case_id: str, out_dir: Path, pred: np.ndarray, base_volume: np.ndarray) -> Dict[str, Optional[str]]:
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


def process_case(
    case_id: str,
    paths: Dict[str, str],
    out_dir: str,
    seg_w: str,
    syn_w: str = "",
    device: str = "cpu",
    roi: Optional[Tuple[int, int, int]] = None,
    syn_steps: int = 50,
    max_size: int = 240,
    post_min_size: int = 100,
    generate_mesh: bool = True,
) -> Dict[str, Any]:
    case_out = Path(out_dir) / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    errors = []

    config = PipelineConfig(
        seg_w=seg_w,
        syn_w=syn_w,
        device=device,
        roi=roi,
        syn_steps=syn_steps,
        max_size=max_size,
        post_min_size=post_min_size,
        generate_mesh=generate_mesh,
    )

    try:
        LOGGER.info("[1/4] Preprocessing case=%s", case_id)
        prep = preprocess_case(paths=paths, lower=0.5, upper=99.9, max_size=max_size)
        stacked = prep.stacked
        missing_flags = {m: 0 if m in prep.available_modalities else 1 for m in MODALITY_ORDER}

        LOGGER.info("[2/4] Synthesis")
        synthesis = SynthesisWrapper(syn_w, device=device)
        if syn_w and any(missing_flags.values()):
            stacked = synthesis.synthesize(
                stacked,
                [missing_flags[m] for m in MODALITY_ORDER],
                num_steps=syn_steps,
            )
            LOGGER.info("[syn] Final synthesis_status=%s", synthesis.status)
        else:
            synthesis.status = "skipped"
            if not syn_w:
                synthesis.error = "Skipped because syn_w is empty"
            elif not any(missing_flags.values()):
                synthesis.error = "Skipped because no modality is missing"
            LOGGER.info("[syn] Skipped. Reason: %s", synthesis.error)

        LOGGER.info("[3/4] Segmentation")
        model, _ = load_trained_model(seg_w)
        pred_raw = _run_segmentation(model, stacked, device=device, roi=roi)
        pred_post = post_process(pred_raw, brain_mask=prep.brain_mask, min_size=post_min_size)

        LOGGER.info("[4/4] Saving outputs")
        pred_raw_path = str(case_out / f"{case_id}_pred_raw.nii.gz")
        pred_post_path = str(case_out / f"{case_id}_pred_post.nii.gz")
        pred_compat_path = str(case_out / f"{case_id}_pred.nii.gz")

        if config.save_raw_prediction:
            save_nifti(pred_raw_path, pred_raw, prep.affine)
        if config.save_post_prediction:
            save_nifti(pred_post_path, pred_post, prep.affine)
            save_nifti(pred_compat_path, pred_post, prep.affine)

        base_volume = stacked[0] if stacked.ndim == 4 else np.asarray(stacked)
        preview_paths = save_preview_images(case_id, case_out, pred_post, base_volume)

        empty_lods = {"low": None, "medium": None, "high": None}
        mesh_paths = export_wt_mesh_lods(pred_post, str(case_out / f"{case_id}_wt")) if generate_mesh else dict(empty_lods)
        brain_mesh_paths = export_brain_mesh_lods(prep.brain_mask, str(case_out / f"{case_id}_brain")) if generate_mesh else dict(empty_lods)
        wt_mesh_paths = export_wt_mesh_lods(pred_post, str(case_out / f"{case_id}_wt_region")) if generate_mesh else dict(empty_lods)
        tc_mesh_paths = export_tc_mesh_lods(pred_post, str(case_out / f"{case_id}_tc_region")) if generate_mesh else dict(empty_lods)
        et_mesh_paths = export_et_mesh_lods(pred_post, str(case_out / f"{case_id}_et_region")) if generate_mesh else dict(empty_lods)
        voxel_counts, mm3 = _compute_region_volumes(pred_post, prep.affine)

        result = PipelineResult(
            case_id=case_id,
            status="completed",
            paths=PipelinePaths(
                pred_raw_path=pred_raw_path,
                pred_post_path=pred_post_path,
                mesh_path=mesh_paths.get("low") or mesh_paths.get("high"),
                mesh_paths=mesh_paths,
                brain_mesh_path=brain_mesh_paths.get("low") or brain_mesh_paths.get("high"),
                brain_mesh_paths=brain_mesh_paths,
                wt_mesh_path=wt_mesh_paths.get("low") or wt_mesh_paths.get("high"),
                wt_mesh_paths=wt_mesh_paths,
                tc_mesh_path=tc_mesh_paths.get("low") or tc_mesh_paths.get("high"),
                tc_mesh_paths=tc_mesh_paths,
                et_mesh_path=et_mesh_paths.get("low") or et_mesh_paths.get("high"),
                et_mesh_paths=et_mesh_paths,
                preview_paths=preview_paths,
                report_path=str(case_out / f"{case_id}_report.json"),
            ),
            missing_flags=missing_flags,
            synthesis_status=synthesis.status,
            downsample_factor=float(prep.ds_factor),
            region_volumes_voxels=voxel_counts,
            region_volumes_mm3=mm3,
            affine=prep.affine.tolist() if prep.affine is not None else None,
            errors=errors,
            metadata={
                "available_modalities": prep.available_modalities,
                "missing_modalities": prep.missing_modalities,
                "preprocess": {
                    "ds_factor": float(prep.ds_factor),
                    "case_shape": list(prep.case_shape),
                    "affine": prep.affine.tolist() if prep.affine is not None else None,
                    "per_modality_info": to_jsonable(
                        {
                            k: {
                                "path": v.path,
                                "original_shape": list(v.original_shape),
                                "processed_shape": list(v.processed_shape),
                                "voxel_spacing_mm": list(v.voxel_spacing_mm),
                                "affine": v.affine.tolist() if v.affine is not None else None,
                            }
                            for k, v in prep.per_modality_info.items()
                        }
                    ),
                },
                "config": to_jsonable(asdict(config)),
                "synthesis_error": synthesis.error,
                "synthesis_loaded_targets": synthesis.loaded_targets,
                "uses_monai": HAS_MONAI,
            },
        )

        result_dict = to_jsonable(result.to_dict())
        with open(result.paths.report_path, "w", encoding="utf-8") as fh:
            json.dump(result_dict, fh, indent=2)
        return result_dict

    except Exception as exc:
        errors.append(traceback.format_exc())
        failed = to_jsonable({"case_id": case_id, "status": "failed", "error": str(exc), "errors": errors})
        with open(case_out / f"{case_id}_report.json", "w", encoding="utf-8") as fh:
            json.dump(failed, fh, indent=2)
        return failed


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir

    parser = argparse.ArgumentParser(description="Brain tumor inference pipeline")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", default=str(project_root / "output"))
    parser.add_argument(
        "--seg-w",
        default=str(project_root / "models" / "segmentation_module" / "model-weight" / "final_model_unet.pth"),
    )
    parser.add_argument(
        "--syn-w",
        default=str(project_root / "models" / "synthesis_module" / "models"),
        help="Path to synthesis models directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
    )
    parser.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--syn-steps", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=240)
    parser.add_argument("--post-min-size", type=int, default=100)
    parser.add_argument("--no-mesh", action="store_true")
    args = parser.parse_args()

    paths = build_modality_paths(args.case_id, args.input_dir)
    report = process_case(
        case_id=args.case_id,
        paths=paths,
        out_dir=args.out_dir,
        seg_w=args.seg_w,
        syn_w=args.syn_w,
        device=args.device,
        roi=tuple(args.roi) if args.roi else None,
        syn_steps=args.syn_steps,
        max_size=args.max_size,
        post_min_size=args.post_min_size,
        generate_mesh=not args.no_mesh,
    )
    raise SystemExit(0 if report.get("status") == "completed" else 1)


if __name__ == "__main__":
    main()
