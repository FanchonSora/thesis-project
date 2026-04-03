from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
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
from preprocessing import (  
    MODALITY_ORDER,
    build_modality_paths,
    preprocess_case,
    save_nifti,
    voxel_spacing_from_affine,
)
import scipy.ndimage
try:
    from models.unet3d import create_unet_curriculum  # type: ignore
except Exception:
    from unet3d import create_unet_curriculum  # type: ignore

LOGGER = logging.getLogger("brain_tumor_pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

@dataclass
class PipelinePaths:
    pred_raw_path: str
    pred_post_path: str
    mesh_path: Optional[str]
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

def to_jsonable(obj):
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
    def __init__(self, ckpt_path: Optional[str], device: str = "cpu"):
        self.device = device
        self.model = None
        self.status = "skipped"
        self.error: Optional[str] = None
        if not ckpt_path or not os.path.exists(ckpt_path):
            self.status = "fallback_mean"
            self.error = "Synthesis checkpoint not found"
            return
        arch_path = CURRENT_DIR / "synthesis-module" / "model" / "architecture.py"
        if not arch_path.exists():
            self.status = "fallback_mean"
            self.error = "Synthesis architecture.py not found"
            return
        try:
            spec = importlib.util.spec_from_file_location("synthesis_arch", str(arch_path))
            if spec is None or spec.loader is None:
                raise ImportError("Could not build module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            factory = getattr(module, "create_model", None) or getattr(module, "DiffusionSynthesisModel", None)
            if factory is None:
                raise ImportError("No recognised constructor found in synthesis architecture")
            self.model = factory()
            self.model.load_state_dict(_load_state_dict(ckpt_path, device), strict=False)
            self.model.to(device).eval()
            self.status = "ready"
        except Exception as exc:
            self.status = "fallback_mean"
            self.error = str(exc)
            self.model = None
    def synthesize(self, stacked: np.ndarray, missing_flags: Sequence[int], num_steps: int = 50) -> np.ndarray:
        completed = stacked.copy()
        if self.model is not None:
            availability = [0 if int(flag) else 1 for flag in missing_flags]
            try:
                with torch.no_grad():
                    for i, is_missing in enumerate(missing_flags):
                        if not is_missing:
                            continue
                        x_dict = {
                            name: torch.from_numpy(completed[j : j + 1][None]).float().to(self.device)
                            for j, name in enumerate(MODALITY_ORDER)
                        }
                        out = self.model(
                            x_dict,
                            None,
                            MODALITY_ORDER[i],
                            torch.tensor([i], dtype=torch.long, device=self.device),
                            torch.tensor([availability], dtype=torch.long, device=self.device),
                            torch.tensor([0], dtype=torch.long, device=self.device),
                            num_infer_steps=num_steps,
                        )
                        x_hat = out["x_hat"] if isinstance(out, dict) and "x_hat" in out else out
                        if isinstance(x_hat, torch.Tensor):
                            completed[i] = x_hat.squeeze().detach().cpu().numpy()
                        availability[i] = 1
                self.status = "success"
                return completed
            except Exception as exc:
                self.status = "fallback_mean"
                self.error = f"Synthesis inference failed: {exc}"
        avail_idx = [i for i, m in enumerate(missing_flags) if not m]
        fill = np.mean(completed[avail_idx], axis=0) if avail_idx else np.zeros(completed.shape[1:], dtype=np.float32)
        for i, is_missing in enumerate(missing_flags):
            if is_missing:
                completed[i] = fill
        return completed

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

def export_wt_mesh(pred: np.ndarray, out_prefix: str) -> Optional[str]:
    try:
        from skimage import measure
    except Exception:
        return None
    wt = (pred > 0).astype(np.float32)
    if wt.sum() < 100:
        LOGGER.warning("Whole-tumor mask too small; mesh skipped")
        return None
    verts, faces, _, _ = measure.marching_cubes(wt, level=0.5, spacing=(1.0, 1.0, 1.0))
    obj_path = out_prefix + ".obj"
    Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
    with open(obj_path, "w", encoding="utf-8") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    return obj_path

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
                    x_np = scipy.ndimage.zoom(x.cpu().numpy(), (1, 1, infer_scale, infer_scale, infer_scale), order=1)
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
            oom = "out of memory" in msg or "cuda" in msg and "memory" in msg or "alloc" in msg
            if oom and attempt < 2:
                infer_scale *= 0.5
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    if pred is None:
        raise RuntimeError("Segmentation inference failed without producing output")
    if infer_scale != 1.0:
        import scipy.ndimage
        zoom_f = tuple(orig_shape[i] / pred.shape[i] for i in range(3))
        pred = scipy.ndimage.zoom(pred, zoom_f, order=0).astype(np.uint8)
    return pred

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
    )
    try:
        LOGGER.info("[1/4] Preprocessing case=%s", case_id)
        prep = preprocess_case(paths=paths, lower=0.5, upper=99.9, max_size=max_size)
        stacked = prep.stacked
        missing_flags = {m: 0 if m in prep.available_modalities else 1 for m in MODALITY_ORDER}
        LOGGER.info("[2/4] Synthesis")
        synthesis = SynthesisWrapper(syn_w, device=device)
        if any(missing_flags.values()):
            stacked = synthesis.synthesize(stacked, [missing_flags[m] for m in MODALITY_ORDER], num_steps=syn_steps)
        else:
            synthesis.status = "skipped"
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
        mesh_path = export_wt_mesh(pred_post, str(case_out / f"{case_id}_wt"))
        voxel_counts, mm3 = _compute_region_volumes(pred_post, prep.affine)
        result = PipelineResult(
            case_id=case_id,
            status="completed",
            paths=PipelinePaths(
                pred_raw_path=pred_raw_path,
                pred_post_path=pred_post_path,
                mesh_path=mesh_path,
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
                    "per_modality_info": to_jsonable({
                        k: {
                            "path": v.path,
                            "original_shape": list(v.original_shape),
                            "processed_shape": list(v.processed_shape),
                            "voxel_spacing_mm": list(v.voxel_spacing_mm),
                            "affine": v.affine.tolist() if v.affine is not None else None,
                        }
                        for k, v in prep.per_modality_info.items()
                    }),
                },
                "config": to_jsonable(asdict(config)),
                "synthesis_error": synthesis.error,
                "uses_monai": HAS_MONAI,
            },
        )

        result_dict = to_jsonable(result.to_dict())

        with open(result.paths.report_path, "w", encoding="utf-8") as fh:
            json.dump(result_dict, fh, indent=2)

        return result_dict
    except Exception as exc:
        errors.append(traceback.format_exc())
        failed = to_jsonable({
            "case_id": case_id,
            "status": "failed",
            "error": str(exc),
            "errors": errors,
        })

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
        default=str(project_root / "segmentation-module" / "model-weight" / "final_model_unet.pth"),
    )
    parser.add_argument(
        "--syn-w",
        default=str(project_root / "synthesis-module" / "model-weight" / "epoch_118.pth"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--syn-steps", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=240)
    parser.add_argument("--post-min-size", type=int, default=100)
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
    )
    raise SystemExit(0 if report.get("status") == "completed" else 1)

if __name__ == "__main__":
    main()
