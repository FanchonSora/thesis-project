import os
import sys
import json
import argparse
import importlib
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
try:
    import torch
except ImportError:
    sys.exit("❌  PyTorch not found.")
try:
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    warnings.warn("MONAI not found — falling back to full-volume inference (may OOM).")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    preprocess_multimodal,
    save_nifti,
    build_modality_paths,
    MODALITY_ORDER,        
)
from models.unet3d import create_unet_curriculum

def extract_logits(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out

def _load_state_dict(path: str, device: str = "cpu") -> dict:
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in ck:
                return ck[key]
    return ck

class SynthesisWrapper:
    def __init__(self, ckpt_path: Optional[str], device: str = "cpu"):
        self.device = device
        self.model  = None
        if not ckpt_path or not os.path.exists(ckpt_path):
            print("  ⚠️  Synthesis checkpoint not found — mean-fill fallback.")
            return
        arch_path = Path(__file__).parent / "synthesis-module" / "model" / "architecture.py"
        if not arch_path.exists():
            print(f"  ⚠️  Synthesis architecture not found — mean-fill fallback.")
            return
        try:
            spec    = importlib.util.spec_from_file_location("synthesis_arch", str(arch_path))
            mod     = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            factory = getattr(mod, "create_model", None) or getattr(mod, "DiffusionSynthesisModel", None)
            if factory is None:
                raise ImportError("No recognised constructor in architecture.py")
            self.model = factory()
            self.model.load_state_dict(_load_state_dict(ckpt_path, device), strict=False)
            self.model.to(device).eval()
            print(f"  ✅ Synthesis model loaded from {ckpt_path}")
        except Exception as exc:
            print(f"  ⚠️  Could not load synthesis model ({exc}) — mean-fill fallback.")
            self.model = None

    def synthesize(
        self,
        stacked:       np.ndarray,
        missing_flags: list,
        num_steps:     int = 50,
    ) -> np.ndarray:
        completed = stacked.copy()
        if self.model is not None:
            avail = [0 if int(m) else 1 for m in missing_flags]
            try:
                with torch.no_grad():
                    for i, is_missing in enumerate(missing_flags):
                        if not is_missing:
                            continue
                        x_dict = {
                            name: torch.from_numpy(completed[j:j+1][None]).float().to(self.device)
                            for j, name in enumerate(MODALITY_ORDER)
                        }
                        out   = self.model(
                            x_dict, None,
                            MODALITY_ORDER[i],
                            torch.tensor([i],     dtype=torch.long, device=self.device),
                            torch.tensor([avail], dtype=torch.long, device=self.device),
                            torch.tensor([0],     dtype=torch.long, device=self.device),
                            num_infer_steps=num_steps,
                        )
                        x_hat = out["x_hat"] if isinstance(out, dict) and "x_hat" in out else out
                        if isinstance(x_hat, torch.Tensor):
                            completed[i] = x_hat.squeeze().cpu().numpy()
                        avail[i] = 1
                return completed
            except Exception as exc:
                print(f"  ⚠️  Synthesis inference failed ({exc}) — mean-fill fallback.")
        avail_idx = [i for i, m in enumerate(missing_flags) if not m]
        fill      = (np.mean(completed[avail_idx], axis=0)
                     if avail_idx else np.zeros(completed.shape[1:], dtype=np.float32))
        for i, is_missing in enumerate(missing_flags):
            if is_missing:
                completed[i] = fill
        return completed

def load_trained_model(
    checkpoint_path:      str,
    base_lr:              float = 2e-4,
    weight_decay:         float = 1e-4,
    class_weights                = None,
    use_deep_supervision: bool   = True,
):
    print("⏳ Loading segmentation model …")
    model, _, _, device = create_unet_curriculum(
        base_lr              = base_lr,
        weight_decay         = weight_decay,
        class_weights        = class_weights,
        use_deep_supervision = use_deep_supervision,
    )
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"  ✅ Loaded checkpoint (epoch {ck.get('epoch', '?')})")
    return model, device

def post_process(
    seg:        np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    min_size:   int = 100,
) -> np.ndarray:
    from scipy.ndimage import label as nd_label
    out = seg.copy()
    for cls in np.unique(out):
        if cls == 0:
            continue
        labeled, num = nd_label(out == cls)
        for i in range(1, num + 1):
            if (labeled == i).sum() < min_size:
                out[labeled == i] = 0
    if brain_mask is not None:
        out[brain_mask == 0] = 0
    return out

def export_wt_mesh(pred: np.ndarray, out_prefix: str) -> Optional[str]:
    try:
        from skimage import measure
    except ImportError:
        return None
    wt = (pred > 0).astype(np.float32)
    if wt.sum() < 100:
        print("  ⚠️  Whole-tumour mask too small — mesh skipped.")
        return None
    verts, faces, _, _ = measure.marching_cubes(wt, level=0.5, spacing=(1.0, 1.0, 1.0))
    obj_path = out_prefix + ".obj"
    Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
    with open(obj_path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for f in faces:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    print(f"  ✅ OBJ → {obj_path}")
    return obj_path

def process_case(
    case_id:   str,
    paths:     Dict[str, str],
    out_dir:   str,
    seg_w:     str,
    syn_w:     str  = "",
    device:    str  = "cpu",
    roi:       Optional[Tuple[int, int, int]] = None,
    syn_steps: int  = 50,
    max_size:  int  = 240,
) -> Dict:
    import scipy.ndimage
    import nibabel as nib
    case_out = Path(out_dir) / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    # ── 1. Preprocess ────────────────────────────────────────────────
    print("\n[1/4] Preprocessing …")
    stacked, brain_mask, affine, ds_factor = preprocess_multimodal(
        paths, lower=0.5, upper=99.9, max_size=max_size,
    )
    print(f"  shape={stacked.shape}  ds_factor={ds_factor:.3f}")
    missing_flags = [0 if (paths.get(m, "") and os.path.exists(paths[m])) else 1
                     for m in MODALITY_ORDER]
    n_missing = sum(missing_flags)
    print(f"  missing: {dict(zip(MODALITY_ORDER, missing_flags))}")
    # ── 2. Synthesise ────────────────────────────────────────────────
    if n_missing:
        print(f"\n[2/4] Synthesising {n_missing} modality/ies …")
        stacked = SynthesisWrapper(syn_w, device=device).synthesize(
            stacked, missing_flags, num_steps=syn_steps
        )
    else:
        print("\n[2/4] All modalities present — skipping synthesis.")
    # ── 3. Segment ───────────────────────────────────────────────────
    print("\n[3/4] Segmenting …")
    try:
        unet, _ = load_trained_model(seg_w)
    except Exception as exc:
        print(f"  ⚠️  Could not load model: {exc}")
        return {"case_id": case_id, "status": "failed", "error": str(exc)}
    device_t   = torch.device(device)
    unet.to(device_t).eval()
    x          = torch.from_numpy(stacked[None]).float().to(device_t)
    orig_shape = stacked.shape[1:]
    roi_use    = roi if roi else tuple(min(d, 64) for d in orig_shape)
    infer_scale = 1.0
    pred        = None
    for attempt in range(3):
        try:
            with torch.no_grad():
                x_in = x
                if infer_scale != 1.0:
                    x_np = scipy.ndimage.zoom(
                        x.cpu().numpy(),
                        (1, 1, infer_scale, infer_scale, infer_scale), order=1,
                    )
                    x_in = torch.from_numpy(x_np).to(device_t)
                    print(f"  Inference on shape {tuple(x_in.shape[2:])}")
                if HAS_MONAI:
                    logits = sliding_window_inference(
                        x_in, roi_use, 1,
                        lambda inp: extract_logits(unet, inp),
                        overlap=0.5, mode="gaussian",
                    )
                else:
                    logits = extract_logits(unet, x_in)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            break
        except RuntimeError as exc:
            if "alloc" in str(exc).lower() and attempt < 2:
                infer_scale *= 0.5
                print(f"  💾 OOM — retrying at scale {infer_scale:.2f} …")
            else:
                raise
    if infer_scale != 1.0:
        zoom_f = tuple(orig_shape[i] / pred.shape[i] for i in range(3))
        pred   = scipy.ndimage.zoom(pred, zoom_f, order=0).astype(np.uint8)
    print(f"  unique labels: {np.unique(pred).tolist()}")
    # ── 4. Post-process + save ────────────────────────────────────────
    print("\n[4/4] Post-processing and saving …")
    pred = post_process(pred, brain_mask)
    if ds_factor != 1.0:
        factor = 1.0 / ds_factor
        print(f"  Resampling to native ×{factor:.3f}")
        pred = scipy.ndimage.zoom(pred, factor, order=0).astype(np.uint8)
    if affine is None:
        affine = np.eye(4)
        for k in MODALITY_ORDER:
            p = paths.get(k, "")
            if p and os.path.exists(p):
                affine = nib.load(p).affine
                break
    pred_path = str(case_out / f"{case_id}_pred.nii.gz")
    save_nifti(pred_path, pred, affine)
    print(f"  ✅ Prediction → {pred_path}")
    mesh_path = export_wt_mesh(pred, str(case_out / f"{case_id}_wt"))
    region_volumes = {
        "WT":  int((pred > 0).sum()),
        "TC":  int(np.isin(pred, [1, 3]).sum()),
        "ET":  int((pred == 3).sum()),
        "NCR": int((pred == 1).sum()),
        "ED":  int((pred == 2).sum()),
    }
    print(f"  Region volumes: {region_volumes}")
    report = {
        "case_id":           case_id,
        "status":            "completed",
        "pred_path":         pred_path,
        "mesh_path":         mesh_path,
        "missing_flags":     dict(zip(MODALITY_ORDER, missing_flags)),
        "downsample_factor": float(ds_factor),
        "region_volumes":    region_volumes,
        "affine":            affine.tolist() if affine is not None else None,
    }
    report_path = str(case_out / f"{case_id}_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  ✅ Report     → {report_path}")
    print(f"\n✅ Done — results in {case_out}\n")
    return report

def main():
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir
    parser = argparse.ArgumentParser(
        description="Brain Tumour Segmentation — inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case-id",   required=True,
                        help="Patient / case identifier (e.g. BraTS2021_00000)")
    parser.add_argument("--input-dir", required=True,
                        help="Folder containing NIfTI files")
    parser.add_argument("--out-dir",   default=str(project_root / "output"),
                        help="Root output directory")
    parser.add_argument("--seg-w",
                        default=str(project_root / "segmentation-module" / "model-weight" / "final_model_unet.pth"),
                        help="Segmentation model weights (.pth)")
    parser.add_argument("--syn-w",
                        default=str(project_root / "synthesis-module" / "model-weight" / "epoch_118.pth"),
                        help="Synthesis model weights (.pth)  [optional]")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu", "mps"])
    parser.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "Z"),
                        help="ROI for sliding-window (default: min(dim,64) per axis)")
    parser.add_argument("--syn-steps", type=int, default=50,
                        help="DDIM steps for synthesis model")
    parser.add_argument("--max-size",  type=int, default=240,
                        help="Downsample when any spatial dim > this value")
    args  = parser.parse_args()
    paths = build_modality_paths(args.case_id, args.input_dir)
    roi   = tuple(args.roi) if args.roi else None
    report = process_case(
        case_id   = args.case_id,
        paths     = paths,
        out_dir   = args.out_dir,
        seg_w     = args.seg_w,
        syn_w     = args.syn_w,
        device    = args.device,
        roi       = roi,
        syn_steps = args.syn_steps,
        max_size  = args.max_size,
    )
    sys.exit(0 if report.get("status") == "completed" else 1)

if __name__ == "__main__":
    main()