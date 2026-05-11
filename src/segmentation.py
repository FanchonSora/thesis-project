from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
import scipy.ndimage
import torch
LOGGER = logging.getLogger("brain_tumor_pipeline")
try:
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except Exception:
    HAS_MONAI = False
    sliding_window_inference = None

def extract_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract logits from model output, handling tuple/list returns."""
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out

def load_trained_model(
    checkpoint_path: str,
    base_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    class_weights=None,
    use_deep_supervision: bool = True,
):
    LOGGER.info("Loading segmentation model from %s", checkpoint_path)
    try:
        from models.unet3d import UNet3D, UNET_Curriculum
    except ImportError:
        from unet3d import UNet3D, UNET_Curriculum
    if class_weights is None:
        class_weights = [1.0, 1.5, 1.2, 2.0]
    base_model = UNet3D(
        input_shape=(4, 64, 64, 64),
        num_classes=4,
        base_filters=32,
        depth=4,
        dropout_rate=0.2,
        use_deep_supervision=use_deep_supervision,
    )
    model = UNET_Curriculum(base_model=base_model, class_weights=class_weights)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loaded checkpoint epoch=%s (model on CPU, will move to %s for inference)", checkpoint.get("epoch", "?"), device)
    return model, device

def run_segmentation(
    model: torch.nn.Module,
    stacked: np.ndarray,
    device: str,
    roi: Optional[Tuple[int, int, int]],
) -> np.ndarray:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device_t = torch.device(device)
    try:
        model.to(device_t).eval()
        x = torch.from_numpy(stacked[None]).float().to(device_t)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            LOGGER.warning("[seg] CUDA OOM when loading model, falling back to CPU")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device_t = torch.device("cpu")
            model.to(device_t).eval()
            x = torch.from_numpy(stacked[None]).float().to(device_t)
        else:
            raise
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
                # On last GPU attempt, fall back to CPU
                if attempt == 1 and device_t.type != "cpu":
                    LOGGER.warning("[seg] GPU OOM after retries, falling back to CPU")
                    device_t = torch.device("cpu")
                    model.to(device_t).eval()
                    x = torch.from_numpy(stacked[None]).float()
                    infer_scale = 1.0  # reset scale for CPU
                continue
            raise
    if pred is None:
        raise RuntimeError("Segmentation inference failed without producing output")
    if infer_scale != 1.0:
        zoom_f = tuple(orig_shape[i] / pred.shape[i] for i in range(3))
        pred = scipy.ndimage.zoom(pred, zoom_f, order=0).astype(np.uint8)
    return pred

def post_process(seg: np.ndarray, brain_mask: Optional[np.ndarray] = None, min_size: int = 100) -> np.ndarray:
    import scipy.ndimage as ndimage
    
    out = seg.copy()

    # 1. Lọc nhiễu bên ngoài bằng Largest Connected Component (LCC) cho Whole Tumor (WT)
    # Loại bỏ các mảnh vỡ nhỏ lẻ loi khỏi khối u chính.
    wt_mask = out > 0
    labeled_wt, num_features = ndimage.label(wt_mask)
    if num_features > 0:
        sizes = ndimage.sum(wt_mask, labeled_wt, range(1, num_features + 1))
        largest_component_id = np.argmax(sizes) + 1
        # Set background for everything outside the largest WT mass
        out[labeled_wt != largest_component_id] = 0

    # Cập nhật lại WT mask sau khi lọc
    wt_mask = out > 0

    # 2. Làm mịn biên và lấp lỗ hổng (Morphological Closing & Fill Holes)
    # Khắc phục tình trạng "xấu", răng cưa và lỗ hổng bên trong của ảnh segmentation.
    struct_elem = ndimage.generate_binary_structure(3, 1) # 3D cross structure
    smoothed_wt = ndimage.binary_closing(wt_mask, structure=struct_elem, iterations=2)
    smoothed_wt = ndimage.binary_fill_holes(smoothed_wt)
    
    # Những pixel được lấp đầy (hỗ hổng/lõm) sẽ được gán mặc định là vùng Edema (label 2)
    new_pixels = smoothed_wt & (~wt_mask)
    out[new_pixels] = 2

    # 3. Xử lý các vùng Enhancing Tumor (ET - label 3) quá nhỏ
    # Theo rule chuẩn của BraTS, nếu ET quá nhỏ (false positive), ta gán nó về Necrotic Core (label 1)
    # thay vì xóa hẳn (biến thành background 0) gây rỗng khối u.
    et_mask = out == 3
    labeled_et, num_et = ndimage.label(et_mask)
    if num_et > 0:
        for i in range(1, num_et + 1):
            if int((labeled_et == i).sum()) < min_size:
                out[labeled_et == i] = 1

    # 4. Áp dụng Brain Mask (nếu có) để đảm bảo không lan ra ngoài não
    if brain_mask is not None and brain_mask.shape == out.shape:
        out[brain_mask == 0] = 0

    return out.astype(np.uint8)
