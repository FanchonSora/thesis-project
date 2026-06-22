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
    """
    Post-process segmentation predictions using morphological operations and connected component analysis.
    
    This function applies a series of refinement algorithms to clean up raw segmentation output:
    - Remove false positives (noise fragments outside the main tumor)
    - Smooth boundaries and fill internal holes
    - Handle small/fragmented tumor regions
    - Apply anatomical constraints (brain boundary mask)
    
    Args:
        seg: Raw segmentation output from neural network (labels: 0=background, 1=necrotic core, 
             2=edema, 3=enhancing tumor)
        brain_mask: Binary brain mask to constrain output (optional)
        min_size: Minimum voxel count for enhancing tumor regions (smaller regions are reclassified)
    
    Returns:
        Refined segmentation prediction as uint8 array
    """
    import scipy.ndimage as ndimage
    
    out = seg.copy()

    # ============================================================================
    # STEP 1: Largest Connected Component (LCC) filtering
    # ============================================================================
    # Purpose: Remove small false positive tumor fragments isolated from the main tumor mass
    # This ensures we keep only the largest, most anatomically coherent tumor region
    #
    # Algorithm:
    #   1. Create binary mask of all tumor voxels (any label > 0)
    #   2. Label connected components in 3D space
    #   3. Find the largest component by voxel count
    #   4. Zero out all voxels not in the largest component
    
    wt_mask = out > 0  # Binary mask: tumor voxels (label 1, 2, or 3)
    labeled_wt, num_features = ndimage.label(wt_mask)  # Label connected components
    
    if num_features > 0:
        # Calculate size of each connected component
        sizes = ndimage.sum(wt_mask, labeled_wt, range(1, num_features + 1))
        largest_component_id = np.argmax(sizes) + 1
        
        # Keep only the largest component; remove noise fragments
        out[labeled_wt != largest_component_id] = 0
        LOGGER.debug("[post] LCC filtering: kept largest tumor component (size=%d voxels), removed %d fragments",
                     int(sizes[largest_component_id - 1]), num_features - 1)

    # Update WT mask after LCC filtering
    wt_mask = out > 0

    # ============================================================================
    # STEP 2: Morphological smoothing and hole filling
    # ============================================================================
    # Purpose: Smooth jagged tumor boundaries and fill internal cavities/gaps
    # This addresses neural network artifacts like discontinuous tumor regions or noisy edges
    #
    # Algorithm:
    #   1. Binary closing: dilate then erode to close small gaps (erosion followed by dilation)
    #   2. Fill holes: fill all internal cavities that don't connect to image boundary
    #   3. Label newly filled voxels as Edema (label 2) by default
    #
    # Why Edema for filled regions:
    # - Filled voxels are likely internal tumor structure, not air/necrosis
    # - Edema is the surrounding infiltration zone, appropriate default label
    
    struct_elem = ndimage.generate_binary_structure(3, 1)  # 3D 6-connectivity (cross-shaped kernel)
    smoothed_wt = ndimage.binary_closing(wt_mask, structure=struct_elem, iterations=2)
    smoothed_wt = ndimage.binary_fill_holes(smoothed_wt)
    
    # Identify newly filled voxels (those in smoothed mask but not in original)
    new_pixels = smoothed_wt & (~wt_mask)
    out[new_pixels] = 2  # Assign filled voxels to Edema region
    LOGGER.debug("[post] Morphological smoothing: filled %d voxels", int(new_pixels.sum()))

    # ============================================================================
    # STEP 3: Remove small enhancing tumor (ET) false positives
    # ============================================================================
    # Purpose: Filter out tiny ET fragments that are likely artifacts (neural network false positives)
    # BraTS standard practice: very small ET regions are unrealistic and should be reclassified
    #
    # Algorithm:
    #   1. Extract all Enhancing Tumor voxels (label 3)
    #   2. Label connected components within ET mask
    #   3. For each ET component smaller than min_size threshold:
    #      - Reclassify as Necrotic Core (label 1) instead of deleting to background
    #      - Rationale: small ET fragments likely represent unclear tumor boundaries,
    #        not empty space; preserving as label 1 maintains tumor coherence
    #
    # Note: Using label 1 (necrotic core) preserves the tumor mass without artificial gaps
    
    et_mask = out == 3  # Extract enhancing tumor voxels
    labeled_et, num_et = ndimage.label(et_mask)
    
    if num_et > 0:
        et_sizes = ndimage.sum(et_mask, labeled_et, range(1, num_et + 1))
        small_et_count = 0
        
        for component_id in range(1, num_et + 1):
            component_size = int(et_sizes[component_id - 1])
            
            if component_size < min_size:
                # Reclassify small ET fragment as Necrotic Core (label 1)
                out[labeled_et == component_id] = 1
                small_et_count += 1
        
        LOGGER.debug("[post] ET filtering: reclassified %d small ET components (threshold=%d voxels)",
                     small_et_count, min_size)

    # ============================================================================
    # STEP 4: Apply brain mask constraint
    # ============================================================================
    # Purpose: Ensure segmentation stays within anatomical brain boundaries
    # This prevents tumor predictions from leaking into skull/background regions
    #
    # Algorithm:
    #   1. If brain mask is provided and matches segmentation shape:
    #      - Zero out all segmentation voxels where brain_mask == 0
    #      - This enforces anatomical constraint: no tumor outside brain tissue
    
    if brain_mask is not None and brain_mask.shape == out.shape:
        out[brain_mask == 0] = 0
        LOGGER.debug("[post] Applied brain mask: zeroed %d voxels outside brain",
                     int((~brain_mask).sum()))

    return out.astype(np.uint8)
