from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
try:
    from skimage import measure as sk_measure
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False
BG = "#0b1220"
PANEL = "#111827"
FG = "#f3f4f6"
MUTED = "#9ca3af"
BORDER = "#374151"
ACCENT = "#93c5fd"
REGION_DEF = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}
REGION_COLORS_RGB = {"WT": (52, 211, 153), "TC": (167, 139, 250), "ET": (239, 68, 68)}
BRAIN_SHELL_RGB = (148, 163, 184)

def _load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)

def _remap_brats(seg: np.ndarray) -> np.ndarray:
    out = seg.astype(np.uint8)
    out[out == 4] = 3
    return out

def _resample_to(volume: np.ndarray, target_shape: Tuple[int, int, int], order: int = 0) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)

def _find_case_prediction(rd: Path) -> Path:
    for name in ("*_pred_post.nii.gz", "*_pred.nii.gz", "*_pred_raw.nii.gz"):
        matches = sorted(rd.glob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No prediction NIfTI found in {rd}")

def _pick_brain_image(input_dir: Path) -> Optional[np.ndarray]:
    patterns = ["*flair*.nii.gz", "*t1ce*.nii.gz", "*t1*.nii.gz", "*.nii.gz"]
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            if "seg" in path.name.lower():
                continue
            try:
                return nib.load(str(path)).get_fdata(dtype=np.float32)
            except Exception:
                continue
    return None

def load_pred_and_gt(result_dir: str, input_dir: str):
    rd = Path(result_dir)
    input_path = Path(input_dir)
    pred_path = _find_case_prediction(rd)
    case = pred_path.name.replace("_pred_post.nii.gz", "").replace("_pred.nii.gz", "").replace("_pred_raw.nii.gz", "")
    pred = _remap_brats(_load_nifti(str(pred_path)))
    gt_candidates = sorted(list(input_path.glob("*_seg.nii.gz")) + list(input_path.glob("*-seg.nii.gz")))
    gt = None
    if gt_candidates:
        gt = _remap_brats(_load_nifti(str(gt_candidates[0])))
        if gt.shape != pred.shape:
            pred = _resample_to(pred, gt.shape, order=0).astype(np.uint8)
    brain_img = _pick_brain_image(input_path)
    if brain_img is not None and brain_img.shape != pred.shape:
        brain_img = _resample_to(brain_img, pred.shape, order=1).astype(np.float32)
    return case, pred, gt, brain_img

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for region, labels in REGION_DEF.items():
        gt_mask = np.isin(gt, labels)
        pred_mask = np.isin(pred, labels)
        tp = float(np.logical_and(gt_mask, pred_mask).sum())
        fp = float(np.logical_and(~gt_mask, pred_mask).sum())
        fn = float(np.logical_and(gt_mask, ~pred_mask).sum())
        tn = float(np.logical_and(~gt_mask, ~pred_mask).sum())
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        metrics[region] = {
            "dice": float(dice),
            "precision": float(precision),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }
    return metrics

def _pick_best_slice(seg: np.ndarray) -> Tuple[int, int, int]:
    wt = seg > 0
    sagittal = int(np.argmax(wt.sum(axis=(1, 2))))
    coronal = int(np.argmax(wt.sum(axis=(0, 2))))
    axial = int(np.argmax(wt.sum(axis=(0, 1))))
    return sagittal, coronal, axial

def _norm_slice(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    pos = img[img > 0]
    if pos.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(pos, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)

def _overlay(base: np.ndarray, seg: np.ndarray) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1)
    wt = np.isin(seg, [1, 2, 3])
    tc = np.isin(seg, [1, 3])
    et = seg == 3
    rgb[wt] = 0.60 * rgb[wt] + 0.40 * np.array([0.2, 0.9, 0.6])
    rgb[tc] = 0.55 * rgb[tc] + 0.45 * np.array([0.7, 0.5, 1.0])
    rgb[et] = 0.45 * rgb[et] + 0.55 * np.array([1.0, 0.2, 0.2])
    return np.clip(rgb, 0, 1)

def plot_slices(pred: np.ndarray, gt: Optional[np.ndarray], brain_img: Optional[np.ndarray], case_id: str, out_path: str) -> None:
    s, c, a = _pick_best_slice(gt if gt is not None and np.any(gt > 0) else pred)
    if brain_img is None:
        brain_img = (pred > 0).astype(np.float32)
    views = [
        (brain_img[s, :, :], pred[s, :, :], None if gt is None else gt[s, :, :], "Sagittal"),
        (brain_img[:, c, :], pred[:, c, :], None if gt is None else gt[:, c, :], "Coronal"),
        (brain_img[:, :, a], pred[:, :, a], None if gt is None else gt[:, :, a], "Axial"),
    ]
    n_cols = 3 if gt is not None else 2
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12), facecolor=BG)
    if n_cols == 2:
        axes = np.asarray(axes).reshape(3, 2)
    for row, (img, pred_seg, gt_seg, title) in enumerate(views):
        base = _norm_slice(np.rot90(img))
        axes[row, 0].imshow(base, cmap="gray")
        axes[row, 0].set_title(f"{title} MRI", color=FG)
        axes[row, 1].imshow(_overlay(base, np.rot90(pred_seg)))
        axes[row, 1].set_title(f"{title} Prediction", color=FG)
        if gt is not None and gt_seg is not None:
            axes[row, 2].imshow(_overlay(base, np.rot90(gt_seg)))
            axes[row, 2].set_title(f"{title} Ground Truth", color=FG)
    for ax in fig.axes:
        ax.set_facecolor(PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Tumor Overlay Views — {case_id}", color=ACCENT, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def plot_metrics(metrics: Dict[str, Dict[str, float]], case_id: str, out_path: str) -> None:
    metric_names = ["dice", "precision", "sensitivity", "specificity"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor=BG)
    for ax, metric_name in zip(axes, metric_names):
        values = [metrics[r][metric_name] for r in REGION_DEF]
        bars = ax.bar(list(REGION_DEF.keys()), values, color=["#34d399", "#a78bfa", "#ef4444"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, min(1.05, value + 0.02), f"{value:.3f}", ha="center", color=FG)
        ax.set_title(metric_name.title(), color=FG)
        ax.set_facecolor(PANEL)
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors=MUTED)
    fig.suptitle(f"Segmentation Metrics — {case_id}", color=ACCENT, fontsize=15, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def plot_volumes(report: dict, case_id: str, out_path: str) -> None:
    rv = report.get("region_volumes_mm3") or report.get("region_volumes_voxels", {})
    units = "mm³" if "region_volumes_mm3" in report else "voxels"
    miss = report.get("missing_flags", {})
    regions = ["WT", "TC", "ET"]
    vals = [rv.get(r, 0) for r in regions]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    ax = axes[0]
    bars = ax.barh(regions, vals, color=["#34d399", "#a78bfa", "#ef4444"])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {val:,.1f}", va="center", color=FG)
    ax.set_title(f"Region Volumes ({units})", color=FG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED)
    ax.invert_yaxis()
    ax = axes[1]
    names = list(miss.keys()) if isinstance(miss, dict) else list(("flair", "t1", "t1ce", "t2"))
    availability = [1 - miss[n] for n in names] if isinstance(miss, dict) else [1, 1, 1, 1]
    ax.bar(names, availability, color=["#10b981" if v else "#ef4444" for v in availability])
    for i, avail in enumerate(availability):
        ax.text(i, 0.5, "Present" if avail else "Missing", ha="center", va="center", color="white")
    ax.set_ylim(0, 1.2)
    ax.set_title("Modality Availability", color=FG)
    ax.set_facecolor(PANEL)
    ax.set_yticks([])
    ax.tick_params(colors=MUTED)
    fig.suptitle(f"Pipeline Summary — {case_id}", color=ACCENT, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def _build_mesh(mask: np.ndarray, smooth: int = 2) -> Optional[dict]:
    if not HAS_SKIMAGE or mask.sum() < 50:
        return None
    try:
        verts, faces, _, _ = sk_measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0), allow_degenerate=False)
    except Exception:
        return None
    if HAS_TRIMESH and smooth > 0:
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=smooth)
            verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        except Exception:
            pass
    return {"verts": verts.astype(np.float32), "faces": faces.astype(np.int32)}

def _build_brain_shell_mesh(brain_img: Optional[np.ndarray], seg: np.ndarray, smooth: int = 3) -> Optional[dict]:
    from scipy.ndimage import binary_dilation, binary_fill_holes
    if brain_img is not None and np.any(brain_img > 0):
        thresh = np.percentile(brain_img[brain_img > 0], 15)
        mask = brain_img > thresh
        filled = np.zeros_like(mask)
        for z in range(mask.shape[2]):
            filled[:, :, z] = binary_fill_holes(mask[:, :, z])
        mask = filled.astype(np.uint8)
    else:
        wt = np.isin(seg, [1, 2, 3]).astype(np.uint8)
        if wt.sum() == 0:
            return None
        mask = binary_dilation(wt, iterations=25).astype(np.uint8)
    return _build_mesh(mask, smooth=smooth)

def save_obj(mesh_data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for v in mesh_data["verts"]:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for f in mesh_data["faces"]:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

def build_all_meshes(seg: np.ndarray, tag: str, out_dir: str, case_id: str, brain_img: Optional[np.ndarray] = None) -> Dict[str, Optional[dict]]:
    mesh_dir = Path(out_dir) / f"{case_id}_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    meshes: Dict[str, Optional[dict]] = {}
    for region, labels in REGION_DEF.items():
        mask = np.isin(seg, labels).astype(np.uint8)
        mesh = _build_mesh(mask)
        if mesh is not None:
            save_obj(mesh, str(mesh_dir / f"{tag}_{region}.obj"))
        meshes[region] = mesh
    shell = _build_brain_shell_mesh(brain_img, seg, smooth=4)
    if shell is not None:
        save_obj(shell, str(mesh_dir / f"{tag}_brain_shell.obj"))
    meshes["brain_shell"] = shell
    return meshes

def _mesh3d_trace(mesh_data: dict, color_rgb: Tuple[int, int, int], name: str, opacity: float, col: int):
    v, f = mesh_data["verts"], mesh_data["faces"]
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})",
        opacity=opacity,
        name=name,
        legendgroup=name,
        showlegend=(col == 1),
    )

def build_3d_html(pred_meshes: Dict[str, Optional[dict]], gt_meshes: Optional[Dict[str, Optional[dict]]], case_id: str, out_path: str) -> None:
    if not HAS_PLOTLY:
        return
    has_gt = gt_meshes is not None
    fig = make_subplots(rows=1, cols=2 if has_gt else 1, specs=[[{"type": "scene"}, {"type": "scene"}]] if has_gt else [[{"type": "scene"}]])
    for region, mesh in pred_meshes.items():
        if mesh is None:
            continue
        color = BRAIN_SHELL_RGB if region == "brain_shell" else REGION_COLORS_RGB[region]
        opacity = 0.08 if region == "brain_shell" else 0.7
        fig.add_trace(_mesh3d_trace(mesh, color, region, opacity, 1), row=1, col=1)
    if has_gt and gt_meshes is not None:
        for region, mesh in gt_meshes.items():
            if mesh is None:
                continue
            color = BRAIN_SHELL_RGB if region == "brain_shell" else REGION_COLORS_RGB[region]
            opacity = 0.08 if region == "brain_shell" else 0.7
            trace = _mesh3d_trace(mesh, color, region, opacity, 2)
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)
    fig.update_layout(
        title=f"3D Tumor Mesh — {case_id}",
        paper_bgcolor=BG,
        font=dict(color=FG),
        width=1400 if has_gt else 800,
        height=700,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")

def build_dashboard(case_id: str, result_dir: str, metrics: Optional[Dict], report: dict, has_gt: bool) -> None:
    rd = Path(result_dir)
    parts = [
        f"<h1>{case_id}</h1>",
        f"<p>Status: {report.get('status')}</p>",
        f"<p>Synthesis status: {report.get('synthesis_status', 'unknown')}</p>",
    ]
    for fname in [f"{case_id}_slices.png", f"{case_id}_metrics.png", f"{case_id}_volumes.png"]:
        if (rd / fname).exists():
            parts.append(f"<h3>{fname}</h3><img src='{fname}' style='max-width:100%;border:1px solid #333;' />")
    if (rd / f"{case_id}_mesh3d.html").exists():
        parts.append(f"<h3>3D Mesh</h3><iframe src='{case_id}_mesh3d.html' style='width:100%;height:760px;border:1px solid #333;'></iframe>")
    (rd / "dashboard.html").write_text("\n".join(parts), encoding="utf-8")

def run_visualization(result_dir: str, input_dir: str) -> None:
    rd = Path(result_dir)
    report_files = sorted(rd.glob("*_report.json"))
    report = json.loads(report_files[0].read_text(encoding="utf-8")) if report_files else {}
    case, pred, gt, brain_img = load_pred_and_gt(result_dir, input_dir)
    metrics = compute_metrics(gt, pred) if gt is not None else None
    plot_slices(pred, gt, brain_img, case, str(rd / f"{case}_slices.png"))
    if metrics:
        plot_metrics(metrics, case, str(rd / f"{case}_metrics.png"))
        report["region_metrics"] = metrics
    plot_volumes(report, case, str(rd / f"{case}_volumes.png"))
    pred_meshes = build_all_meshes(pred, "pred", result_dir, case, brain_img)
    gt_meshes = build_all_meshes(gt, "gt", result_dir, case, brain_img) if gt is not None else None
    build_3d_html(pred_meshes, gt_meshes, case, str(rd / f"{case}_mesh3d.html"))
    build_dashboard(case, result_dir, metrics, report, gt is not None)

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pipeline outputs")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    run_visualization(args.result_dir, args.input_dir)

if __name__ == "__main__":
    main()
