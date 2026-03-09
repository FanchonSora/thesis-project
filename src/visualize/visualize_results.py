from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from skimage import measure as sk_measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

SEG_CMAP   = ListedColormap(["none", "#e74c3c", "#f1c40f", "#1abc9c"])
SEG_LABELS = {
    1: "NCR — Necrotic Core",
    2: "ED — Edema",
    3: "ET — Enhancing Tumour",
}
LEGEND_COLORS = {"NCR": "#e74c3c", "ED": "#f1c40f", "ET": "#1abc9c"}
REGION_DEF: Dict[str, List[int]] = {
    "WT":  [1, 2, 3],
    "TC":  [1, 3],
    "ET":  [3],
    "ED":  [2],
    "NCR": [1],
}
REGION_COLORS_RGB = {
    "WT":  (100, 220, 100),
    "TC":  (180, 130, 255),
    "ET":  ( 26, 188, 156),
    "ED":  (241, 196,  15),
    "NCR": (231,  76,  60),
}
BG = "#0d1117"
FG = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#00d4ff"
PANEL  = "#111827"
BORDER = "#1e2d45"

def _load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32).astype(np.uint8)

def _remap_brats(seg: np.ndarray) -> np.ndarray:
    out = seg.copy()
    out[out == 4] = 3
    return out

def _resample_to(volume: np.ndarray, target_shape: Tuple) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=0).astype(np.uint8)

def load_pred_and_gt(result_dir: str, input_dir: str):
    rd   = Path(result_dir)
    case = rd.name
    pred_path = rd / f"{case}_pred.nii.gz"
    if not pred_path.exists():
        matches   = list(rd.glob("*_pred.nii.gz"))
        pred_path = matches[0] if matches else None
    if pred_path is None or not pred_path.exists():
        sys.exit(f"❌  Prediction not found in {rd}")
    pred = _remap_brats(_load_nifti(str(pred_path)))
    print(f"  Prediction : {pred.shape}  labels={np.unique(pred).tolist()}")
    id_dir = Path(input_dir)
    gt_candidates = (
        list(id_dir.glob("*-seg.nii.gz")) +
        list(id_dir.glob("*_seg.nii.gz")) +
        list(id_dir.glob("*-seg.nii"))    +
        list(id_dir.glob("*_seg.nii"))
    )
    if not gt_candidates:
        print("  ⚠️  No GT segmentation found — metrics and GT mesh skipped.")
        return case, pred, None
    gt = _remap_brats(_load_nifti(str(gt_candidates[0])))
    print(f"  GT         : {gt.shape}  labels={np.unique(gt).tolist()}")
    if pred.shape != gt.shape:
        print(f"  ⚠️  Shape mismatch — resampling prediction {pred.shape} → {gt.shape}")
        pred = _resample_to(pred, gt.shape)

    return case, pred, gt

def load_report(result_dir: str, case: str) -> dict:
    p = Path(result_dir) / f"{case}_report.json"
    return json.loads(p.read_text()) if p.exists() else {}

def _dice(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    yt, yp = y_true.astype(bool), y_pred.astype(bool)
    inter  = np.sum(yt & yp)
    union  = np.sum(yt) + np.sum(yp)
    if union == 0:
        return 1.0
    return float((2 * inter + eps) / (union + eps))

def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = y_true.astype(bool), y_pred.astype(bool)
    if not yp.any():
        return 1.0 if not yt.any() else 0.0
    return float(np.sum(yt & yp) / np.sum(yp))

def _sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = y_true.astype(bool), y_pred.astype(bool)
    if not yt.any():
        return 1.0
    return float(np.sum(yt & yp) / np.sum(yt))

def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = y_true.astype(bool), y_pred.astype(bool)
    bg_t, bg_p = ~yt, ~yp
    tn = np.sum(bg_p & bg_t)
    fp = np.sum(~bg_p & bg_t)
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

def compute_region_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for region, labels in REGION_DEF.items():
        gt_mask   = np.isin(gt,   labels).astype(np.uint8)
        pred_mask = np.isin(pred, labels).astype(np.uint8)
        metrics[region] = {
            "dice":        _dice(gt_mask, pred_mask),
            "precision":   _precision(gt_mask, pred_mask),
            "sensitivity": _sensitivity(gt_mask, pred_mask),
            "specificity": _specificity(gt_mask, pred_mask),
        }
    return metrics

def plot_slices(
    pred:     np.ndarray,
    gt:       Optional[np.ndarray],
    case_id:  str,
    out_path: str,
) -> None:
    H, W, D = pred.shape
    slices   = {"Axial": (pred[:, :, D//2], gt[:, :, D//2] if gt is not None else None),
                "Coronal": (pred[:, W//2, :], gt[:, W//2, :] if gt is not None else None),
                "Sagittal": (pred[H//2, :, :], gt[H//2, :, :] if gt is not None else None)}
    n_rows = 2 if gt is not None else 1
    fig    = plt.figure(figsize=(18, 6 * n_rows), facecolor=BG)
    gs     = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.06, wspace=0.04)
    def _draw(ax, img, title):
        ax.imshow(np.ones_like(img) * 0.12, cmap="gray", vmin=0, vmax=1)
        ax.imshow(img, cmap=SEG_CMAP, vmin=0, vmax=3, alpha=0.92, interpolation="nearest")
        ax.set_title(title, color=FG, fontsize=11, pad=5)
        ax.axis("off")
    for col, (plane, (p_sl, g_sl)) in enumerate(slices.items()):
        if gt is not None:
            _draw(fig.add_subplot(gs[0, col]), g_sl, f"{plane}  |  Ground Truth")
            _draw(fig.add_subplot(gs[1, col]), p_sl, f"{plane}  |  Prediction")
        else:
            _draw(fig.add_subplot(gs[0, col]), p_sl, f"{plane}  |  Prediction")
    # Legend
    patches = [mpatches.Patch(color=c, label=lbl)
               for lbl, c in LEGEND_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.3, facecolor=PANEL,
               labelcolor=FG, edgecolor=BORDER, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Segmentation Slices — {case_id}",
                 color=ACCENT, fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def plot_metrics(
    metrics:  Dict[str, Dict[str, float]],
    case_id:  str,
    out_path: str,
) -> None:
    metric_names = ["dice", "precision", "sensitivity", "specificity"]
    metric_labels = ["Dice", "Precision", "Sensitivity", "Specificity"]
    regions = list(metrics.keys())
    bar_colors = ["#00d4ff", "#a78bfa", "#1abc9c", "#e74c3c", "#f1c40f"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.35)
    for ax, mkey, mlabel in zip(axes, metric_names, metric_labels):
        vals  = [metrics[r][mkey] for r in regions]
        bars  = ax.bar(regions, vals, color=bar_colors[:len(regions)],
                       edgecolor="none", width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(val + 0.02, 1.06),
                    f"{val:.3f}",
                    ha="center", va="bottom", color=FG, fontsize=9)
        ax.set_facecolor(PANEL)
        ax.set_title(mlabel, color=FG, fontsize=12)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.spines[:].set_color(BORDER)
        ax.axhline(1.0, color=BORDER, linewidth=0.8, linestyle="--")
    fig.suptitle(f"Segmentation Metrics — {case_id}",
                 color=ACCENT, fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✅ Metrics       → {out_path}")

def plot_volumes(
    report:   dict,
    metrics:  Optional[Dict],
    case_id:  str,
    out_path: str,
) -> None:
    rv       = report.get("region_volumes", {})
    miss_raw = report.get("missing_flags", {})
    dsf      = report.get("downsample_factor", 1.0)
    regions  = ["WT", "TC", "ET", "NCR", "ED"]
    vals     = [rv.get(r, 0) for r in regions]
    colors   = ["#00d4ff", "#a78bfa", "#1abc9c", "#e74c3c", "#f1c40f"]
    # Modality availability
    if isinstance(miss_raw, dict):
        mod_names  = list(miss_raw.keys())
        mod_avail  = [1 - v for v in miss_raw.values()]
    else:
        mod_names  = ["flair", "t1", "t1ce", "t2"]
        mod_avail  = [1 - m for m in miss_raw] if miss_raw else [1, 1, 1, 1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    # ── Region volumes ────────────────────────────────────────────────
    ax = axes[0]
    mx = max(vals) if max(vals) > 0 else 1
    bars = ax.barh(regions, vals, color=colors, edgecolor="none", height=0.55)
    for bar, val in zip(bars, vals):
        ax.text(val + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", color=FG, fontsize=9)
    ax.set_facecolor(PANEL)
    ax.set_title("Region Volumes (voxels)", color=FG, fontsize=12)
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(BORDER)
    ax.set_xlim(0, mx * 1.2)
    ax.invert_yaxis()
    # ── Modality availability ─────────────────────────────────────────
    ax = axes[1]
    bar_c = ["#10b981" if a else "#ef4444" for a in mod_avail]
    ax.bar(mod_names, mod_avail, color=bar_c, edgecolor="none", width=0.5)
    for i, (name, a) in enumerate(zip(mod_names, mod_avail)):
        ax.text(i, 0.5, "✓ Present" if a else "✗ Missing",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.set_facecolor(PANEL)
    ax.set_title("Modality Availability", color=FG, fontsize=12)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(BORDER)
    fig.suptitle(f"Pipeline Summary — {case_id}  (ds={dsf:.3f})",
                 color=ACCENT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✅ Volumes       → {out_path}")

def _build_mesh(mask: np.ndarray, smooth: int = 2) -> Optional[dict]:
    if not HAS_SKIMAGE or mask.sum() < 50:
        return None
    try:
        verts, faces, _, _ = sk_measure.marching_cubes(
            mask.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False,
        )
    except Exception as exc:
        print(f"    marching_cubes failed: {exc}")
        return None
    if HAS_TRIMESH and smooth > 0:
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=smooth)
            verts, faces = np.array(mesh.vertices), np.array(mesh.faces)
        except Exception:
            pass
    return {"verts": verts.astype(np.float32), "faces": faces.astype(np.int32)}

def save_obj(mesh_data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for v in mesh_data["verts"]:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for f in mesh_data["faces"]:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

def build_all_meshes(
    seg:      np.ndarray,
    tag:      str,          # "pred" or "gt"
    out_dir:  str,
    case_id:  str,
) -> Dict[str, Optional[dict]]:
    mesh_dir = Path(out_dir) / f"{case_id}_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    meshes: Dict[str, Optional[dict]] = {}
    for region, labels in REGION_DEF.items():
        mask = np.isin(seg, labels).astype(np.uint8)
        m    = _build_mesh(mask)
        if m is not None:
            obj_path = str(mesh_dir / f"{tag}_{region}.obj")
            save_obj(m, obj_path)
            print(f"    {tag}/{region}: {len(m['verts'])} verts → {Path(obj_path).name}")
        meshes[region] = m
    return meshes

def _mesh3d_trace(
    mesh_data: dict,
    color_rgb: Tuple[int, int, int],
    name:      str,
    opacity:   float = 0.70,
    col:       int   = 1,       # subplot column (1=pred, 2=gt)
) -> "go.Mesh3d":
    r, g, b = color_rgb
    v, f    = mesh_data["verts"], mesh_data["faces"]
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=f"rgb({r},{g},{b})",
        opacity=opacity,
        name=name,
        legendgroup=name,
        showlegend=(col == 1),   # show legend entry only once
        hovertemplate=f"<b>{name}</b><extra></extra>",
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.85, specular=0.25, roughness=0.5),
        lightposition=dict(x=200, y=300, z=200),
    )

def build_3d_html(
    pred_meshes: Dict[str, Optional[dict]],
    gt_meshes:   Optional[Dict[str, Optional[dict]]],
    case_id:     str,
    out_path:    str,
) -> None:
    if not HAS_PLOTLY:
        print("  ⚠️  plotly not installed — skipping 3D HTML  (pip install plotly)")
        return
    has_gt   = gt_meshes is not None
    n_cols   = 2 if has_gt else 1
    titles   = (["Prediction", "Ground Truth"] if has_gt else ["Prediction"])
    subplot_types = [{"type": "scene"}] * n_cols
    fig = make_subplots(
        rows=1, cols=n_cols,
        specs=[subplot_types],
        subplot_titles=titles,
        horizontal_spacing=0.04,
    )
    # ── Add pred traces (col 1) ───────────────────────────────────────
    for region, m in pred_meshes.items():
        if m is None:
            continue
        trace = _mesh3d_trace(m, REGION_COLORS_RGB[region],
                               name=region, col=1)
        fig.add_trace(trace, row=1, col=1)
    # ── Add GT traces (col 2) ─────────────────────────────────────────
    if has_gt:
        for region, m in gt_meshes.items():
            if m is None:
                continue
            trace = _mesh3d_trace(m, REGION_COLORS_RGB[region],
                                   name=region, col=2)
            trace.legendgroup    = region
            trace.showlegend     = False
            fig.add_trace(trace, row=1, col=2)
    # ── Layout ───────────────────────────────────────────────────────
    scene_cfg = dict(
        xaxis=dict(backgroundcolor=BG, gridcolor=BORDER, color=MUTED, showticklabels=False),
        yaxis=dict(backgroundcolor=BG, gridcolor=BORDER, color=MUTED, showticklabels=False),
        zaxis=dict(backgroundcolor=BG, gridcolor=BORDER, color=MUTED, showticklabels=False),
        bgcolor=BG,
        aspectmode="data",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
    )
    fig.update_layout(
        title=dict(text=f"3D Tumour Mesh — {case_id}",
                   font=dict(color=ACCENT, size=16), x=0.5),
        paper_bgcolor=BG,
        font=dict(color=FG),
        scene  = scene_cfg,
        scene2 = scene_cfg,
        legend=dict(
            x=0.5, y=-0.05, xanchor="center", orientation="h",
            bgcolor="rgba(17,24,39,0.8)", bordercolor=BORDER, borderwidth=1,
        ),
        width=1400 if has_gt else 800,
        height=700,
        margin=dict(l=0, r=0, t=60, b=60),
    )
    # Subplot title colour
    for ann in fig.layout.annotations:
        ann.font.color = FG
        ann.font.size  = 13
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  ✅ 3D HTML       → {out_path}")

def build_dashboard(
    case_id:     str,
    result_dir:  str,
    metrics:     Optional[Dict],
    report:      dict,
    has_gt:      bool,
) -> str:
    rv   = report.get("region_volumes", {})
    dsf  = report.get("downsample_factor", 1.0)
    miss = report.get("missing_flags", {})

    def _img(fname, caption="", width="100%"):
        return f"""
        <figure style="margin:0">
          <img src="{fname}" style="width:{width};border-radius:8px;border:1px solid {BORDER}">
          <figcaption style="color:{MUTED};font-size:12px;margin-top:4px">{caption}</figcaption>
        </figure>"""

    def _metric_table(m: dict) -> str:
        header = "<tr>" + "".join(f"<th>{c}</th>" for c in ["Region", "Dice", "Precision", "Sensitivity", "Specificity"]) + "</tr>"
        rows   = ""
        for region, vals in m.items():
            d = vals['dice']; p = vals['precision']; s = vals['sensitivity']; sp = vals['specificity']
            color = "#10b981" if d >= 0.8 else ("#f59e0b" if d >= 0.5 else "#ef4444")
            rows += (f"<tr><td><b>{region}</b></td>"
                     f"<td style='color:{color}'>{d:.4f}</td>"
                     f"<td>{p:.4f}</td><td>{s:.4f}</td><td>{sp:.4f}</td></tr>")
        return f"<table>{header}{rows}</table>"

    def _vol_table() -> str:
        rows = ""
        for k, v in rv.items():
            rows += f"<tr><td><b>{k}</b></td><td>{v:,}</td></tr>"
        return f"<table><tr><th>Region</th><th>Voxels</th></tr>{rows}</table>"
    metrics_section = ""
    if metrics:
        metrics_section = f"""
        <section>
          <h2>📊 Segmentation Metrics</h2>
          {_metric_table(metrics)}
          {_img(f"{case_id}_metrics.png")}
        </section>"""
    mesh_section = f"""
    <section>
      <h2>🧠 3D Mesh Comparison (Pred vs GT)</h2>
      <iframe src="{case_id}_mesh3d.html"
              width="100%" height="720"
              style="border:1px solid {BORDER};border-radius:8px;background:{BG}">
      </iframe>
    </section>"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Brain Tumour — {case_id}</title>
  <style>
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
    body   {{ background:{BG}; color:{FG}; font-family: 'Segoe UI', system-ui, sans-serif; padding:24px; }}
    h1     {{ color:{ACCENT}; font-size:1.8rem; margin-bottom:16px; }}
    h2     {{ color:{FG}; font-size:1.2rem; margin:24px 0 12px; border-bottom:1px solid {BORDER}; padding-bottom:6px; }}
    section {{ margin-bottom:32px; }}
    table  {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid {BORDER}; }}
    th     {{ color:{MUTED}; font-weight:600; background:{PANEL}; }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .grid  {{ display:grid; gap:16px; }}
    .grid-2 {{ grid-template-columns: 1fr 1fr; }}
    .card  {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:16px; }}
    .tag-ok  {{ background:#065f46; color:#6ee7b7; padding:2px 8px; border-radius:4px; font-size:11px; }}
    .tag-err {{ background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:4px; font-size:11px; }}
    figcaption {{ text-align:center; }}
    @media (max-width:900px) {{ .grid-2 {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <h1>🧠 Brain Tumour Analysis — <span style="color:{FG}">{case_id}</span></h1>
  <div class="grid grid-2">
    <div class="card">
      <h2 style="margin-top:0">Pipeline Info</h2>
      <table>
        <tr><td>Case ID</td><td><b>{case_id}</b></td></tr>
        <tr><td>Status</td><td><span class="tag-ok">{report.get('status','?')}</span></td></tr>
        <tr><td>Downsample factor</td><td>{dsf:.4f}</td></tr>
      </table>
    </div>
    <div class="card">
      <h2 style="margin-top:0">Modality Availability</h2>
      <table>
        {''.join(f"<tr><td>{k.upper()}</td><td><span class=\"{'tag-ok' if not v else 'tag-err'}\">{'Present' if not v else 'Missing'}</span></td></tr>" for k,v in (miss.items() if isinstance(miss,dict) else zip(['flair','t1','t1ce','t2'],miss)))}
      </table>
    </div>
  </div>
  <section>
    <h2>🔬 Slice Comparison {"(GT vs Prediction)" if has_gt else "(Prediction)"}</h2>
    {_img(f"{case_id}_slices.png")}
  </section>
  {metrics_section}
  <section>
    <h2>📦 Region Volumes</h2>
    <div class="grid grid-2">
      <div>{_vol_table()}</div>
      <div>{_img(f"{case_id}_volumes.png")}</div>
    </div>
  </section>
  {mesh_section}
  <footer style="margin-top:48px;color:{MUTED};font-size:12px;text-align:center">
    Generated by Brain Tumour Segmentation Pipeline
  </footer>
</body>
</html>"""
    out_path = str(Path(result_dir) / "dashboard.html")
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"  ✅ Dashboard     → {out_path}")
    return out_path

def run_visualization(result_dir: str, input_dir: str) -> None:
    rd = Path(result_dir)
    if not rd.exists():
        sys.exit(f"❌  result-dir not found: {rd}")
    # ── Load data ─────────────────────────────────────────────────────
    case, pred, gt = load_pred_and_gt(result_dir, input_dir)
    report         = load_report(result_dir, case)
    has_gt         = gt is not None
    # ── Metrics ───────────────────────────────────────────────────────
    metrics: Optional[Dict] = None
    if has_gt:
        print("\n[1/5] Computing metrics …")
        metrics = compute_region_metrics(gt, pred)
        print("  Metrics:")
        for region, m in metrics.items():
            print(f"    {region:3s}: Dice={m['dice']:.4f}  "
                  f"Prec={m['precision']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  "
                  f"Spec={m['specificity']:.4f}")
        report["region_metrics"] = metrics
    else:
        print("\n[1/5] No GT found — skipping metrics.")
    # ── Slice plots ───────────────────────────────────────────────────
    print("\n[2/5] Generating slice comparison …")
    plot_slices(pred, gt, case, str(rd / f"{case}_slices.png"))
    # ── Metrics plot ──────────────────────────────────────────────────
    if metrics:
        print("\n[3/5] Generating metrics chart …")
        plot_metrics(metrics, case, str(rd / f"{case}_metrics.png"))
    else:
        print("\n[3/5] Metrics chart skipped (no GT).")
    # ── Volume / availability plot ────────────────────────────────────
    print("\n[4/5] Generating volume chart …")
    plot_volumes(report, metrics, case, str(rd / f"{case}_volumes.png"))
    # ── 3D meshes (pred + gt side-by-side) ────────────────────────────
    print("\n[5/5] Building 3D meshes …")
    if HAS_SKIMAGE:
        print("  Building Prediction meshes …")
        pred_meshes = build_all_meshes(pred, "pred", result_dir, case)
        gt_meshes   = None
        if has_gt:
            print("  Building GT meshes …")
            gt_meshes = build_all_meshes(gt, "gt", result_dir, case)
        build_3d_html(pred_meshes, gt_meshes, case, str(rd / f"{case}_mesh3d.html"))
    else:
        print("  ⚠️  scikit-image not installed — skipping mesh generation  (pip install scikit-image)")
    build_dashboard(case, result_dir, metrics, report, has_gt)
    print(f"\n✅ All outputs written to: {rd}")
    print("   Files:")
    for f in sorted(rd.iterdir()):
        print(f"   {f.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualise brain-tumour segmentation pipeline outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-dir", required=True,
                        help="Case output folder (contains *_pred.nii.gz, *_report.json)")
    parser.add_argument("--input-dir",  required=True,
                        help="Original NIfTI input folder (for GT segmentation)")
    args = parser.parse_args()
    run_visualization(args.result_dir, args.input_dir)

if __name__ == "__main__":
    main()