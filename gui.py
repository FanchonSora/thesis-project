#!/usr/bin/env python3
"""Command‑line interface for the brain tumour pipeline.

Replaces the previous Streamlit GUI – everything now runs in the terminal and
uses matplotlib for simple on‑screen visualization.  You can rebuild a GUI
later if needed; for now the script accepts file paths and prints/logs the
results.

Example:
    python gui.py --case-id case1 \
                 --modality t1=path/to/case1_t1.nii.gz \
                 --modality flair=path/to/case1_flair.nii.gz \
                 --output-dir output --device cpu

"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from run_pipeline import process_case


def parse_args():
    p = argparse.ArgumentParser(description="Run brain tumour pipeline from CLI")
    p.add_argument("--case-id", required=True, help="Identifier for this case")
    p.add_argument(
        "--modality", action="append",
        help="Specify modality file(s) as mod=path, where mod is one of t1,t1ce,t2,flair."
    )
    p.add_argument("--input-dir", help="Directory containing files named <case>_<mod>.nii.gz")
    p.add_argument("--output-dir", help="Output directory")
    p.add_argument(
        "--seg-weights",
        default="segmentaion-module/model-weight/final_model_unet.pth",
        help="Path to segmentation weights"
    )
    p.add_argument(
        "--syn-weights",
        default="synthesis-module/model-weight/epoch_118.pth",
        help="Path to synthesis weights (optional, may be ignored)"
    )
    p.add_argument("--device", default="cpu", choices=["cpu","cuda"],
                   help="Computation device")
    p.add_argument("--roi", nargs=3, type=int, metavar=("X","Y","Z"),
                   help="Sliding‑window ROI size (ignored if MONAI not installed)")
    return p.parse_args()


def build_paths(case_id, args):
    # start with empty paths that may not exist; missing volumes will be handled by process_case
    mods = {m: "" for m in ["t1", "t1ce", "t2", "flair"]}
    if args.input_dir:
        for m in mods:
            mods[m] = os.path.join(args.input_dir, f"{case_id}_{m}.nii.gz")
    if args.modality:
        for entry in args.modality:
            if "=" in entry:
                m, p = entry.split("=", 1)
                if m in mods:
                    mods[m] = p
                else:
                    print(f"[WARN] unknown modality '{m}' ignored")
            else:
                print(f"[WARN] modality argument '{entry}' not in mod=path form")
    return mods


def display_results(report):
    pred_vol = report.get("pred_vol")
    mesh_verts = report.get("mesh_verts")
    mesh_faces = report.get("mesh_faces")

    if pred_vol is not None:
        pred_vol = pred_vol.astype(np.uint8)
        # show central slices
        x0, y0, z0 = [dim // 2 for dim in pred_vol.shape]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(pred_vol[x0, :, :], cmap="gray")
        axes[0].set_title(f"Sagittal @ {x0}")
        axes[1].imshow(pred_vol[:, y0, :], cmap="gray")
        axes[1].set_title(f"Coronal @ {y0}")
        axes[2].imshow(pred_vol[:, :, z0], cmap="gray")
        axes[2].set_title(f"Axial @ {z0}")
        plt.tight_layout()
        plt.show()
    else:
        print("[INFO] no segmentation volume produced")

    if mesh_verts is not None and mesh_faces is not None:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_trisurf(
                mesh_verts[:, 0], mesh_verts[:, 1], mesh_faces, mesh_verts[:, 2],
                color="lightpink", alpha=0.7
            )
            ax.set_title("3D mesh")
            plt.show()
        except Exception as e:
            print(f"[WARN] unable to plot 3D mesh: {e}")

    # print text report
    simple = {k: v for k, v in report.items() if k not in ("pred_vol", "mesh_verts", "mesh_faces")}
    print("\n=== report ===")
    for k, v in simple.items():
        print(f"{k}: {v}")


def main():
    args = parse_args()
    paths = build_paths(args.case_id, args)
    roi = tuple(args.roi) if args.roi else None
    report = process_case(
        args.case_id,
        paths,
        args.output_dir,
        args.seg_weights,
        args.syn_weights,
        device=args.device,
        roi=roi
    )
    display_results(report)


if __name__ == "__main__":
    main()

